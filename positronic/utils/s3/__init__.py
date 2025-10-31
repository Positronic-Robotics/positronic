"""S3 Mirror - Context manager for syncing directories with S3."""

import itertools
import logging
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

logger = logging.getLogger(__name__)


class null_tqdm(nullcontext):
    def update(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self


def _parse_s3_url(s3_url: str) -> tuple[str, str]:
    """
    Parse S3 URL into bucket and key.

    Args:
        s3_url: S3 URL like 's3://bucket/prefix/path'

    Returns:
        (bucket, key) tuple
    """
    parsed = urlparse(s3_url)
    if parsed.scheme != 's3':
        raise ValueError(f'Not an S3 URL: {s3_url}')

    return parsed.netloc, parsed.path.lstrip('/')


def _is_s3_path(path: str) -> bool:
    return path.startswith('s3://')


def _process_futures(futures, operation: str):
    """Process futures and log errors."""
    for future in futures:
        try:
            future.result()
        except Exception as e:
            logger.error(f'{operation} failed: {e}')


def _needs_download(s3_obj: dict, local_path: Path) -> bool:
    """Check if file needs to be downloaded."""
    # Could also check ETag here for more robust comparison
    # For now, matching size is sufficient
    return not local_path.exists() or local_path.stat().st_size != s3_obj['Size']


def _needs_upload(local_path: Path, s3_obj: dict | None) -> bool:
    """Check if file needs to be uploaded."""
    # Could also check modification time or compute ETag
    # For now, matching size is sufficient
    return s3_obj is None or local_path.stat().st_size != s3_obj['Size']


def _rglob(path: Path) -> Iterator[Path]:
    """Recursively glob files, handling both files and directories."""
    if path.is_file():
        yield path
    else:
        yield from (p for p in path.rglob('*') if p.is_file())


@dataclass
class Options:
    """Configuration for Mirror sync behavior"""

    cache_root: str = '~/.cache/positronic/s3/'
    show_progress: bool = True  # tqdm progress bars in MB
    max_workers: int = 10  # parallel upload/download workers

    def _generate_cache_path(self, s3_url: str) -> Path:
        """
        Generate local cache path from S3 URL.

        Args:
            s3_url: S3 URL like 's3://bucket/prefix/path'

        Returns:
            Local path like ~/.cache/positronic/s3/bucket/prefix/path
        """
        bucket, key = _parse_s3_url(s3_url)
        cache_path = Path(self.cache_root).expanduser().absolute() / bucket / key
        return cache_path


@dataclass
class Download:
    """Spec for downloading from S3 to local"""

    remote: str  # S3 path (s3://bucket/key) or a local path
    local: str | None = None  # Local path, or None to auto-generate in cache

    @property
    def is_remote(self):
        return _is_s3_path(self.remote)

    def local_path(self, options: Options):
        if self.is_remote:
            if self.local:
                return Path(self.local).expanduser().absolute()
            else:
                return options._generate_cache_path(self.remote)
        else:
            path = Path(self.remote).expanduser().absolute()
            if not path.exists():
                raise FileNotFoundError(f'Local download path does not exist: {path}')
            return path


@dataclass
class Upload:
    """Spec for uploading from local to S3 with sync"""

    remote: str  # S3 path (s3://bucket/key) or a local path
    local: str | None = None  # Local path, or None to create temp dir
    interval: int | None = 300  # Background sync interval (seconds), None = sync only on exit
    delete: bool = True  # Delete S3 files not present locally (true sync)

    @property
    def is_remote(self):
        return _is_s3_path(self.remote)

    def local_path(self, options: Options):
        if self.is_remote:
            if self.local:
                return Path(self.local).expanduser().absolute()
            else:
                return options._generate_cache_path(self.remote)
        else:
            return Path(self.remote).expanduser().absolute()


class Mirror:
    """
    Context manager for syncing directories with S3.

    Usage:
        with Mirror(
            dataset=Download('s3://bucket/data'),
            checkpoints=Upload('s3://bucket/outputs', local='./outputs')
        ) as m:
            train(m.dataset, m.checkpoints)
    """

    def __init__(self, options: Options | None = None, **paths):
        """
        Args:
            options: Configuration options (uses Options() defaults if None)
            **paths: Named Download/Upload specs
        """
        self.options = options or Options()
        self.paths = paths

        # Expand cache root
        self.cache_root = Path(self.options.cache_root).expanduser().absolute()
        self.cache_root.mkdir(parents=True, exist_ok=True)

        # S3 client
        self.s3_client = boto3.client('s3')

        # State tracking
        self._local_paths: dict[str, Path] = {}
        self._downloads: dict[str, Download] = {}
        self._uploads: dict[str, Upload] = {}
        self._sync_thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None

        # Separate downloads and uploads
        for name, spec in paths.items():
            if isinstance(spec, Download):
                self._downloads[name] = spec
            elif isinstance(spec, Upload):
                self._uploads[name] = spec
            else:
                raise TypeError(f"Path spec for '{name}' must be Download or Upload, got {type(spec)}")

        # Validate no conflicts: S3 paths cannot overlap between download and upload
        # Only check S3 paths (local paths are allowed to overlap)
        download_s3_paths = [spec.remote for spec in self._downloads.values() if spec.is_remote]
        upload_s3_paths = [spec.remote for spec in self._uploads.values() if spec.is_remote]

        conflicts = self._check_s3_path_conflicts(download_s3_paths, upload_s3_paths)
        if conflicts:
            raise ValueError(f'S3 paths conflict (overlapping directories or exact matches): {conflicts}')

    def __enter__(self):
        """Download all Download specs in parallel and start background upload sync"""
        # Prepare download tasks (only for S3 paths)
        download_tasks = []
        for name, download_spec in self._downloads.items():
            local_path = download_spec.local_path(self.options)
            self._local_paths[name] = local_path
            if download_spec.is_remote:
                download_tasks.append((name, download_spec.remote, local_path))

        # Download all S3 files in parallel
        if download_tasks:
            self._download_all_parallel(download_tasks)

        # Initialize Upload specs
        for name, upload_spec in self._uploads.items():
            local_path = upload_spec.local_path(self.options)
            local_path.mkdir(parents=True, exist_ok=True)
            self._local_paths[name] = local_path

        # Start single background sync thread if any S3 upload has an interval
        has_background_sync = any(spec.interval is not None and spec.is_remote for spec in self._uploads.values())
        if has_background_sync:
            self._start_background_sync()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop background thread and perform final sync"""
        # Stop background sync thread if running
        if self._stop_event:
            self._stop_event.set()

        if self._sync_thread:
            self._sync_thread.join(timeout=60)

        # Final sync for all S3 uploads in parallel
        # Skip local-only uploads (no sync needed)
        upload_tasks = []
        for name, upload_spec in self._uploads.items():
            if upload_spec.is_remote:
                local_path = self._local_paths[name]
                if local_path.exists():
                    upload_tasks.append((name, upload_spec.remote, local_path, upload_spec.delete))

        if upload_tasks:
            self._upload_all_parallel(upload_tasks)

        # Don't suppress exceptions
        return False

    def __getattr__(self, name: str) -> Path:
        """Return local path for named spec"""
        if name in self._local_paths:
            return self._local_paths[name]
        raise AttributeError(f"No path spec named '{name}'")

    # Path detection and parsing
    @staticmethod
    def _s3_path_contains(left: str, right: str) -> bool:
        """Check if one S3 path contains another."""
        # Normalize paths (remove trailing slashes)
        left_norm = left.rstrip('/')
        right_norm = right.rstrip('/')

        # Check if child starts with parent + '/'
        # This ensures s3://bucket/data contains s3://bucket/data/subset
        # but s3://bucket/data does NOT contain s3://bucket/data-new
        return (
            left_norm == right_norm or right_norm.startswith(left_norm + '/') or left_norm.startswith(right_norm + '/')
        )

    @staticmethod
    def _check_s3_path_conflicts(download_paths: list[str], upload_paths: list[str]) -> list[tuple[str, str]]:
        """
        Check for conflicts between download and upload S3 paths.

        A conflict occurs when:
        1. Exact match: same path used for both download and upload
        2. Subdirectory: one path is a subdirectory of the other

        Args:
            download_paths: List of download S3 paths
            upload_paths: List of upload S3 paths

        Returns:
            List of conflicting path pairs (download, upload)
        """
        conflicts = []

        for dl_path, ul_path in itertools.product(download_paths, upload_paths):
            if Mirror._s3_path_contains(dl_path, ul_path):
                conflicts.append((dl_path, ul_path))

        return conflicts

    # Download implementation

    def _list_s3_objects(self, bucket: str, key: str):
        """
        List all S3 objects under a path. Handles both files and directories.

        Yields S3 object metadata dicts. May yield:
        - Nothing if path doesn't exist
        - Single item if key is a file
        - Multiple items if key is a directory/prefix
        """
        # First try to get it as a single file
        try:
            obj = self.s3_client.head_object(Bucket=bucket, Key=key)
            # Create new dict to match paginator format
            yield {**obj, 'Key': key}
            return
        except ClientError as e:
            if e.response['Error']['Code'] != '404':
                raise

        # Not a single file, list as directory/prefix
        paginator = self.s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=key):
            yield from page.get('Contents', [])

    def _progress_bar(self, total_bytes: int, desc: str):
        return (
            tqdm(total=total_bytes, unit='B', unit_scale=True, unit_divisor=1024, desc=desc)
            if self.options.show_progress
            else null_tqdm()
        )

    def _download_all_parallel(self, download_tasks: list[tuple[str, str, Path]]):
        """Download all downloads in parallel."""
        # Collect all files to download from all S3 paths
        all_downloads = []  # List of (bucket, obj, local_path)
        total_bytes = 0

        for _name, s3_url, local_path in download_tasks:
            bucket, key = _parse_s3_url(s3_url)

            # List all objects (handles both files and directories)
            for obj in self._list_s3_objects(bucket, key):
                relative_key = obj['Key'][len(key) :].lstrip('/')
                file_local_path = local_path / relative_key if relative_key else local_path

                if _needs_download(obj, file_local_path):
                    all_downloads.append((bucket, obj, file_local_path))
                    total_bytes += obj['Size']

        if all_downloads:
            with (
                self._progress_bar(total_bytes, 'Downloading') as pbar,
                ThreadPoolExecutor(max_workers=self.options.max_workers) as executor,
            ):
                futures = [executor.submit(self._download_file, *args, pbar) for args in all_downloads]
                _process_futures(as_completed(futures), 'Download')

    def _download_file(self, bucket: str, obj: dict, local_path: Path, pbar: tqdm):
        """Download a single file from S3."""
        # Create parent directory
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download with progress callback
        try:
            self.s3_client.download_file(bucket, obj['Key'], str(local_path), Callback=pbar.update)
        except Exception as e:
            logger.error(f'Failed to download {obj["Key"]}: {e}')
            raise

    # Upload implementation

    def _upload_all_parallel(self, upload_tasks: list[tuple[str, str, Path, bool]]):
        """
        Upload all uploads in parallel.

        Args:
            upload_tasks: List of (name, s3_url, local_path, delete) tuples
        """
        # Collect all files to upload from all local paths
        all_uploads = []  # List of (local_path, bucket, s3_key)
        all_deletes = []  # List of (bucket, s3_key)
        total_bytes = 0

        for _name, s3_url, local_path, delete in upload_tasks:
            bucket, prefix = _parse_s3_url(s3_url)

            # List local files (handles both files and directories)
            local_files = {file_path.relative_to(local_path).as_posix() for file_path in _rglob(local_path)}
            s3_objects = {obj['Key'][len(prefix) :].lstrip('/'): obj for obj in self._list_s3_objects(bucket, prefix)}

            # Determine files to upload
            for relative_path in local_files:
                file_local_path = local_path / relative_path
                if _needs_upload(file_local_path, s3_objects.get(relative_path)):
                    s3_key = prefix + ('/' + relative_path if relative_path != '.' else '')
                    all_uploads.append((file_local_path, bucket, s3_key))
                    total_bytes += file_local_path.stat().st_size

            # Determine files to delete
            if delete:
                for relative_path in set(s3_objects.keys()) - local_files:
                    s3_key = prefix + ('/' + relative_path if relative_path != '.' else '')
                    all_deletes.append((bucket, s3_key))

        # Upload all files in parallel with single progress bar
        if all_uploads:
            with (
                self._progress_bar(total_bytes, 'Uploading') as pbar,
                ThreadPoolExecutor(max_workers=self.options.max_workers) as executor,
            ):
                futures = [executor.submit(self._upload_file, *args, pbar) for args in all_uploads]
                _process_futures(as_completed(futures), 'Upload')

        if all_deletes:  # Delete files in parallel
            with ThreadPoolExecutor(max_workers=self.options.max_workers) as executor:
                futures = [executor.submit(self._delete_file, *args) for args in all_deletes]

                iter = as_completed(futures)
                if self.options.show_progress:
                    iter = tqdm(iter, total=len(all_deletes), desc='Deleting')

                _process_futures(iter, 'Delete')

    def _upload_file(self, local_path: Path, bucket: str, s3_key: str, pbar: tqdm):
        """Upload a single file to S3."""
        try:
            self.s3_client.upload_file(str(local_path), bucket, s3_key, Callback=pbar.update)
        except Exception as e:
            logger.error(f'Failed to upload {local_path} to {s3_key}: {e}')
            raise

    def _delete_file(self, bucket: str, s3_key: str):
        """Delete a file from S3."""
        try:
            self.s3_client.delete_object(Bucket=bucket, Key=s3_key)
        except Exception as e:
            logger.error(f'Failed to delete {s3_key}: {e}')
            raise

    # Background sync

    def _start_background_sync(self):
        """Start single background sync thread for all uploads. Syncs each upload based on its interval."""
        self._stop_event = threading.Event()

        thread = threading.Thread(target=self._background_sync_worker, daemon=True)
        thread.start()
        self._sync_thread = thread

    def _background_sync_worker(self):
        """
        Background worker that syncs all uploads periodically.
        Each upload is synced according to its own interval.
        """
        # Track last sync time for each upload
        last_sync = dict.fromkeys(self._uploads.keys(), 0.0)

        while not self._stop_event.wait(1):  # Check every second
            current_time = time.time()

            # Collect all uploads that need syncing
            upload_tasks = []
            for name, upload_spec in self._uploads.items():
                if upload_spec.interval is not None and upload_spec.is_remote:
                    time_since_last = current_time - last_sync[name]  # Check if it's time to sync this upload
                    if time_since_last >= upload_spec.interval:
                        local_path = self._local_paths[name]
                        upload_tasks.append((name, upload_spec.remote, local_path, upload_spec.delete))
                        last_sync[name] = current_time

            self._upload_all_parallel(upload_tasks)


__all__ = ['Mirror', 'Download', 'Upload', 'Options', '_parse_s3_url']
