"""Global S3 mirror API with dynamic registration."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

logger = logging.getLogger(__name__)


class _NullTqdm(nullcontext):
    def update(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - trivial
        pass

    def __enter__(self) -> _NullTqdm:
        return self


def _parse_s3_url(s3_url: str) -> tuple[str, str]:
    parsed = urlparse(s3_url)
    if parsed.scheme != 's3':
        raise ValueError(f'Not an S3 URL: {s3_url}')
    return parsed.netloc, parsed.path.lstrip('/')


def _normalize_s3_url(s3_url: str) -> str:
    bucket, key = _parse_s3_url(s3_url)
    key = key.strip('/')
    return f's3://{bucket}/{key}' if key else f's3://{bucket}'


def _is_s3_path(path: str) -> bool:
    return path.startswith('s3://')


def _s3_paths_conflict(left: str, right: str) -> bool:
    left_norm = left.rstrip('/')
    right_norm = right.rstrip('/')
    if left_norm == right_norm:
        return True
    return left_norm.startswith(right_norm + '/') or right_norm.startswith(left_norm + '/')


def _process_futures(futures, operation: str) -> None:
    for future in futures:
        try:
            future.result()
        except Exception as exc:
            logger.error('%s failed: %s', operation, exc)


@dataclass(frozen=True)
class FileInfo:
    """Represents a file or directory with metadata for sync operations."""

    relative_path: str  # Relative path from root (empty string for root file/dir)
    size: int  # File size in bytes, 0 for directories
    is_dir: bool  # True if this represents a directory


def _scan_local(path: Path) -> Iterator[FileInfo]:
    if not path.exists():
        return

    base = path
    stack = [path]
    # path => relative
    while stack:
        p = stack.pop()
        relative = p.relative_to(base).as_posix() if p != base else ''
        if p.is_dir():
            if relative:
                yield FileInfo(relative_path=relative, size=0, is_dir=True)
            stack.extend(p.iterdir())
        else:
            yield FileInfo(relative_path=relative, size=p.stat().st_size, is_dir=False)


def _compute_sync_diff(source: Iterator[FileInfo], target: Iterator[FileInfo]) -> tuple[list[FileInfo], list[FileInfo]]:
    source_map: dict[str, FileInfo] = {info.relative_path: info for info in source}
    target_map: dict[str, FileInfo] = {info.relative_path: info for info in target}

    to_copy, to_delete = [], []

    for relative_path, source_info in source_map.items():
        target_info = target_map.get(relative_path)

        if target_info is None:
            to_copy.append(source_info)
        elif source_info.is_dir != target_info.is_dir:
            to_delete.append(target_info)
            to_copy.append(source_info)
        elif not source_info.is_dir and source_info.size != target_info.size:
            to_copy.append(source_info)

    for relative_path, target_info in target_map.items():
        if relative_path not in source_map:
            to_delete.append(target_info)

    return to_copy, to_delete


@dataclass
class _Options:
    cache_root: str = '~/.cache/positronic/s3/'
    show_progress: bool = True
    max_workers: int = 10

    def cache_path_for(self, remote: str) -> Path:
        bucket, key = _parse_s3_url(remote)
        cache_root = Path(self.cache_root).expanduser().resolve()
        return cache_root / bucket / key


@dataclass
class _DownloadRegistration:
    remote: str
    local_path: Path
    delete: bool
    ready: threading.Event = field(default_factory=threading.Event)
    error: Exception | None = None

    def __eq__(self, other):
        if not isinstance(other, _DownloadRegistration):
            return False
        return self.remote == other.remote and self.local_path == other.local_path and self.delete == other.delete


@dataclass
class _UploadRegistration:
    remote: str
    local_path: Path
    interval: int | None
    delete: bool
    sync_on_error: bool
    last_sync: float = 0.0

    def __eq__(self, other):
        if not isinstance(other, _UploadRegistration):
            return False
        return (
            self.remote == other.remote
            and self.local_path == other.local_path
            and self.interval == other.interval
            and self.delete == other.delete
            and self.sync_on_error == other.sync_on_error
        )


_ACTIVE_MIRROR: ContextVar[_Mirror | None] = ContextVar('_ACTIVE_MIRROR', default=None)
_GLOBAL_MIRROR_LOCK = threading.RLock()
_GLOBAL_ACTIVE_MIRROR: _Mirror | None = None


class _Mirror:
    def __init__(self, options: _Options):
        self.options = options
        self.cache_root = Path(self.options.cache_root).expanduser().resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)

        self.s3_client = boto3.client('s3')

        self._downloads: dict[str, _DownloadRegistration] = {}
        self._uploads: dict[str, _UploadRegistration] = {}
        self._lock = threading.RLock()

        self._stop_event: threading.Event | None = None
        self._sync_thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._stop_event is not None

    def start(self) -> None:
        if not self.running:
            self._stop_event = threading.Event()

    def stop(self, had_error: bool = False) -> None:
        if self.running:
            self._stop_event.set()

            if self._sync_thread:
                self._sync_thread.join(timeout=60)
                self._sync_thread = None
            self._final_sync(had_error=had_error)
            self._stop_event = None

    def download(self, remote: str, local: str | Path | None, delete: bool) -> Path:
        if not _is_s3_path(remote):
            path = Path(remote).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f'Local download path does not exist: {path}')
            return path

        normalized = _normalize_s3_url(remote)
        local_path = self.options.cache_path_for(remote) if local is None else Path(local).expanduser().resolve()
        new_registration = _DownloadRegistration(remote=normalized, local_path=local_path, delete=delete)

        with self._lock:
            existing = self._downloads.get(normalized)
            if existing:
                if existing != new_registration:
                    raise ValueError(f"Download for '{normalized}' already registered with different parameters")
                registration = existing
                need_download = False
            else:
                self._check_download_conflicts(normalized)
                self._downloads[normalized] = new_registration
                registration = new_registration
                need_download = True

        if need_download:
            try:
                self._perform_download(normalized, local_path, delete)
            except Exception as exc:
                registration.error = exc
                registration.ready.set()
                with self._lock:
                    self._downloads.pop(normalized, None)
                raise
            else:
                registration.ready.set()
        else:
            registration.ready.wait()
            if registration.error is not None:
                raise registration.error

        return local_path

    def upload(self, remote, local, interval, delete, sync_on_error) -> Path:
        """
        Register (and perform if needed) an upload from a local directory or file to a remote S3 bucket path.

        Args:
            remote (str): Destination S3 URL (e.g., "s3://bucket/key/prefix")
            local (str | Path | None): Local directory or file to upload. If None, determines default from options.
            interval (int | None): If set, enables periodic background uploads (seconds between syncs).
            delete (bool): If True, deletes remote files not present locally.
            sync_on_error (bool): If True, attempts to sync files even when encountering errors.

        Returns:
            Path: The canonical local path associated with this upload registration.

        Raises:
            ValueError: If upload registration conflicts with an existing download or upload or parameters differ.
        """
        if not _is_s3_path(remote):
            path = Path(remote).expanduser().resolve()
            path.mkdir(parents=True, exist_ok=True)
            return path

        normalized = _normalize_s3_url(remote)
        local_path = self.options.cache_path_for(remote) if local is None else Path(local).expanduser().resolve()

        new_registration = _UploadRegistration(
            remote=normalized,
            local_path=local_path,
            interval=interval,
            delete=delete,
            sync_on_error=sync_on_error,
            last_sync=0,
        )

        with self._lock:
            existing = self._uploads.get(normalized)
            if existing:
                if existing != new_registration:
                    raise ValueError(f"Upload for '{normalized}' already registered with different parameters")
                return existing.local_path

            self._check_upload_conflicts(new_registration)
            self._uploads[normalized] = new_registration
            if interval is not None:
                self._ensure_background_thread_unlocked()

        return local_path

    def _check_download_conflicts(self, candidate: str) -> None:
        for upload_remote in self._uploads:
            if _s3_paths_conflict(candidate, upload_remote):
                raise ValueError(f"Conflict: download '{candidate}' overlaps with upload '{upload_remote}'")

    def _check_upload_conflicts(self, new_registration) -> None:
        candidate = new_registration.remote
        for download_remote in self._downloads:
            if _s3_paths_conflict(candidate, download_remote):
                raise ValueError(f"Conflict: upload '{candidate}' overlaps with download '{download_remote}'")
        for upload_remote, reg in self._uploads.items():
            if _s3_paths_conflict(candidate, upload_remote):
                same_remote = candidate == upload_remote
                if not same_remote or reg != new_registration:
                    raise ValueError(f"Conflict: upload '{candidate}' overlaps with upload '{upload_remote}'")

    def _ensure_background_thread_unlocked(self) -> None:
        assert self.running, 'The mirror must be started before performing any uploads'
        if not self._sync_thread or not self._sync_thread.is_alive():
            thread = threading.Thread(target=self._background_worker, name='positronic-s3-sync', daemon=True)
            thread.start()
            self._sync_thread = thread

    def _background_worker(self) -> None:
        while not self._stop_event.wait(1):
            now = time.monotonic()
            due: list[_UploadRegistration] = []
            with self._lock:
                for registration in self._uploads.values():
                    if registration.interval is not None and now - registration.last_sync >= registration.interval:
                        registration.last_sync = now
                        due.append(registration)

            self._sync_uploads(due)

    def _final_sync(self, had_error: bool = False) -> None:
        with self._lock:
            uploads = list(self._uploads.values())
        if had_error:
            uploads = [u for u in uploads if u.sync_on_error]
        self._sync_uploads(uploads)

    def _sync_uploads(self, registrations: Iterable[_UploadRegistration]) -> None:
        tasks: list[tuple[str, Path, bool]] = []
        for registration in registrations:
            if registration.local_path.exists():
                tasks.append((registration.remote, registration.local_path, registration.delete))

        if not tasks:
            return

        to_put: list[tuple[FileInfo, Path, str, str]] = []
        to_remove: list[tuple[str, str]] = []
        total_bytes = 0

        for remote, local_path, delete in tasks:
            bucket, prefix = _parse_s3_url(remote)
            to_copy, to_delete = _compute_sync_diff(_scan_local(local_path), self._scan_s3(bucket, prefix))

            for info in to_copy:
                s3_key = prefix + ('/' + info.relative_path if info.relative_path else '')
                to_put.append((info, local_path, bucket, s3_key))
                total_bytes += info.size

            for info in to_delete if delete else []:
                s3_key = prefix + ('/' + info.relative_path if info.relative_path else '')
                to_remove.append((bucket, s3_key))

        if to_put:
            with (
                self._progress_bar(total_bytes, f'Uploading {remote}') as pbar,
                ThreadPoolExecutor(max_workers=self.options.max_workers) as executor,
            ):
                futures = [
                    executor.submit(self._put_to_s3, info, local_path, bucket, key, pbar)
                    for info, local_path, bucket, key in to_put
                ]
                _process_futures(as_completed(futures), 'Upload')

        if to_remove:
            to_remove_sorted = sorted(to_remove, key=lambda x: x[1].count('/'), reverse=True)
            with ThreadPoolExecutor(max_workers=self.options.max_workers) as executor:
                futures = [executor.submit(self._remove_from_s3, bucket, key) for bucket, key in to_remove_sorted]
                iterator = as_completed(futures)
                if self.options.show_progress:
                    iterator = tqdm(iterator, total=len(to_remove_sorted), desc='Deleting')
                _process_futures(iterator, 'Delete')

    def _perform_download(self, remote: str, local_path: Path, delete: bool) -> None:
        bucket, prefix = _parse_s3_url(remote)
        to_copy, to_delete = _compute_sync_diff(self._scan_s3(bucket, prefix), _scan_local(local_path))

        to_put: list[tuple[FileInfo, str, str, Path]] = []
        to_remove: list[Path] = []
        total_bytes = 0

        for info in to_copy:
            s3_key = prefix + ('/' + info.relative_path if info.relative_path else '')
            to_put.append((info, bucket, s3_key, local_path))
            total_bytes += info.size

        for info in to_delete if delete else []:
            target = local_path / info.relative_path if info.relative_path else local_path
            to_remove.append(target)

        if to_put:
            with (
                self._progress_bar(total_bytes, f'Downloading {remote}') as pbar,
                ThreadPoolExecutor(max_workers=self.options.max_workers) as executor,
            ):
                futures = [executor.submit(self._put_locally, *args, pbar) for args in to_put]
                _process_futures(as_completed(futures), 'Download')

        if to_remove:
            to_remove_sorted = sorted(to_remove, key=lambda x: len(x.parts), reverse=True)
            with ThreadPoolExecutor(max_workers=self.options.max_workers) as executor:
                futures = [executor.submit(self._remove_locally, path) for path in to_remove_sorted]
                iterator = as_completed(futures)
                if self.options.show_progress:
                    iterator = tqdm(iterator, total=len(to_remove_sorted), desc='Deleting')
                _process_futures(iterator, 'Delete')

    def _list_s3_objects(self, bucket: str, key: str) -> Iterator[dict]:
        try:
            obj = self.s3_client.head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            error_code = exc.response['Error']['Code']
            if error_code != '404':
                raise
        else:
            yield {**obj, 'Key': key}
            return

        paginator = self.s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=key):
            yield from page.get('Contents', [])

    def _scan_s3(self, bucket: str, prefix: str) -> Iterator[FileInfo]:
        seen_dirs: set[str] = set()

        for obj in self._list_s3_objects(bucket, prefix):
            key = obj['Key']
            relative = key[len(prefix) :].lstrip('/')

            if key.endswith('/'):
                relative = relative.rstrip('/')
                if relative:
                    yield FileInfo(relative_path=relative, size=0, is_dir=True)
                    seen_dirs.add(relative)
            else:
                yield FileInfo(relative_path=relative, size=obj['Size'], is_dir=False)

                if '/' in relative:
                    parts = relative.split('/')
                    for i in range(len(parts) - 1):
                        dir_path = '/'.join(parts[: i + 1])
                        if dir_path and dir_path not in seen_dirs:
                            yield FileInfo(relative_path=dir_path, size=0, is_dir=True)
                            seen_dirs.add(dir_path)

    def _progress_bar(self, total_bytes: int, desc: str):
        if not self.options.show_progress:
            return _NullTqdm()
        return tqdm(total=total_bytes, unit='B', unit_scale=True, unit_divisor=1024, desc=desc)

    def _put_to_s3(self, info: FileInfo, local_path: Path, bucket: str, key: str, pbar) -> None:
        try:
            if info.is_dir:
                key += '/' if not key.endswith('/') else ''
                self.s3_client.put_object(Bucket=bucket, Key=key, Body=b'')
            else:
                file_path = local_path / info.relative_path if info.relative_path else local_path
                self.s3_client.upload_file(str(file_path), bucket, key, Callback=pbar.update)
        except Exception as exc:
            logger.error('Failed to put %s to %s/%s: %s', local_path, bucket, key, exc)
            raise

    def _remove_from_s3(self, bucket: str, key: str) -> None:
        try:
            self.s3_client.delete_object(Bucket=bucket, Key=key)
        except Exception as exc:
            logger.error('Failed to remove %s/%s: %s', bucket, key, exc)
            raise

    def _put_locally(self, info: FileInfo, bucket: str, key: str, local_path: Path, pbar) -> None:
        try:
            target = local_path / info.relative_path if info.relative_path else local_path
            if info.is_dir:
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                self.s3_client.download_file(bucket, key, str(target), Callback=pbar.update)
        except Exception as exc:
            logger.error('Failed to put %s locally: %s', key, exc)
            raise

    def _remove_locally(self, path: Path) -> None:
        try:
            if path.is_dir():
                path.rmdir()
            else:
                path.unlink()
        except Exception as exc:
            logger.error('Failed to remove %s: %s', path, exc)
            raise


@contextmanager
def mirror(cache_root: str = '~/.cache/positronic/s3/', show_progress: bool = True, max_workers: int = 10):
    global _GLOBAL_ACTIVE_MIRROR
    options = _Options(cache_root=cache_root, show_progress=show_progress, max_workers=max_workers)

    with _GLOBAL_MIRROR_LOCK:
        if _GLOBAL_ACTIVE_MIRROR is not None:
            raise RuntimeError('Mirror already active')

        mirror_obj = _Mirror(options)
        mirror_obj.start()
        _GLOBAL_ACTIVE_MIRROR = mirror_obj

    token = _ACTIVE_MIRROR.set(mirror_obj)
    had_error = False
    try:
        yield
    except Exception:
        had_error = True
        raise
    finally:
        try:
            mirror_obj.stop(had_error=had_error)
        finally:
            with _GLOBAL_MIRROR_LOCK:
                _GLOBAL_ACTIVE_MIRROR = None
            _ACTIVE_MIRROR.reset(token)


def with_mirror(cache_root: str = '~/.cache/positronic/s3/', show_progress: bool = True, max_workers: int = 10):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with mirror(cache_root=cache_root, show_progress=show_progress, max_workers=max_workers):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def _require_active_mirror() -> _Mirror:
    mirror_obj = _ACTIVE_MIRROR.get()
    if mirror_obj is not None:
        return mirror_obj

    global _GLOBAL_ACTIVE_MIRROR
    if _GLOBAL_ACTIVE_MIRROR is not None:
        return _GLOBAL_ACTIVE_MIRROR

    raise RuntimeError('No active mirror context')


def download(remote: str, local: str | Path | None = None, delete: bool = True) -> Path:
    mirror_obj = _require_active_mirror()
    return mirror_obj.download(remote, local, delete)


def upload(
    remote: str,
    local: str | Path | None = None,
    interval: int | None = 300,
    delete: bool = True,
    sync_on_error: bool = False,
) -> Path:
    mirror_obj = _require_active_mirror()
    return mirror_obj.upload(remote, local, interval, delete, sync_on_error)


__all__ = ['mirror', 'download', 'upload', '_parse_s3_url']
