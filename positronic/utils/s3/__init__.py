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


def _needs_download(s3_obj: dict, local_path: Path) -> bool:
    return not local_path.exists() or local_path.stat().st_size != s3_obj['Size']


def _needs_upload(local_path: Path, s3_obj: dict | None) -> bool:
    return s3_obj is None or local_path.stat().st_size != s3_obj['Size']


def _rglob(path: Path) -> Iterator[Path]:
    if path.is_file():
        yield path
    else:
        yield from (p for p in path.rglob('*') if p.is_file())


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
    ready: threading.Event = field(default_factory=threading.Event)
    error: Exception | None = None


@dataclass
class _UploadRegistration:
    remote: str
    local_path: Path
    interval: int | None
    delete: bool
    last_sync: float = 0.0


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

    # lifecycle -----------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._stop_event is not None

    def start(self) -> None:
        if not self.running:
            self._stop_event = threading.Event()

    def stop(self) -> None:
        if self.running:
            self._stop_event.set()

            if self._sync_thread:
                self._sync_thread.join(timeout=60)
                self._sync_thread = None
            self._final_sync()
            self._stop_event = None

    # registration --------------------------------------------------------------
    def download(self, remote: str, local: str | Path | None) -> Path:
        if not _is_s3_path(remote):
            path = Path(remote).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f'Local download path does not exist: {path}')
            return path

        normalized = _normalize_s3_url(remote)
        local_path = self.options.cache_path_for(remote) if local is None else Path(local).expanduser().resolve()

        with self._lock:
            existing = self._downloads.get(normalized)
            if existing:
                if existing.local_path != local_path:
                    raise ValueError(f"Download for '{normalized}' already registered at '{existing.local_path}'")
                registration = existing
                need_download = False
            else:
                self._check_download_conflicts(normalized)
                registration = _DownloadRegistration(remote=normalized, local_path=local_path)
                self._downloads[normalized] = registration
                need_download = True

        if need_download:
            try:
                self._perform_download(normalized, local_path)
            except Exception as exc:  # pragma: no cover - error path
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

    def upload(self, remote: str, local: str | Path | None, interval: int | None, delete: bool) -> Path:
        if not _is_s3_path(remote):
            path = Path(remote).expanduser().resolve()
            path.mkdir(parents=True, exist_ok=True)
            return path

        normalized = _normalize_s3_url(remote)
        local_path = self.options.cache_path_for(remote) if local is None else Path(local).expanduser().resolve()

        with self._lock:
            existing = self._uploads.get(normalized)
            if existing:
                if existing.local_path != local_path:
                    raise ValueError(f"Upload for '{normalized}' already registered at '{existing.local_path}'")
                if existing.interval != interval or existing.delete != delete:
                    raise ValueError(
                        f"Upload for '{normalized}' already registered with interval={existing.interval} "
                        f'and delete={existing.delete}'
                    )
                return existing.local_path

            self._check_upload_conflicts(normalized, local_path, interval, delete)
            self._uploads[normalized] = _UploadRegistration(
                remote=normalized, local_path=local_path, interval=interval, delete=delete, last_sync=time.monotonic()
            )
            if interval is not None:
                self._ensure_background_thread_unlocked()

        return local_path

    # guards --------------------------------------------------------------------
    def _check_download_conflicts(self, candidate: str) -> None:
        for upload_remote in self._uploads:
            if _s3_paths_conflict(candidate, upload_remote):
                raise ValueError(f"Conflict: download '{candidate}' overlaps with upload '{upload_remote}'")

    def _check_upload_conflicts(self, candidate: str, local_path: Path, interval: int | None, delete: bool) -> None:
        for download_remote in self._downloads:
            if _s3_paths_conflict(candidate, download_remote):
                raise ValueError(f"Conflict: upload '{candidate}' overlaps with download '{download_remote}'")
        for upload_remote, reg in self._uploads.items():
            if _s3_paths_conflict(candidate, upload_remote):
                same_remote = candidate == upload_remote
                same_params = reg.local_path == local_path and reg.interval == interval and reg.delete == delete
                if not same_remote or not same_params:
                    raise ValueError(f"Conflict: upload '{candidate}' overlaps with upload '{upload_remote}'")

    # background sync -----------------------------------------------------------
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

    # sync helpers --------------------------------------------------------------
    def _final_sync(self) -> None:
        with self._lock:
            uploads = list(self._uploads.values())
        self._sync_uploads(uploads)

    def _sync_uploads(self, registrations: Iterable[_UploadRegistration]) -> None:
        tasks: list[tuple[str, Path, bool]] = []
        for registration in registrations:
            if registration.local_path.exists():
                tasks.append((registration.remote, registration.local_path, registration.delete))

        if not tasks:
            return

        uploads: list[tuple[Path, str, str]] = []
        deletes: list[tuple[str, str]] = []
        total_bytes = 0

        for remote, local_path, delete in tasks:
            bucket, prefix = _parse_s3_url(remote)
            local_files = {p.relative_to(local_path).as_posix() for p in _rglob(local_path)}
            s3_objects = {obj['Key'][len(prefix) :].lstrip('/'): obj for obj in self._list_s3_objects(bucket, prefix)}

            for relative in local_files:
                file_local = local_path / relative if relative != '.' else local_path
                if _needs_upload(file_local, s3_objects.get(relative)):
                    s3_key = prefix + ('/' + relative if relative not in ('', '.') else '')
                    uploads.append((file_local, bucket, s3_key))
                    total_bytes += file_local.stat().st_size

            if delete:
                for orphan in set(s3_objects.keys()) - local_files:
                    s3_key = prefix + ('/' + orphan if orphan not in ('', '.') else '')
                    deletes.append((bucket, s3_key))

        if uploads:
            with (
                self._progress_bar(total_bytes, 'Uploading') as pbar,
                ThreadPoolExecutor(max_workers=self.options.max_workers) as executor,
            ):
                futures = [executor.submit(self._upload_file, path, bucket, key, pbar) for path, bucket, key in uploads]
                _process_futures(as_completed(futures), 'Upload')

        if deletes:
            with ThreadPoolExecutor(max_workers=self.options.max_workers) as executor:
                futures = [executor.submit(self._delete_file, bucket, key) for bucket, key in deletes]
                iterator = as_completed(futures)
                if self.options.show_progress:
                    iterator = tqdm(iterator, total=len(deletes), desc='Deleting')
                _process_futures(iterator, 'Delete')

    # downloads -----------------------------------------------------------------
    def _perform_download(self, remote: str, local_path: Path) -> None:
        bucket, key = _parse_s3_url(remote)

        downloads = []
        total_bytes = 0

        for obj in self._list_s3_objects(bucket, key):
            relative = obj['Key'][len(key) :].lstrip('/')
            target = local_path / relative if relative else local_path
            if _needs_download(obj, target):
                downloads.append((bucket, obj['Key'], target))
                total_bytes += obj['Size']

        if not downloads:
            return

        with (
            self._progress_bar(total_bytes, 'Downloading') as pbar,
            ThreadPoolExecutor(max_workers=self.options.max_workers) as executor,
        ):
            futures = [executor.submit(self._download_file, *args, pbar) for args in downloads]
            _process_futures(as_completed(futures), 'Download')

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

    def _progress_bar(self, total_bytes: int, desc: str):
        if not self.options.show_progress:
            return _NullTqdm()
        return tqdm(total=total_bytes, unit='B', unit_scale=True, unit_divisor=1024, desc=desc)

    def _download_file(self, bucket: str, key: str, local_path: Path, pbar) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.s3_client.download_file(bucket, key, str(local_path), Callback=pbar.update)
        except Exception as exc:
            logger.error('Failed to download %s: %s', key, exc)
            raise

    # uploads -------------------------------------------------------------------
    def _upload_file(self, local_path: Path, bucket: str, key: str, pbar) -> None:
        try:
            self.s3_client.upload_file(str(local_path), bucket, key, Callback=pbar.update)
        except Exception as exc:  # pragma: no cover - logging path
            logger.error('Failed to upload %s to %s/%s: %s', local_path, bucket, key, exc)
            raise

    def _delete_file(self, bucket: str, key: str) -> None:
        try:
            self.s3_client.delete_object(Bucket=bucket, Key=key)
        except Exception as exc:  # pragma: no cover - logging path
            logger.error('Failed to delete %s/%s: %s', bucket, key, exc)
            raise


# public API --------------------------------------------------------------------
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
    try:
        yield
    finally:
        try:
            mirror_obj.stop()
        finally:
            with _GLOBAL_MIRROR_LOCK:
                _GLOBAL_ACTIVE_MIRROR = None
            _ACTIVE_MIRROR.reset(token)


def with_mirror(cache_root: str = '~/.cache/positronic/s3/', show_progress: bool = True, max_workers: int = 10):
    """Decorator to run a function inside a mirror context."""

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


def download(remote: str, local: str | Path | None = None) -> Path:
    mirror_obj = _require_active_mirror()
    return mirror_obj.download(remote, local)


def upload(remote: str, local: str | Path | None = None, interval: int | None = 300, delete: bool = True) -> Path:
    mirror_obj = _require_active_mirror()
    return mirror_obj.upload(remote, local, interval, delete)


__all__ = ['mirror', 'download', 'upload', '_parse_s3_url']
