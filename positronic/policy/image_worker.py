"""Run the image work of a policy call in a child process.

The arrays and bytes of a call pass through shared memory. The pipe carries only the function, its small arguments
and the layout of the shared memory. A control loop in the parent process then keeps its interpreter lock while a
call resizes and encodes camera frames.
"""

import multiprocessing as mp
import os
import threading
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Self

import numpy as np
from positronic_model_server.tree import map_leaves

# A core that the control loop and the worker both want goes to the control loop.
_NICENESS = 10
_ALIGNMENT = 64
_GROWTH = 1.5


class ImageWorkerError(RuntimeError):
    """The image worker process ended, or did not answer in time."""


@dataclass(frozen=True)
class _ArraySlot:
    offset: int
    shape: tuple[int, ...]
    dtype: np.dtype

    def write(self, buf: memoryview, value: np.ndarray) -> None:
        np.ndarray(self.shape, self.dtype, buffer=buf, offset=self.offset)[...] = value

    def read(self, buf: memoryview, copy: bool) -> np.ndarray:
        view = np.ndarray(self.shape, self.dtype, buffer=buf, offset=self.offset)
        return view.copy() if copy else view


@dataclass(frozen=True)
class _BytesSlot:
    offset: int
    size: int

    def write(self, buf: memoryview, value: bytes) -> None:
        buf[self.offset : self.offset + self.size] = value

    def read(self, buf: memoryview, copy: bool) -> bytes:
        return bytes(buf[self.offset : self.offset + self.size])


_Slot = _ArraySlot | _BytesSlot


@dataclass(frozen=True)
class _Call:
    function: Callable[..., Any]
    args: tuple[Any, ...]
    block: str | None
    layout: Any


@dataclass(frozen=True)
class _Answer:
    block: str | None
    layout: Any


@dataclass(frozen=True)
class _Failure:
    error: Exception


@dataclass(frozen=True)
class _Ready:
    pass


class _Writer:
    """The shared memory block that one side writes. A larger block replaces it when a tree does not fit."""

    def __init__(self) -> None:
        self._block: SharedMemory | None = None

    def write(self, tree: Any) -> tuple[str | None, Any]:
        """Copy each array and bytes value of ``tree`` into the block. Return the block name and ``tree`` with
        each such value replaced by its slot."""
        placed: list[tuple[_Slot, Any]] = []
        end = 0

        def place(value: Any) -> Any:
            nonlocal end
            if isinstance(value, np.ndarray) and not value.dtype.hasobject:
                slot, size = _ArraySlot(end, value.shape, value.dtype), value.nbytes
            elif isinstance(value, bytes):
                slot, size = _BytesSlot(end, len(value)), len(value)
            else:
                return value
            placed.append((slot, value))
            end += -(-size // _ALIGNMENT) * _ALIGNMENT
            return slot

        layout = map_leaves(tree, place)
        if not placed:
            return None, layout
        if self._block is None or self._block.size < end:
            self.close()
            self._block = SharedMemory(create=True, size=max(int(end * _GROWTH), _ALIGNMENT))
        buf = self._block.buf
        assert buf is not None
        for slot, value in placed:
            slot.write(buf, value)
        return self._block.name, layout

    def close(self) -> None:
        if self._block is not None:
            self._block.close()
            with suppress(FileNotFoundError):
                self._block.unlink()
            self._block = None


class _Reader:
    """The mapping of the shared memory block that the other side writes."""

    def __init__(self) -> None:
        self._block: SharedMemory | None = None

    def read(self, name: str | None, layout: Any, copy: bool) -> Any:
        """``layout`` with each slot replaced by its value in the block ``name``."""
        if name is None:
            return layout
        if self._block is None or self._block.name != name:
            self.close()
            self._block = SharedMemory(name=name)
        buf = self._block.buf
        assert buf is not None
        return map_leaves(layout, lambda value: value.read(buf, copy) if isinstance(value, _Slot) else value)

    def close(self, unlink: bool = False) -> None:
        if self._block is not None:
            self._block.close()
            if unlink:
                with suppress(FileNotFoundError):
                    self._block.unlink()
            self._block = None


def _answer(call: _Call, inputs: _Reader, outputs: _Writer) -> _Answer | _Failure:
    try:
        return _Answer(*outputs.write(call.function(inputs.read(call.block, call.layout, copy=False), *call.args)))
    except Exception as error:
        return _Failure(error)


def _serve(conn: Connection) -> None:
    os.nice(_NICENESS)
    inputs, outputs = _Reader(), _Writer()
    conn.send(_Ready())
    try:
        while True:
            try:
                call = conn.recv()
            except EOFError:
                return
            reply = _answer(call, inputs, outputs)
            try:
                conn.send(reply)
            except Exception as error:
                conn.send(_Failure(ImageWorkerError(f'The image worker cannot send its answer: {error!r}')))
            del reply
    finally:
        inputs.close()
        outputs.close()


class ImageWorker:
    """A child process that runs the image work of policy calls, one call at a time.

    A call waits ``timeout_sec`` for its answer, and the first call also waits ``start_timeout_sec`` for the
    process to start. A process that ends or does not answer in time is stopped and the call raises
    ``ImageWorkerError``. The next call starts a new process.
    """

    def __init__(self, timeout_sec: float = 10.0, start_timeout_sec: float = 60.0) -> None:
        self._timeout_sec = timeout_sec
        self._start_timeout_sec = start_timeout_sec
        self._lock = threading.Lock()
        self._inputs = _Writer()
        self._outputs = _Reader()
        self._process: BaseProcess | None = None
        self._ready = False
        weakref.finalize(self, ImageWorker._release, self._inputs, self._outputs)
        self._start()

    def _start(self) -> None:
        context = mp.get_context('spawn')
        self._conn, child = context.Pipe()
        self._process = context.Process(target=_serve, args=(child,), name='image-worker', daemon=True)
        self._process.start()
        child.close()
        self._ready = False

    def _receive(self, timeout_sec: float, waited_for: str) -> _Ready | _Answer | _Failure:
        """The next message of the process. Stop the process when it ends or sends none in ``timeout_sec``."""
        try:
            if self._conn.poll(timeout_sec):
                return self._conn.recv()
        except (EOFError, OSError) as error:
            self._stop()
            raise ImageWorkerError(f'The image worker process ended before it answered {waited_for}') from error
        self._stop()
        raise ImageWorkerError(f'The image worker did not answer {waited_for} in {timeout_sec:.1f} s')

    def _stop(self) -> None:
        assert self._process is not None
        self._process.kill()
        self._process.join()
        self._conn.close()
        self._process = None
        # A killed process cannot remove the block that it created.
        self._outputs.close(unlink=True)

    def call(self, function: Callable[..., Any], tree: Any, /, *args: Any) -> Any:
        """``function(tree, *args)``, run in the worker process. ``function`` and ``args`` must pickle."""
        with self._lock:
            if self._process is None:
                self._start()
            if not self._ready:
                self._receive(self._start_timeout_sec, 'its start')
                self._ready = True
            block, layout = self._inputs.write(tree)
            try:
                self._conn.send(_Call(function, args, block, layout))
            except OSError as error:
                self._stop()
                raise ImageWorkerError('The image worker process ended before it took a call') from error
            reply = self._receive(self._timeout_sec, 'a call')
            if isinstance(reply, _Failure):
                raise reply.error
            assert isinstance(reply, _Answer)
            return self._outputs.read(reply.block, reply.layout, copy=True)

    def close(self) -> None:
        """Stop the process and remove the shared memory. A later call starts a new process."""
        with self._lock:
            if self._process is not None:
                self._stop()
            ImageWorker._release(self._inputs, self._outputs)

    @staticmethod
    def _release(inputs: _Writer, outputs: _Reader) -> None:
        inputs.close()
        outputs.close(unlink=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


_in_use: ContextVar[ImageWorker | None] = ContextVar('positronic_image_worker', default=None)


@contextmanager
def use_image_worker(worker: ImageWorker | None) -> Iterator[None]:
    """Run the image codecs in ``worker`` inside the block, and in each policy call that the block submits."""
    token = _in_use.set(worker)
    try:
        yield
    finally:
        _in_use.reset(token)


def image_worker_in_use() -> ImageWorker | None:
    """The worker that ``use_image_worker`` set for this context, or None to run the image codecs here."""
    return _in_use.get()
