import multiprocessing as mp
import multiprocessing.shared_memory
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ironic2.core import Clock, Message, SignalEmitter, SignalReader


class SMCompliant(ABC):
    """Interface for data that could be used as view of some contiguous buffer."""

    def buf_size(self) -> int:
        """Return the buffer size needed for this instance."""
        return 0

    def instantiation_params(self) -> tuple[Any, ...]:
        """Return the parameters needed to instantiate this class from the buffer."""
        return ()

    @abstractmethod
    def set_to_buffer(self, buffer: memoryview | bytes | bytearray) -> None:
        """Serialize data to the given buffer.

        Args:
            buffer: The memory buffer to serialize to.
        """
        pass

    @abstractmethod
    def read_from_buffer(self, buffer: memoryview | bytes) -> None:
        """Deserialize data from the given buffer.

        Args:
            buffer: The memory buffer to deserialize from.
        """
        pass


class NumpySMAdapter(SMCompliant):
    """SMAdapter implementation for numpy arrays."""

    def __init__(self, shape: tuple[int, ...], dtype: np.dtype):
        self.array = np.empty(shape, dtype=dtype)

    def instantiation_params(self) -> tuple[Any, ...]:
        return (self.array.shape, self.array.dtype)

    def buf_size(self) -> int:
        return self.array.nbytes

    def set_to_buffer(self, buffer: memoryview | bytes | bytearray) -> None:
        buffer[:self.array.nbytes] = self.array.view(np.uint8).reshape(-1).data

    def read_from_buffer(self, buffer: memoryview | bytes) -> None:
        self.array[:] = np.frombuffer(buffer[:self.array.nbytes], dtype=self.array.dtype).reshape(self.array.shape)


class SharedMemoryEmitter(SignalEmitter):
    def __init__(self, lock: mp.Lock, ts_value: mp.Value, sm_queue: mp.Queue, clock: Clock):
        self._data_type: type[SMCompliant] | None = None
        self._lock = lock
        self._ts_value = ts_value
        self._sm_queue = sm_queue

        self._sm = None
        self._expected_buf_size = None
        self._clock = clock

    def emit(self, data: Any, ts: int = -1) -> bool:
        ts = ts if ts >= 0 else self._clock.now_ns()
        if self._data_type is None:
            self._data_type = type(data)
            assert issubclass(self._data_type, SMCompliant), f"Data type {self._data_type} is not SMCompliant"
        else:
            assert isinstance(data, self._data_type), f"Data type mismatch: {type(data)} != {self._data_type}"

        buf_size = data.buf_size()

        if self._sm is None:  # First emit - create shared memory with the size from this instance
            self._expected_buf_size = buf_size
            # Note that size of shared_memory could be larger than the requested size depending on the platform
            self._sm = mp.shared_memory.SharedMemory(create=True, size=buf_size)
            self._sm_queue.put((self._sm, self._data_type, data.instantiation_params()))
        else:  # Subsequent emits - validate buffer size matches
            assert buf_size == self._expected_buf_size, \
                f"Buffer size mismatch: expected {self._expected_buf_size}, got {buf_size}. " \
                "All data instances must have the same buffer size for a given channel."

        with self._lock:
            data.set_to_buffer(self._sm.buf)
            self._ts_value.value = ts

        return True

    def close(self):
        """Release references to shared memory to allow proper cleanup."""
        if self._sm is not None:
            self._sm.close()
            self._sm.unlink()
            self._sm = None


class SharedMemoryReader(SignalReader):
    def __init__(self, lock: mp.Lock, ts_value: mp.Value, sm_queue: mp.Queue):
        self._lock = lock
        self._ts_value = ts_value
        self._sm_queue = sm_queue

        self._out_value = None
        self._return_value = None
        self._readonly_buffer = None
        self._sm = None

    def read(self) -> Message | None:
        with self._lock:
            if self._ts_value.value == -1:
                return None

        if self._out_value is None:
            self._sm, data_type, instantiation_params = self._sm_queue.get_nowait()
            self._readonly_buffer = self._sm.buf.toreadonly()
            self._out_value = data_type(*instantiation_params)

        with self._lock:
            if self._ts_value.value == -1:
                return None

            self._out_value.read_from_buffer(self._readonly_buffer)

            return Message(data=self._out_value, ts=self._ts_value.value)

    def close(self):
        """Release references to shared memory to allow proper cleanup."""
        if self._readonly_buffer is not None:
            self._readonly_buffer.release()
            self._readonly_buffer = None

        if self._sm is not None:
            # Don't call unlink here, it will be called by the emitter
            self._sm.close()
            self._sm = None
