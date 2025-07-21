from contextlib import contextmanager
import multiprocessing as mp
import multiprocessing.shared_memory
import struct
from abc import ABC, abstractmethod
from typing import Any, ContextManager, Self

import numpy as np

from ironic2.core import Clock, Message, SignalEmitter, SignalReader

# Requirements:
# - Writer does not know which communication channel is used
# - Shared memory is zero-copy, i.e. both reader and writer point to the same memory
# - Numpy Array must be passed as easily as possible


class SMCompliant(ABC):
    """Interface for data that could be used as view of some contigubuffer."""

    def buf_size(self) -> int:
        """Return the buffer size needed for `data`."""
        return 0

    def instantiation_params(self) -> tuple[Any, ...]:
        """Return the parameters needed to instantiate the class from the buffer."""
        return ()

    @abstractmethod
    def set_to_buffer(self, buffer: memoryview | bytes | bytearray) -> None:
        """Bind the instance to a memory buffer (kinda zero-copy serialization).
        This method is called at most once per `data` instance.
        After the call, all the data must be stored within the buffer and all 'updates' to
        the data must be done through the buffer.

        Args:
            data: The data to bind to the buffer.
            buffer: The memory buffer to bind to.
        """
        pass

    @abstractmethod
    def read_from_buffer(cls, buffer: memoryview | bytes) -> None:
        """Given a memoryview, create an instance of the class from the memoryview (kinda zero-copy deserialization).

        Args:
            buffer: The memory buffer to create the instance from. Can be a memoryview, bytes, or bytearray.

        Returns:
            The 'deserialized' data mapped to the buffer.
        """
        pass



class NumpySMAdapter(SMCompliant):
    """SMAdapter implementation for numpy arrays with support for all numeric dtypes."""
    _array: np.ndarray | None = None

    def __init__(self, shape: tuple[int, ...], dtype: np.dtype):
        self._array = np.empty(shape, dtype=dtype)

    def instantiation_params(self) -> tuple[Any, ...]:
        return (self._array.shape, self._array.dtype)

    @property
    def array(self) -> np.ndarray:
        return self._array

    @array.setter
    def array(self, array: np.ndarray) -> None:
        self._array = array

    def buf_size(self) -> int:
        return self._array.nbytes

    def set_to_buffer(self, buffer: memoryview | bytes | bytearray) -> None:
        buffer[:] = self._array.view(np.uint8).reshape(-1).data

    def read_from_buffer(self, buffer: memoryview | bytes) -> None:
        self._array[:] = np.frombuffer(buffer, dtype=self._array.dtype).reshape(self._array.shape)



class ZeroCopySMEmitter(SignalEmitter):
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


class ZeroCopySMReader(SignalReader):
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
