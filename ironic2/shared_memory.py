import multiprocessing as mp
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
    def buf_size(self) -> int:
        """Return the buffer size needed for `data`."""
        return 0

    @abstractmethod
    def move_to_buffer(self, buffer: memoryview | bytes | bytearray) -> None:
        """Bind the instance to a memory buffer (kinda zero-copy serialization).
        This method is called at most once per `data` instance.
        After the call, all the data must be stored within the buffer and all 'updates' to
        the data must be done through the buffer.

        Args:
            data: The data to bind to the buffer.
            buffer: The memory buffer to bind to.
        """
        pass

    @classmethod
    @abstractmethod
    def create_from_memoryview(cls, buffer: memoryview | bytes) -> Self:
        """Given a memoryview, create an instance of the class from the memoryview (kinda zero-copy deserialization).

        Args:
            buffer: The memory buffer to create the instance from. Can be a memoryview, bytes, or bytearray.

        Returns:
            The 'deserialized' data mapped to the buffer.
        """
        pass

    def close(self):
        """Release the resources used by the instance (usually shared memory)"""
        pass


class NumpySMAdapter(SMCompliant):
    """SMAdapter implementation for numpy arrays with support for all numeric dtypes."""

    # Mapping of numpy dtypes to codes
    DTYPE_TO_CODE = {
        # Unsigned integers
        np.uint8: 0,
        np.uint16: 1,
        np.uint32: 2,
        np.uint64: 3,
        # Signed integers
        np.int8: 4,
        np.int16: 5,
        np.int32: 6,
        np.int64: 7,
        # Floating point
        np.float16: 8,
        np.float32: 9,
        np.float64: 10,
        # Complex floating point
        np.complex64: 11,
        np.complex128: 12,
    }

    # Reverse mapping
    CODE_TO_DTYPE = {v: k for k, v in DTYPE_TO_CODE.items()}

    _array: np.ndarray

    def __init__(self, array: np.ndarray):
        if not array.flags.c_contiguous:
            raise ValueError("Array must be C-contiguous. Use np.ascontiguousarray() to make it contiguous.")
        self._array = array

    @property
    def array(self) -> np.ndarray:
        return self._array

    def buf_size(self) -> int:
        """Calculate buffer size needed for numpy array."""
        # Buffer layout:
        # 1 byte: dtype code
        # 1 byte: number of dimensions
        # 4 bytes per dimension: dimension sizes (uint32)
        # remaining bytes: array data

        header_size = 1 + 1 + (4 * self._array.ndim)  # dtype + ndim + shape
        data_size = self._array.nbytes
        return header_size + data_size

    def move_to_buffer(self, buffer: memoryview | bytes | bytearray) -> None:
        """Bind numpy array to shared memory buffer."""
        # Ensure array is still contiguous
        if not self._array.flags.c_contiguous:
            raise ValueError("Array must be C-contiguous. Use np.ascontiguousarray() to make it contiguous.")

        # Ensure buffer is writable
        if isinstance(buffer, bytes):
            raise ValueError("Buffer must be writable")

        if not isinstance(buffer, memoryview):  # Convert to memoryview for easier manipulation
            buffer = memoryview(buffer)

        if buffer.readonly:
            raise ValueError("Buffer must be writable")

        offset = 0

        dtype_code = self.DTYPE_TO_CODE[self._array.dtype.type]
        buffer[offset] = dtype_code
        offset += 1

        buffer[offset] = self._array.ndim
        offset += 1

        for dim_size in self._array.shape:
            struct.pack_into('<I', buffer, offset, dim_size)  # little-endian uint32
            offset += 4

        data_buffer = buffer[offset:offset + self._array.nbytes]
        data_buffer[:] = self._array.tobytes()

        # Now the array is bound to the buffer - use exact size
        data_slice = buffer[offset:offset + self._array.nbytes]
        self._array = np.frombuffer(data_slice, dtype=self._array.dtype).reshape(self._array.shape)

    @classmethod
    def create_from_memoryview(cls, buffer: memoryview | bytes) -> Self:
        """Create numpy array from shared memory buffer."""
        if isinstance(buffer, bytes):
            buffer = memoryview(buffer)

        offset = 0

        # Read dtype code (1 byte)
        dtype_code = buffer[offset]
        if dtype_code not in cls.CODE_TO_DTYPE:
            raise ValueError(f"Invalid dtype code: {dtype_code}")
        dtype = cls.CODE_TO_DTYPE[dtype_code]
        offset += 1

        # Read number of dimensions (1 byte)
        ndim = buffer[offset]
        offset += 1

        # Read shape (4 bytes per dimension)
        shape = []
        for _ in range(ndim):
            dim_size = struct.unpack_from('<I', buffer, offset)[0]  # little-endian uint32
            shape.append(dim_size)
            offset += 4
        shape = tuple(shape)

        # Create array from remaining buffer data
        array_size = np.prod(shape, dtype=int) * np.dtype(dtype).itemsize
        data_buffer = buffer[offset:offset + array_size]
        array = np.frombuffer(data_buffer, dtype=dtype).reshape(shape)

        return cls(array)

    def close(self):
        self._array = None


class ZeroCopySMEmitter(SignalEmitter):

    def __init__(self, manager: mp.Manager, sm_manager: mp.managers.SharedMemoryManager, clock: Clock):
        self._data_type: type[SMCompliant] | None = None
        self._sm = None
        self._lock = manager.Lock()
        self._ts_value = manager.Value('Q', -1)
        self._out_data = None
        self._expected_buf_size = None
        self._sm_manager = sm_manager
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
            self._sm = self._sm_manager.SharedMemory(size=buf_size)
            data.move_to_buffer(self._sm.buf)
            self._out_data = data
        else:  # Subsequent emits - validate buffer size matches
            assert buf_size == self._expected_buf_size, \
                f"Buffer size mismatch: expected {self._expected_buf_size}, got {buf_size}. " \
                "All data instances must have the same buffer size for a given channel."
            assert data is self._out_data, "SMEmitter can only emit the same object multiple times"

        with self._lock:
            self._ts_value.value = ts
        return True

    def close(self):
        """Release references to shared memory to allow proper cleanup."""
        if self._out_data is not None:
            self._out_data.close()
            self._out_data = None

    def zc_lock(self) -> ContextManager[None]:
        return self._lock


class ZeroCopySMReader(SignalReader):

    def __init__(self, emitter: ZeroCopySMEmitter):
        self._emitter = emitter
        self._out_value = None
        self._readonly_buffer = None

    def read(self) -> Message | None:
        if self._emitter._sm is None:
            return None

        if self._out_value is None:
            self._readonly_buffer = self._emitter._sm.buf.toreadonly()
            self._out_value = self._emitter._data_type.create_from_memoryview(self._readonly_buffer)

        with self._emitter._lock:
            if self._emitter._ts_value.value == -1:
                return None
            return Message(data=self._out_value, ts=self._emitter._ts_value.value)

    def close(self):
        """Release references to shared memory to allow proper cleanup."""
        if self._out_value is not None:
            self._out_value.close()
            self._out_value = None

        if self._readonly_buffer is not None:
            self._readonly_buffer.release()
            self._readonly_buffer = None

    def zc_lock(self) -> ContextManager[None]:
        return self._emitter.zc_lock()
