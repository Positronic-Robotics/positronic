import multiprocessing as mp
import multiprocessing.managers


from ironic2.core import Message, SignalEmitter, SignalReader, system_clock


class SMCompliant:
    def buf_size(self) -> int:
        """Return the buffer size needed for this instance."""
        return 0

    def bind_to_buffer(self, buffer: bytes, lock: mp.Lock) -> None:
        """Bind the instance to a memory buffer. This method is called at most once per instance.
        After the call, all the data must be stored within the buffer and all 'updates' to
        the data must be done through the buffer.

        To some extent can be thought as a serialization method.

        Args:
            buffer: The memory buffer to bind to.
            lock: The lock to use to protect access to the shared memory.
        """
        raise NotImplementedError()

    @classmethod
    def create_from_memoryview(cls, buffer: memoryview, lock: mp.Lock) -> None:
        """Given a memoryview and a lock, create an instance of the class from the memoryview. Can be thought
        as a deserialization method.

        Args:
            buffer: The memory buffer to create the instance from.
            lock: The lock to use to protect access to the shared memory.
        """
        raise NotImplementedError()

    def close(self):
        """Release the resources used by the instance (usually shared memory)"""
        pass

    @property
    def lock(self) -> mp.Lock:
        """Protects access to the shared memory. Use it when reading or writing data from the instance."""
        raise NotImplementedError()


class ZeroCopySMEmitter(SignalEmitter):

    def __init__(self, data_type: type[SMCompliant], manager: mp.Manager, sm_manager: mp.managers.SharedMemoryManager):
        self._data_type = data_type
        self._manager = manager
        self._sm = None
        self._lock = None
        self._ts_value = None
        self._out_data = None
        self._expected_buf_size = None
        self._sm_manager = sm_manager

    def emit(self, data: SMCompliant, ts: int = 0) -> bool:
        assert isinstance(data, self._data_type), f"Data type mismatch: {type(data)} != {self._data_type}"
        ts = ts or system_clock()

        buf_size = data.buf_size()

        if self._sm is None:  # First emit - create shared memory with the size from this instance
            self._expected_buf_size = buf_size
            self._sm = self._sm_manager.SharedMemory(size=buf_size)
            self._ts_value = self._manager.Value('Q', 0)
            self._lock = self._manager.Lock()
            data.bind_to_buffer(self._sm.buf, self._lock)
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
            if hasattr(self._out_data, 'close'):
                self._out_data.close()
            self._out_data = None

        # Clear other references to help with cleanup
        self._ts_value = None
        self._lock = None
        # Note: Don't close self._sm here as it's managed by World.__exit__

    @property
    def shared_memory(self) -> mp.shared_memory.SharedMemory:
        """Get the shared memory object (only available after first emit)."""
        return self._sm

    @property
    def ts_value(self) -> mp.Value:
        """Get the timestamp value (only available after first emit)."""
        return self._ts_value

    @property
    def lock(self) -> mp.Lock:
        """Get the lock (only available after first emit)."""
        return self._lock


class ZeroCopySMReader(SignalReader):

    def __init__(self, emitter: ZeroCopySMEmitter, data_type: type[SMCompliant]):
        self._emitter = emitter
        self._data_type = data_type
        self._out_value = None
        self._readonly_buffer = None

    def read(self) -> Message | None:
        if self._emitter.shared_memory is None:
            return None

        if self._out_value is None:
            self._readonly_buffer = self._emitter.shared_memory.buf.toreadonly()
            self._out_value = self._data_type.create_from_memoryview(
                self._readonly_buffer,
                self._emitter.lock
            )
            self._ts_value = self._emitter.ts_value

        with self._emitter.lock:
            if self._emitter.ts_value.value == 0:
                return None
            return Message(data=self._out_value, ts=self._emitter.ts_value.value)

    def close(self):
        """Release references to shared memory to allow proper cleanup."""
        if self._out_value is not None:
            if hasattr(self._out_value, 'close'):
                self._out_value.close()
            self._out_value = None
        if self._readonly_buffer is not None:
            self._readonly_buffer.release()
            self._readonly_buffer = None
