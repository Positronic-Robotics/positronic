from abc import ABC, abstractmethod
from typing import Any

import numpy as np


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
