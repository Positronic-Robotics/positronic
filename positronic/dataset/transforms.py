from typing import Callable, Sequence, TypeVar, Tuple
from collections.abc import Sequence as SequenceABC

import numpy as np

from positronic.dataset.core import Signal

T = TypeVar('T')
U = TypeVar('U')


def one_to_many(f: Callable[[T], Sequence[U]]) -> Callable[[T | Sequence[T]], Sequence[U]]:

    def wrapper(x: T | Sequence[T]) -> Sequence[U]:
        if isinstance(x, Sequence) or isinstance(x, np.ndarray):
            return [f(xi) for xi in x]
        return f(x)

    return wrapper


class Elementwise(Signal[U]):

    def __init__(self, signal: Signal[T], f: Callable[[T | Sequence[T]], U | Sequence[U]]) -> None:
        self._signal = signal
        self._f = f

    def __len__(self) -> int:
        return len(self._signal)

    def _ts_at(self, index_or_indices: int | Sequence[int] | np.ndarray) -> int | Sequence[int] | np.ndarray:
        return self._signal._ts_at(index_or_indices)

    def _values_at(self, index_or_indices: int | Sequence[int] | np.ndarray) -> U | Sequence[U]:
        return self._f(self._signal._values_at(index_or_indices))

    def _search_ts(self, ts_or_array: int | Sequence[int] | np.ndarray) -> int | np.ndarray:
        return self._signal._search_ts(ts_or_array)


class Previous(Signal[Tuple[T, T, int]]):

    def __init__(self, signal: Signal[T]) -> None:
        self._signal = signal

    def __len__(self) -> int:
        return len(self._signal) - 1

    def _ts_at(self, index_or_indices: int | Sequence[int] | np.ndarray) -> int | Sequence[int] | np.ndarray:
        return self._signal._ts_at(index_or_indices)

    def _values_at(self,
                   index_or_indices: int | Sequence[int] | np.ndarray) -> Tuple[T, T, int] | Sequence[Tuple[T, T, int]]:
        match index_or_indices:
            case int() | np.np.integer as index:
                return (self._signal._values_at(index + 1), self._signal._values_at(index),
                        self._signal._ts_at(index + 1) - self._signal._ts_at(index))
            case slice() | np.ndarray() | SequenceABC() as idxs:
                idx = np.asarray(idxs)
                prev_values = self._signal._values_at(idx)
                cur_values = self._signal._values_at(idx + 1)
                prev_ts = self._signal._ts_at(idx + 1)
                cur_ts = self._signal._ts_at(idx)
                return np.stack([(prev_values[i], cur_values[i], prev_ts[i] - cur_ts[i])
                                 for i in range(len(prev_values))])
            case _:
                raise TypeError(f"Unsupported index type: {type(index_or_indices)}")

    def _search_ts(self, ts_or_array: int | Sequence[int] | np.ndarray) -> int | np.ndarray:
        return self._signal._search_ts(ts_or_array)
