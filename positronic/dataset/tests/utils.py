import numpy as np

from positronic.dataset.episode import Episode, EpisodeContainer
from positronic.dataset.signal import IndicesLike, RealNumericArrayLike, Signal, is_realnum_dtype
from positronic.dataset.transforms import Elementwise, EpisodeTransform


class DummySignal(Signal[int]):
    """Minimal array-backed Signal implementing the abstract API only.

    Used to validate core Signal's generic indexing/time logic and views.
    """

    def __init__(self, timestamps, values):
        ts_arr = np.asarray(timestamps, dtype=np.int64)
        vals_arr = np.asarray(values)
        assert ts_arr.ndim == 1
        assert vals_arr.shape[0] == ts_arr.shape[0]
        self._ts = ts_arr
        self._vals = vals_arr

    def __len__(self) -> int:
        return int(self._ts.shape[0])

    def _ts_at(self, index_or_indices: IndicesLike) -> np.ndarray:
        idxs = np.asarray(index_or_indices, dtype=np.int64)
        if idxs.size == 0:
            return np.array([], dtype=np.int64)
        return self._ts[idxs]

    def _values_at(self, index_or_indices: IndicesLike):
        idxs = np.asarray(index_or_indices, dtype=np.int64)
        if idxs.size == 0:
            return []
        return self._vals[idxs]

    def _search_ts(self, ts_or_array: RealNumericArrayLike) -> np.ndarray:
        req = np.asarray(ts_or_array)
        if req.size == 0:
            return np.array([], dtype=np.int64)
        if not is_realnum_dtype(req.dtype):
            raise TypeError(f'Invalid timestamp array dtype: {req.dtype}')
        return np.searchsorted(self._ts, req, side='right') - 1


class DummyTransform(EpisodeTransform):
    """Configurable transform for testing.

    Applies elementwise operations to signals from the input episode.

    Example:
        # Transform that multiplies 's' by 10 and outputs as 'a', and adds 1 to 's'
        DummyTransform(
            operations={'a': ('s', lambda x: x * 10), 's': ('s', lambda x: x + 1)},
            pass_through=True
        )
    """

    def __init__(self, operations: dict[str, tuple[str, callable]], pass_through: bool | list[str] = False):
        """
        Args:
            operations: Dict mapping output_key -> (input_key, transform_func).
                       The transform_func receives arrays and returns transformed arrays.
            pass_through: Whether to pass through keys from input episode (True/False/list of keys)
        """
        self._operations = operations
        self._pass_through = pass_through

    def __call__(self, episode: Episode) -> Episode:
        data = {}

        # Apply all operations
        for out_key, (in_key, func) in self._operations.items():
            input_signal = episode[in_key]
            data[out_key] = Elementwise(input_signal, lambda seq, f=func: np.asarray(f(seq)))

        # Handle pass_through logic
        if self._pass_through is True:
            for key in episode:
                if key not in data:
                    data[key] = episode[key]
        elif isinstance(self._pass_through, list):
            for key in self._pass_through:
                if key not in data and key in episode:
                    data[key] = episode[key]

        return EpisodeContainer(data=data, meta=episode.meta)
