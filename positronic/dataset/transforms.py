from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence, TypeVar, Tuple

import numpy as np
import cv2
from PIL import Image as PilImage

from .signal import Signal, IndicesLike, RealNumericArrayLike
from .episode import Episode

T = TypeVar("T")
U = TypeVar("U")


class Elementwise(Signal[U]):
    """Element-wise value transform view over a Signal.

    Wraps another `Signal[T]` and applies a function `f` to its values while
    preserving timestamps and ordering. Length and time indexing semantics are
    identical to the underlying signal.
    """

    def __init__(self, signal: Signal[T], fn: Callable[[Sequence[T]], Sequence[U]]):
        self._signal = signal
        self._fn = fn

    def __len__(self) -> int:
        return len(self._signal)

    def _ts_at(self, indices: IndicesLike) -> Sequence[int] | np.ndarray:
        return self._signal._ts_at(indices)

    def _values_at(self, indices: IndicesLike) -> Sequence[U]:
        return self._fn(self._signal._values_at(indices))

    def _search_ts(self, ts_array: RealNumericArrayLike) -> IndicesLike:
        return self._signal._search_ts(ts_array)


class IndexOffsets(Signal[Tuple[Tuple[T, ...], Tuple[int, ...]]]):
    """Join values and timestamps at relative indices around a reference index.

    Given a list of relative indices D = [d1, d2, ..., dN] (each may be negative
    or positive), produces a view over the reference indices i where all
    (i + dk) are in-bounds. For each valid i, returns
        ((v[i+d1], ..., v[i+dN], t[i+d1], ..., t[i+dN]), t[i]).

    Examples:
      - Next with step=1  -> D = [0, 1]
      - Previous step=1   -> D = [0, -1]
    """

    def __init__(self, signal: Signal[T], relative_indices: Sequence[int]) -> None:
        self._signal = signal
        offs = np.asarray(relative_indices, dtype=np.int64)
        if offs.size == 0:
            raise ValueError("relative_indices must be non-empty")
        self._offs = offs
        self._min_off = int(np.min(self._offs))
        self._max_off = int(np.max(self._offs))

    def __len__(self) -> int:
        n = len(self._signal)
        start_trim = max(0, -self._min_off)
        end_trim = max(0, self._max_off)
        return max(0, n - start_trim - end_trim)

    def _base_start(self) -> int:
        return max(0, -self._min_off)

    def _base_last(self) -> int:
        n = len(self._signal)
        return n - 1 - max(0, self._max_off)

    def _ts_at(self, indices: IndicesLike) -> Sequence[int] | np.ndarray:
        base = np.asarray(indices, dtype=np.int64) + self._base_start()
        return self._signal._ts_at(base)

    def _values_at(self, indices: IndicesLike):
        base = np.asarray(indices, dtype=np.int64) + self._base_start()
        vals_parts = []
        ts_parts = []
        for off in self._offs:
            idxs = base + int(off)
            vals_parts.append(self._signal._values_at(idxs))
            ts_parts.append(np.asarray(self._signal._ts_at(idxs)))

        return list(zip(*vals_parts, *ts_parts, strict=False))

    def _search_ts(self, ts_array: RealNumericArrayLike) -> IndicesLike:
        # Map parent floor indices to view indices, clamping to valid range.
        n = len(self)
        t = np.asarray(ts_array)
        if n == 0:
            return np.full_like(t, -1, dtype=np.int64)  # nothing valid in this view
        p = np.asarray(self._signal._search_ts(t))
        base_start = self._base_start()
        base_last = self._base_last()
        view_idx = p - base_start
        view_idx[p < base_start] = -1
        view_idx[p > base_last] = n - 1
        return view_idx


class TimeOffsets(Signal[Tuple[Tuple[T, ...], Tuple[int, ...]]]):
    """Sample values at time offsets relative to each reference timestamp.

    Given deltas D = [d1, d2, ..., dN], for each valid reference index i returns
        ((v_at(ts[i]+d1), ..., v_at(ts[i]+dN), t_at(ts[i]+d1), ..., t_at(ts[i]+dN)), ts[i])
    where v_at(t) carries back the last value at-or-before t.

    Semantics:
    - For negative deltas: elements whose shifted time precedes the first
      timestamp are dropped (affects the start of the series).
    - For non-negative deltas: when the shifted time exceeds the last
      timestamp, sampling clamps to the last element.
    """

    def __init__(self, signal: Signal[T], deltas_ts: Sequence[int]) -> None:
        self._signal = signal
        offs = np.asarray(deltas_ts, dtype=np.int64)
        if offs.size == 0:
            raise ValueError("deltas_ts must be non-empty")
        self._deltas = offs
        self._bounds_ready = False
        self._start_offset = 0
        self._last_index = -1

    def _compute_bounds(self) -> None:
        if self._bounds_ready:
            return
        n = len(self._signal)
        if n == 0:
            self._start_offset = 0
            self._last_index = -1
            self._bounds_ready = True
            return
        start_offset = 0
        last_index = n - 1
        neg = self._deltas[self._deltas < 0]
        if neg.size > 0:
            thr = int(self._signal.start_ts + int(np.max(-neg)))
            floor_idx = int(np.asarray(self._signal._search_ts([thr]))[0])
            if floor_idx < 0:
                start_offset = 0
            else:
                floor_ts = _ts_at_index(self._signal, floor_idx)
                start_offset = floor_idx if floor_ts == thr else floor_idx + 1
        self._start_offset = start_offset
        self._last_index = last_index
        self._bounds_ready = True

    def __len__(self) -> int:
        self._compute_bounds()
        if self._last_index < self._start_offset:
            return 0
        return self._last_index - self._start_offset + 1

    def _ts_at(self, indices: IndicesLike) -> Sequence[int] | np.ndarray:
        self._compute_bounds()
        idxs = np.asarray(indices, dtype=np.int64)
        if self._start_offset == 0:
            return self._signal._ts_at(idxs)
        else:
            return self._signal._ts_at(idxs + self._start_offset)

    def _values_at(self, indices: IndicesLike):
        self._compute_bounds()
        base = np.asarray(indices, dtype=np.int64)
        if self._start_offset > 0:
            base = base + self._start_offset
        ref_ts = np.asarray(self._signal._ts_at(base))
        vals_parts = []
        ts_parts = []
        for d in self._deltas:
            target_ts = ref_ts + int(d)
            idx = np.asarray(self._signal._search_ts(target_ts))
            vals_parts.append(self._signal._values_at(idx))
            ts_parts.append(np.asarray(self._signal._ts_at(idx)))
        out = []
        for row in zip(*vals_parts, *ts_parts, strict=False):
            out.append(tuple(row))
        return out

    def _search_ts(self, ts_array: RealNumericArrayLike) -> IndicesLike:
        self._compute_bounds()
        parent_idx = np.asarray(self._signal._search_ts(ts_array))
        if (self._deltas < 0).any():
            shifted = parent_idx - self._start_offset
            shifted[parent_idx < self._start_offset] = -1
            return shifted
        return parent_idx


def _ts_at_index(sig: Signal[T], idx: int) -> int:
    """Fetch a single timestamp at the given index as int."""
    return int(np.asarray(sig._ts_at([idx]))[0])


def _first_idx_at_or_after(sig: Signal[T], ts: int) -> int:
    floor = int(np.asarray(sig._search_ts([ts]))[0])
    if floor < 0:
        return 0
    floor_ts = _ts_at_index(sig, floor)
    return floor if floor_ts == ts else floor + 1


class Join(Signal[Tuple[T, U, int]]):
    """Join two signals on the union of their timestamps with carry-back.

    - Reference times: sorted union of parents' timestamps, starting from
      max(s1.start_ts, s2.start_ts). Equal timestamps are collapsed into a
      single entry.
    - Values: at each union timestamp t, returns
      ((v1_at_or_before_t, v2_at_or_before_t, ts2_ref - ts1_ref), t), where
      ts*_ref are the timestamps of the carried-back values in each parent.
    - Union timestamps are precomputed for O(log N) time lookups.
    """

    def __init__(self, s1: Signal[T], s2: Signal[U]) -> None:
        self._s1 = s1
        self._s2 = s2
        self._bounds_ready = False
        self._s1_start = 0
        self._s2_start = 0
        self._length = 0
        self._union_ts: np.ndarray | None = None

    def _compute_bounds(self) -> None:
        if self._bounds_ready:
            return
        n1, n2 = len(self._s1), len(self._s2)
        if n1 == 0 or n2 == 0:
            self._s1_start = n1
            self._s2_start = n2
            self._length = 0
            self._bounds_ready = True
            return
        start_ts = max(self._s1.start_ts, self._s2.start_ts)
        self._s1_start = _first_idx_at_or_after(self._s1, start_ts)
        self._s2_start = _first_idx_at_or_after(self._s2, start_ts)
        # Build union timestamps with duplicates collapsed
        self._union_ts = np.asarray(list(self._iter_merged_ts(dedup=True)), dtype=np.int64)
        self._length = int(self._union_ts.shape[0])
        self._bounds_ready = True

    def _iter_merged_ts(self, dedup: bool):
        i1, i2 = self._s1_start, self._s2_start
        n1, n2 = len(self._s1), len(self._s2)
        inf_ts = np.iinfo(np.int64).max
        while i1 < n1 or i2 < n2:
            ts1 = _ts_at_index(self._s1, i1) if i1 < n1 else inf_ts
            ts2 = _ts_at_index(self._s2, i2) if i2 < n2 else inf_ts
            if dedup and ts1 == ts2:
                yield ts1
                i1 += 1
                i2 += 1
            elif ts1 <= ts2:
                yield ts1
                i1 += 1
            else:
                yield ts2
                i2 += 1

    def __len__(self) -> int:
        self._compute_bounds()
        return self._length

    def _ts_at(self, indices: IndicesLike) -> Sequence[int] | np.ndarray:
        self._compute_bounds()
        idxs = np.asarray(indices)
        return self._union_ts[idxs]

    def _values_at(self, indices: IndicesLike) -> Sequence[Tuple[T, U, int]]:
        ts = np.asarray(self._ts_at(indices))
        # Sample both parents at these timestamps and compute dt between refs
        idx1 = np.asarray(self._s1._search_ts(ts))
        idx2 = np.asarray(self._s2._search_ts(ts))
        v1 = self._s1._values_at(idx1)
        v2 = self._s2._values_at(idx2)
        t1 = np.asarray(self._s1._ts_at(idx1))
        t2 = np.asarray(self._s2._ts_at(idx2))
        dt = t2 - t1
        return list(zip(v1, v2, dt, strict=True))

    def _search_ts(self, ts_array: RealNumericArrayLike) -> IndicesLike:
        self._compute_bounds()
        t = np.asarray(ts_array)
        assert self._union_ts is not None
        return np.searchsorted(self._union_ts, t, side="right") - 1


class EpisodeTransform(ABC):
    """Transform an episode into a new episode."""

    @property
    @abstractmethod
    def keys(self) -> Sequence[str]:
        pass

    @abstractmethod
    def transform(self, name: str, episode: Episode) -> Signal[Any] | Any:
        pass


class TransformEpisode(Episode):
    """Transform an episode into a new view of the episode."""

    def __init__(self, episode: Episode, transform: EpisodeTransform, pass_through: bool = False) -> None:
        self._episode = episode
        self._transform = transform
        self._pass_through = pass_through

    @property
    def keys(self) -> Sequence[str]:
        # Preserve order: all transform keys first, then pass-through keys
        # from the original episode that are not overridden by the transform.
        ordered: list[str] = []
        seen: set[str] = set()
        for k in self._transform.keys:
            if k not in seen:
                ordered.append(k)
                seen.add(k)
        if self._pass_through:
            for k in self._episode.keys:
                if k not in seen:
                    ordered.append(k)
                    seen.add(k)
        return ordered

    def __getitem__(self, name: str) -> Signal[Any] | Any:
        # If the transform defines this key, it takes precedence.
        if name in self._transform.keys:
            return self._transform.transform(name, self._episode)
        if self._pass_through:
            return self._episode[name]
        raise KeyError(name)

    @property
    def meta(self) -> dict[str, Any]:
        return self._episode.meta


class _LazySequence(Sequence[U]):
    """Lazy, indexable view that applies `fn` on element access.

    - Supports `len()` and integer indexing.
    - Slicing returns another lazy view without materializing elements.
    """

    def __init__(self, seq: Sequence[T], fn: Callable[[T], U]) -> None:
        self._seq = seq
        self._fn = fn

    def __len__(self) -> int:
        return len(self._seq)

    def __getitem__(self, index: int | slice) -> U | "_LazySequence[U]":
        if isinstance(index, slice):
            return _LazySequence(self._seq[index], self._fn)
        return self._fn(self._seq[int(index)])


class Image:
    @staticmethod
    def resize(width: int,
               height: int,
               signal: Signal[np.ndarray],
               interpolation: int = cv2.INTER_LINEAR) -> Signal[np.ndarray]:
        """Return a Signal view with frames resized using OpenCV.

        Args:
            resolution: Target size as (width, height) for cv2.resize (W, H).
            signal: Input image Signal with frames shaped (H, W, 3), dtype uint8.
            interpolation: OpenCV interpolation flag (e.g., cv2.INTER_LINEAR).
        """
        interp_flag = int(interpolation)

        def per_frame(img: np.ndarray) -> np.ndarray:
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"Expected frame shape (H, W, 3), got {img.shape}")
            return cv2.resize(img, dsize=(width, height), interpolation=interp_flag)

        def fn(x: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
            return _LazySequence(x, per_frame)

        return Elementwise(signal, fn)

    @staticmethod
    def _resize_with_pad_pil(image: PilImage.Image, height: int, width: int, method: int) -> PilImage.Image:
        """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
        width without distortion by padding with zeros.
        Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
        """
        cur_width, cur_height = image.size
        if cur_width == width and cur_height == height:
            return image  # No need to resize if the image is already the correct size.

        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        resized_image = image.resize((resized_width, resized_height), resample=method)

        zero_image = PilImage.new(resized_image.mode, (width, height), 0)
        pad_height = max(0, int((height - resized_height) / 2))
        pad_width = max(0, int((width - resized_width) / 2))
        zero_image.paste(resized_image, (pad_width, pad_height))
        assert zero_image.size == (width, height)
        return zero_image

    @staticmethod
    def resize_with_pad(width: int,
                        height: int,
                        signal: Signal[np.ndarray],
                        method=PilImage.Resampling.BILINEAR) -> Signal[np.ndarray]:
        """Return a Signal view with frames resized-with-pad using PIL.

        Args:
            height: Target height (H).
            width: Target width (W).
            signal: Input image Signal with frames shaped (H, W, 3), dtype uint8.
            method: PIL resampling method (e.g., PilImage.Resampling.BILINEAR).
        """

        def per_frame(img: np.ndarray) -> np.ndarray:
            if img.shape[0] == height and img.shape[1] == width:
                return img  # No need to resize if the image is already the correct size.

            return np.array(
                Image._resize_with_pad_pil(PilImage.fromarray(img), height, width, method=method)
            )

        def fn(x: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
            return _LazySequence(x, per_frame)

        return Elementwise(signal, fn)


def concat(keys: Sequence[str], episode: Episode) -> Signal[np.ndarray]:
    """Concatenate multiple 1D float signals into a single 1D float array."""
    if len(keys) == 1:
        return episode[keys[0]]

    raise NotImplementedError("concat for multiple keys not implemented yet")
