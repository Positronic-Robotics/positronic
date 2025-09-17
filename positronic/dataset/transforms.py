from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Sequence, TypeVar

import cv2
import numpy as np
from PIL import Image as PilImage

from positronic import geom

from .dataset import Dataset
from .episode import Episode
from .signal import IndicesLike, Kind, RealNumericArrayLike, Signal, SignalMeta

T = TypeVar("T")
U = TypeVar("U")


def _is_nonempty_name(name: str | None) -> bool:
    """Return True when the provided feature name is a non-empty string."""
    return isinstance(name, str) and name != ""


def _format_join_component_name(names: Sequence[str] | None) -> tuple[str, bool]:
    """Collapse a sequence of names into display form for join-style transforms."""
    if names is None:
        return "", False
    filtered = [n for n in names if _is_nonempty_name(n)]
    if not filtered:
        return "", False
    if len(filtered) == 1:
        return filtered[0], True
    return "(" + " ".join(filtered) + ")", True


def _maybe_names(names: list[str]) -> list[str] | None:
    """Return names if any are non-empty; otherwise propagate None."""
    return names if any(_is_nonempty_name(n) for n in names) else None


def _format_offset_names(src_names: Sequence[str] | None, offsets: Sequence[int],
                         label_fn: Callable[[str, int], str]) -> list[str] | None:
    """Produce feature names for offset-based transforms with shared rules."""
    if src_names is None:
        return None

    def format_name(name: str, offset: int) -> str:
        if not _is_nonempty_name(name):
            return ""
        return name if offset == 0 else label_fn(name, offset)

    offsets = [int(o) for o in offsets]
    if len(offsets) == 1:
        return _maybe_names([format_name(name, offsets[0]) for name in src_names])

    joined, has_any = _format_join_component_name(src_names)
    if not has_any:
        return None
    return _maybe_names([format_name(joined, off) for off in offsets])


class Elementwise(Signal[U]):
    """Element-wise value transform view over a Signal.

    Wraps another `Signal[T]` and applies a function `f` to its values while
    preserving timestamps and ordering. Length and time indexing semantics are
    identical to the underlying signal.

    When the source signal exposes feature names, they are decorated with
    ``"{fn_name} of {source}"`` to indicate the transformation applied.
    Non-numeric signals fall back to the base metadata.
    """

    def __init__(self, signal: Signal[T], fn: Callable[[Sequence[T]], Sequence[U]], names: Sequence[str] | None = None):
        self._signal = signal
        self._fn = fn
        self._meta = None
        self._names_override = list(names) if names is not None else None

    def __len__(self) -> int:
        return len(self._signal)

    def _ts_at(self, indices: IndicesLike) -> Sequence[int] | np.ndarray:
        return self._signal._ts_at(indices)

    def _values_at(self, indices: IndicesLike) -> Sequence[U]:
        return self._fn(self._signal._values_at(indices))

    def _search_ts(self, ts_array: RealNumericArrayLike) -> IndicesLike:
        return self._signal._search_ts(ts_array)

    @staticmethod
    def _best_fn_name(fn: Callable[..., Any]) -> str:
        base = fn
        if isinstance(base, partial):
            base = base.func
        name = getattr(base, '__name__', None)
        if name is not None:
            return 'lambda' if name == '<lambda>' else name
        # Fallback to class name for callables
        cls = getattr(base, '__class__', None)
        if cls is not None and hasattr(cls, '__name__'):
            return cls.__name__
        return 'fn'

    @property
    def meta(self) -> SignalMeta:
        if self._meta is None:
            base = super().meta  # infers dtype/shape from transformed first element
            # Only craft names if numeric; otherwise use default
            if self._names_override is not None:
                self._meta = base.with_names(self._names_override)
            elif base.kind != Kind.NUMERIC:
                self._meta = base
            else:
                fn_name = Elementwise._best_fn_name(self._fn)
                src_names = self._signal.names
                if src_names is None:
                    name = fn_name
                else:
                    src_part = src_names[0] if len(src_names) == 1 else str(list(src_names))
                    name = f"{fn_name} of {src_part}"
                self._meta = base.with_names([name])
        return self._meta


class IndexOffsets(Signal[tuple]):
    """Join values (and optionally timestamps) at relative index offsets.

    For a base signal ``s`` and relative offsets ``D = [d1, d2, ..., dN]`` (each
    may be negative or positive), this view iterates over base indices ``i`` for
    which all ``i+dk`` are in-bounds. For each valid ``i`` the element is:

    - If ``include_ref_ts=False`` (default):
        ((v[i+d1], ..., v[i+dN]), t[i])
    - If ``include_ref_ts=True`` and ``N == 1`` (single offset):
        ((v[i+d1], t[i+d1]), t[i])  # legacy 2-tuple form for backwards-compatibility
    - If ``include_ref_ts=True`` and ``N > 1``:
        (((v[i+d1], ..., v[i+dN]), np.ndarray[int64]([t[i+d1], ..., t[i+dN]])), t[i])

    Notes:
      - The grouped timestamp array uses dtype int64 and has shape (N,).
      - This class does not modify values; it only aligns and groups neighbors.
      - Feature names are derived from the source signal. Individual offsets are
        labelled ``"index offset {d} of {name}"`` (or the original name when
        ``d == 0``). When multiple offsets are present, a component name forms
        ``"(name1 name2 ...)"`` following Join semantics.
    """

    def __init__(self,
                 signal: Signal[T],
                 *relative_indices: int,
                 include_ref_ts: bool = False,
                 names: Sequence[str] | None = None) -> None:
        self._signal = signal
        if len(relative_indices) == 0:
            raise ValueError("relative_indices must be non-empty")
        offs = np.asarray(relative_indices, dtype=np.int64)
        if offs.size == 0:
            raise ValueError("relative_indices must be non-empty")
        self._offs = offs
        self._min_off = int(np.min(self._offs))
        self._max_off = int(np.max(self._offs))
        self._include_ref_ts = bool(include_ref_ts)
        self._meta: SignalMeta | None = None
        self._names_override = list(names) if names is not None else None

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
            if self._include_ref_ts:
                ts_parts.append(self._signal._ts_at(idxs))

        n = len(self._offs)
        if not self._include_ref_ts:
            return list(zip(*vals_parts, strict=False)) if n > 1 else vals_parts[0]
        else:
            if n == 1:
                return list(zip(vals_parts[0], ts_parts[0], strict=False))

            ts = np.stack(ts_parts, axis=1)
            out = [(tuple(parts[i] for parts in vals_parts), ts[i]) for i in range(len(base))]
            return out

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

    @property
    def meta(self) -> SignalMeta:

        def index_offset_label(name: str, offset: int):
            return f"index offset {offset} of {name}"

        if self._meta is None:
            base_meta = super().meta
            if self._names_override is not None:
                names = self._names_override
            else:
                names = _format_offset_names(self._signal.names, self._offs, index_offset_label)
            self._meta = base_meta.with_names(names)
        return self._meta


class TimeOffsets(Signal[tuple]):
    """Sample at-or-before values at time offsets relative to each base timestamp.

    For deltas ``D = [d1, d2, ..., dN]`` and base timestamps ``ts[i]``, this
    view returns values and (optionally) the reference timestamps from the
    carried-back samples taken at times ``ts[i] + dk``.

    - If ``include_ref_ts=False`` (default):
        ((v_at(ts[i]+d1), ..., v_at(ts[i]+dN)), ts[i])
    - If ``include_ref_ts=True`` and ``N == 1`` (single delta):
        ((v_at(ts[i]+d1), t_at(ts[i]+d1)), ts[i])  # legacy 2-tuple form
    - If ``include_ref_ts=True`` and ``N > 1``:
        (((v_at(ts[i]+d1), ..., v_at(ts[i]+dN)), np.ndarray[int64]([t_at(ts[i]+d1), ..., t_at(ts[i]+dN)])), ts[i])

    Semantics:
      - Negative deltas can drop initial rows if ``ts[i]+d`` precedes the first
        timestamp.
      - Non-negative deltas clamp to the last element when the shifted time
        exceeds the last timestamp.
      - Grouped reference timestamp arrays always use dtype int64 and shape (N,).
      - Names mirror ``IndexOffsets`` semantics but use ``"time offset {sec}"``.
        Zero offsets retain the original source name.
    """

    def __init__(self,
                 signal: Signal[T],
                 *deltas_ts: int,
                 include_ref_ts: bool = False,
                 names: Sequence[str] | None = None) -> None:
        self._signal = signal
        if len(deltas_ts) == 0:
            raise ValueError("deltas_ts must be non-empty")
        offs = np.asarray(deltas_ts, dtype=np.int64)
        if offs.size == 0:
            raise ValueError("deltas_ts must be non-empty")
        self._deltas = offs
        self._include_ref_ts = bool(include_ref_ts)
        self._bounds_ready = False
        self._start_offset = 0
        self._last_index = -1
        self._meta: SignalMeta | None = None
        self._names_override = list(names) if names is not None else None

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
            floor_idx = int(self._signal._search_ts([thr])[0])
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
        ref_ts = np.asarray(self._signal._ts_at(base), dtype=np.int64)

        vals_parts = []
        ts_parts = []
        for d in self._deltas:
            target_ts = ref_ts + int(d)
            idx = self._signal._search_ts(target_ts)
            vals_parts.append(self._signal._values_at(idx))
            if self._include_ref_ts:
                ts_parts.append(self._signal._ts_at(idx))

        n = len(self._deltas)
        if not self._include_ref_ts:
            return list(zip(*vals_parts, strict=False)) if n > 1 else vals_parts[0]

        if n == 1:
            return list(zip(vals_parts[0], ts_parts[0], strict=False))

        ts = np.stack([np.asarray(t, dtype=np.int64) for t in ts_parts], axis=1)
        out = [(tuple(parts[i] for parts in vals_parts), ts[i]) for i in range(len(base))]
        return out

    def _search_ts(self, ts_array: RealNumericArrayLike) -> IndicesLike:
        self._compute_bounds()
        parent_idx = np.asarray(self._signal._search_ts(ts_array))
        if (self._deltas < 0).any():
            shifted = parent_idx - self._start_offset
            shifted[parent_idx < self._start_offset] = -1
            return shifted
        return parent_idx

    @property
    def meta(self) -> SignalMeta:

        def time_offset_label(name: str, offset: int):
            return f"time offset {offset / 1e9:.2f} sec of {name}"

        if self._meta is None:
            base_meta = super().meta
            if self._names_override is not None:
                names = self._names_override
            else:
                names = _format_offset_names(self._signal.names, self._deltas, time_offset_label)
            self._meta = base_meta.with_names(names)
        return self._meta


def _ts_at_index(sig: Signal[T], idx: int) -> int:
    """Fetch a single timestamp at the given index as int."""
    return int(sig._ts_at([idx])[0])


def _first_idx_at_or_after(sig: Signal[T], ts: int) -> int:
    floor = int(sig._search_ts([ts])[0])
    if floor < 0:
        return 0
    floor_ts = _ts_at_index(sig, floor)
    return floor if floor_ts == ts else floor + 1


class Join(Signal[tuple]):
    """Join multiple signals on the union of their timestamps with carry-back.

    Semantics:
      - Reference timeline: sorted union of all signals' timestamps, starting
        from ``max(s_k.start_ts)``. Equal timestamps across signals are
        collapsed.
      - Values: at each union timestamp ``t``, return a tuple of carried-back
        values from each signal. If requested, also include reference timestamps
        for those carried-back samples.

      Shapes:
        - If ``include_ref_ts=False`` (default):
            ((v1, v2, ..., vN), t)
        - If ``include_ref_ts=True`` (N ≥ 2 by construction):
            (((v1, v2, ..., vN), np.ndarray[int64]([t1_ref, ..., tN_ref])), t)

      where ``vK`` is the value of signal K at the last timestamp at-or-before
      ``t`` and ``tK_ref`` is that reference timestamp. The grouped timestamp
      array is dtype int64 with shape (N,).

    Raises:
        ValueError: if fewer than two signals are provided.

    Names:
        When source signals provide feature names, each column preserves its
        source naming. Multi-column signals collapse their names as
        ``"(name1 name2 ...)"`` consistent with the semantics used by other
        multi-argument transforms. Signals without names propagate ``None``.
    """

    def __init__(self, *signals: Signal[Any], include_ref_ts: bool = False, names: Sequence[str] | None = None) -> None:
        if len(signals) < 2:
            raise ValueError("Join requires at least two signals")
        self._signals: tuple[Signal[Any], ...] = tuple(signals)
        self._include_ref_ts = bool(include_ref_ts)
        self._bounds_ready = False
        self._starts: list[int] = [0] * len(self._signals)
        self._length = 0
        self._union_ts: np.ndarray | None = None
        self._meta: SignalMeta | None = None
        self._names_override = list(names) if names is not None else None

    def _compute_bounds(self) -> None:
        if self._bounds_ready:
            return
        n_all = [len(s) for s in self._signals]
        if any(n == 0 for n in n_all):
            self._starts = n_all[:]  # irrelevant; establishes emptiness
            self._length = 0
            self._bounds_ready = True
            return
        start_ts = max(s.start_ts for s in self._signals)
        self._starts = [_first_idx_at_or_after(s, start_ts) for s in self._signals]
        # Collect timestamps from each signal starting at its aligned start
        ts_arrays = []
        for s, st in zip(self._signals, self._starts):
            if st < len(s):
                idxs = np.arange(st, len(s), dtype=np.int64)
                ts_arrays.append(s._ts_at(idxs))
        if len(ts_arrays) == 0:
            self._union_ts = np.empty((0, ), dtype=np.int64)
        else:
            all_ts = np.concatenate(ts_arrays).astype(np.int64, copy=False)
            self._union_ts = np.unique(all_ts)
        self._length = int(self._union_ts.shape[0])
        self._bounds_ready = True

    # Note: previous 2-way merge helper removed; union now built via numpy unique.

    def __len__(self) -> int:
        self._compute_bounds()
        return self._length

    def _ts_at(self, indices: IndicesLike) -> Sequence[int] | np.ndarray:
        self._compute_bounds()
        idxs = np.asarray(indices)
        return self._union_ts[idxs]

    def _values_at(self, indices: IndicesLike):
        ts = np.asarray(self._ts_at(indices), dtype=np.int64)
        # For each signal, sample at-or-before these timestamps
        idx_all = [s._search_ts(ts) for s in self._signals]
        vals_all = [s._values_at(idx) for s, idx in zip(self._signals, idx_all)]
        tss_all = [s._ts_at(idx) for s, idx in zip(self._signals, idx_all)]

        if not self._include_ref_ts:
            return [tuple(row) for row in zip(*vals_all, strict=False)]
        else:
            ts_mat = np.stack([np.asarray(t, dtype=np.int64) for t in tss_all], axis=1)
            return [(tuple(row_vals), ts_mat[i]) for i, row_vals in enumerate(zip(*vals_all, strict=False))]

    def _search_ts(self, ts_array: RealNumericArrayLike) -> IndicesLike:
        self._compute_bounds()
        assert self._union_ts is not None
        return np.searchsorted(self._union_ts, ts_array, side="right") - 1

    @property
    def meta(self) -> SignalMeta:
        if self._meta is None:
            base_meta = super().meta
            if self._names_override is not None:
                names = self._names_override
            else:
                parts = [_format_join_component_name(sig.names) for sig in self._signals]
                names = [name if has else "" for name, has in parts] if any(has for _, has in parts) else None
            self._meta = base_meta.with_names(names)
        return self._meta


class EpisodeTransform(ABC):
    """Transform an episode into a new episode."""

    @property
    @abstractmethod
    def keys(self) -> Sequence[str]:
        """Keys that this transform generates."""
        ...

    @abstractmethod
    def transform(self, name: str, episode: Episode) -> Signal[Any] | Any:
        """For given output key, return the transformed signal or static value."""
        ...


class TransformedEpisode(Episode):
    """Transform an episode into a new view of the episode.

    Supports one or more transforms. When multiple transforms are provided,
    their keys are concatenated in the given order. For duplicate keys across
    transforms, the first transform providing the key takes precedence.

    If ``pass_through`` is True, any keys from the underlying episode that are
    not provided by the transforms are appended after the transformed keys.
    """

    def __init__(self, episode: Episode, *transforms: EpisodeTransform, pass_through: bool = False) -> None:
        if not transforms:
            raise ValueError("TransformedEpisode requires at least one transform")
        self._episode = episode
        self._transforms: tuple[EpisodeTransform, ...] = tuple(transforms)
        self._pass_through = pass_through

    @property
    def keys(self) -> Sequence[str]:
        # Preserve order across all transforms first, then pass-through keys
        # from the original episode that are not overridden by any transform.
        ordered: list[str] = []
        seen: set[str] = set()
        for tf in self._transforms:
            for k in tf.keys:
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
        # If any transform defines this key, the first one takes precedence.
        for tf in self._transforms:
            if name in tf.keys:
                return tf.transform(name, self._episode)
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

        return Elementwise(signal, fn, names=['height', 'width', 'channel'])

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
    def resize_with_pad_per_frame(width: int, height: int, method, img: np.ndarray) -> np.ndarray:
        if img.shape[0] == height and img.shape[1] == width:
            return img  # No need to resize if the image is already the correct size.

        return np.array(Image._resize_with_pad_pil(PilImage.fromarray(img), height, width, method=method))

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

        def fn(x: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
            return _LazySequence(x, partial(Image.resize_with_pad_per_frame, width, height, method))

        return Elementwise(signal, fn, names=['height', 'width', 'channel'])


def _concat_per_frame(dtype: np.dtype | None, x: Sequence[tuple]) -> np.ndarray:
    """Pickable callable that concatenates multiple array signals into a single array signal."""
    # x is a sequence of tuples (v1, v2, ..., vN) for the requested indices.
    # High-performance path: preallocate (batch, total_dim) and fill via slicing.
    batch = len(x)
    if batch == 0:
        return np.empty((0, 0), dtype=dtype or np.float32)

    # Infer per-signal dimensions and dtype from the first row
    first_parts = [np.asarray(v) for v in x[0]]
    dims = [p.size for p in first_parts]
    offsets = np.cumsum([0] + dims[:-1]) if dims else [0]
    total_dim = int(sum(dims))
    if dtype is None:
        dtype = np.result_type(*[p.dtype for p in first_parts]) if first_parts else np.float32
    out = np.empty((batch, total_dim), dtype=dtype)

    for j, p in enumerate(first_parts):  # Fill first row
        out[0, offsets[j]:offsets[j] + dims[j]] = p.ravel().astype(dtype, copy=False)
    for i in range(1, batch):  # Fill remaining rows
        row = x[i]
        for j, v in enumerate(row):
            arr = np.asarray(v)
            if arr.size != dims[j]:
                raise ValueError("concat: inconsistent vector size across rows")
            out[i, offsets[j]:offsets[j] + dims[j]] = arr.ravel().astype(dtype, copy=False)
    return out


def concat(*signals, dtype: np.dtype | str | None = None, names: Sequence[str] | None = None) -> Signal[np.ndarray]:
    """Concatenate multiple 1D array signals into a single array signal.

    - Aligns signals on the union of timestamps with carry-back semantics.
    - Values are vector-wise concatenations of each signal's values at-or-before t.
    - For batched requests, returns a single 2D array (batch, dim).
    """
    n = len(signals)
    if n == 0:
        raise ValueError("concat requires at least one key")
    if n == 1:
        return signals[0]

    if names is None:
        joined_names: list[str] = []
        any_named = False
        for sig in signals:
            joined, has = _format_join_component_name(sig.names)
            joined_names.append(joined)
            any_named = any_named or has
        names = joined_names if any_named else None
    return Elementwise(Join(*signals), partial(_concat_per_frame, dtype), names=_maybe_names(names or []))


def _astype_per_frame(dtype: np.dtype, x: np.ndarray) -> np.ndarray:
    """Pickable callable that casts arrays to a target dtype."""
    arr = np.asarray(x)
    if arr.dtype == dtype:
        return arr
    return arr.astype(dtype, copy=False)


def astype(signal: Signal[np.ndarray], dtype: np.dtype) -> Signal[np.ndarray]:
    """Return a Signal view that casts batched values to a given dtype."""
    return Elementwise(signal, partial(_astype_per_frame, dtype), names=signal.names)


class _PairwiseMap:

    def __init__(self, op: Callable[[Any, Any], Any]):
        self._op = op

    def __call__(self, rows: Sequence[tuple]) -> Sequence[Any]:
        out: list[Any] = []
        for a, b in rows:
            out.append(self._op(a, b))
        return out


def pairwise(a: Signal[Any],
             b: Signal[Any],
             op: Callable[[Any, Any], Any],
             names: Sequence[str] | None = None) -> Signal[Any]:
    """Apply a binary operation pairwise across two signals aligned on time.

    - Aligns `a` and `b` on the union of timestamps with carry-back semantics.
    - Applies `op(a_value, b_value)` per row and returns a new Signal view of results.
    """
    return Elementwise(Join(a, b), _PairwiseMap(op), names=names)


def recode_rotation(rep_from: geom.Rotation.Representation, rep_to: geom.Rotation.Representation,
                    signal: Signal[np.ndarray]) -> Signal[np.ndarray]:
    """Return a Signal view with rotation vectors recoded to a different representation.

    Args:
        rep_from: Input rotation representation.
        rep_to: Output rotation representation.
        signal: Input Signal with frames shaped (dim,), where dim depends on rep_from.
    """
    if rep_from == rep_to:
        return signal

    def fn(x: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        # TODO: Use batch conversion instead.
        return _LazySequence(x, lambda v: geom.Rotation.create_from(v, rep_from).to(rep_to).flatten())

    return Elementwise(signal, fn)


class TransformedDataset(Dataset):
    """Transform a dataset into a new view of the dataset."""

    def __init__(self, dataset: Dataset, *transforms: EpisodeTransform, pass_through: bool = False) -> None:
        self._dataset = dataset
        self._transforms = transforms
        self._pass_through = pass_through
        self._meta = None

    def __len__(self) -> int:
        return len(self._dataset)

    def _get_episode(self, index: int) -> Episode:
        episode = self._dataset[index]
        return TransformedEpisode(episode, *self._transforms, pass_through=self._pass_through)

    @property
    def signals_meta(self) -> dict[str, SignalMeta]:
        if self._meta is None:
            ep = self[0]
            self._meta = {name: ep[name].meta for name in ep.signals.keys()}
        return self._meta
