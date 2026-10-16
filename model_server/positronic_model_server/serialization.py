"""Wire serialization helpers for numpy arrays, images and standard Python types.

Supports:
- built-in scalars: `str`, `int`, `float`, `bool`, `None`
- containers: `dict` / `list` / `tuple` recursively composed of supported values
- numeric numpy values: `numpy.ndarray` and `numpy` scalar types
- images JPEG-encoded with ``encode_jpeg``

Nothing here knows a domain type. A boundary that carries one writes its own msgpack hooks and
delegates to ``pack`` / ``unpack`` for the rest.
"""

import collections.abc as cabc
import functools
import hashlib
import io
from dataclasses import dataclass
from typing import Any

import msgpack
import numpy as np
from PIL import Image as PilImage

# The envelope each encoded value travels in: a marker naming the type, and the fields carrying it.
_NDARRAY = b'__ndarray__'
_NPGENERIC = b'__npgeneric__'
_JPEG = b'__jpeg__'
_DATA = b'data'
_DTYPE = b'dtype'
_SHAPE = b'shape'
_FRAMES = b'frames'
_NDIM = b'ndim'
# Beside ``_FRAMES``: for each frame, the digest of a frame the receiver holds, or ``None`` for a frame in ``_FRAMES``.
_REFS = b'refs'

DEFAULT_JPEG_QUALITY = 90


@dataclass(frozen=True)
class JpegEncoding:
    """An explicit mapping/list path to an RGB image; an empty path selects the whole value."""

    path: tuple[str | int, ...]
    quality: int = DEFAULT_JPEG_QUALITY

    def __post_init__(self) -> None:
        if any(type(part) not in (str, int) for part in self.path):
            raise ValueError('Image paths contain only mapping keys and list indices')
        if type(self.quality) is not int or not 0 <= self.quality <= 100:
            raise ValueError('JPEG quality must be an integer between 0 and 100')

    def encode(self, value: Any) -> Any:
        """Copy the selected container path and encode its leaf, preserving unrelated values."""

        def at(node: Any, path: tuple[str | int, ...]) -> Any:
            if not path:
                if not isinstance(node, np.ndarray):
                    raise ValueError(f'JPEG path {self.path!r} does not select an array')
                return encode_jpeg(node, self.quality)
            key, *rest = path
            if isinstance(node, cabc.Mapping):
                return {**node, key: at(node[key], tuple(rest))}
            if isinstance(node, list | tuple) and type(key) is int:
                copy = list(node)
                copy[key] = at(node[key], tuple(rest))
                return tuple(copy) if isinstance(node, tuple) else copy
            raise ValueError(f'JPEG path {self.path!r} does not match the value')

        return at(value, self.path)


@dataclass(frozen=True)
class EncodedFrame:
    """One frame and the JPEG bytes ``encode_jpeg`` made of it at ``quality``.

    ``pixels`` is a read-only copy, so the bytes stay valid for it when the caller changes its own array.
    """

    pixels: np.ndarray
    jpeg: bytes
    quality: int


def encode_images(value: Any, encodings: cabc.Sequence[JpegEncoding]) -> Any:
    """Encode only the explicitly selected images, in observations or model results."""
    for encoding in encodings:
        value = encoding.encode(value)
    return value


def encode_jpeg(
    image: np.ndarray, quality: int = DEFAULT_JPEG_QUALITY, encoded: cabc.Sequence[EncodedFrame] = ()
) -> dict[bytes, Any]:
    """JPEG-encode RGB images shaped ``(..., H, W, 3)``, casting pixels to uint8.

    Nonempty 3D/4D inputs use the v1/v2 marker. Extra leading dimensions and empty batches carry
    their full shape. JPEG changes pixels; ordinary arrays passed to ``serialise`` stay lossless.
    ``encoded`` holds frames of an earlier call. Each leading frame equal to the frame of the same index
    in ``encoded``, and encoded at the same quality, uses its JPEG bytes and is not encoded again.
    """
    if image.ndim < 3 or image.shape[-1] != 3 or min(image.shape[-3:-1]) < 1:
        raise ValueError('JPEG images must have shape (..., H, W, 3) with positive height and width')
    if type(quality) is not int or not 0 <= quality <= 100:
        raise ValueError('JPEG quality must be an integer between 0 and 100')
    frames = image.reshape((-1, *image.shape[-3:]))
    bufs = []
    for index, frame in enumerate(frames):
        if index < len(encoded):
            earlier = encoded[index]
            if earlier.quality == quality and np.array_equal(earlier.pixels, frame):
                bufs.append(earlier.jpeg)
                continue
        encoded = ()  # Only the leading frames use earlier bytes.
        buf = io.BytesIO()
        PilImage.fromarray(np.ascontiguousarray(frame, dtype=np.uint8)).save(buf, format='JPEG', quality=quality)
        bufs.append(buf.getvalue())
    if image.ndim > 4 or not bufs:
        return {_JPEG: True, _FRAMES: bufs, _SHAPE: image.shape}
    return {_JPEG: True, _FRAMES: bufs, _NDIM: int(image.ndim)}


def encoded_frames(image: np.ndarray, marker: cabc.Mapping[bytes, Any], quality: int) -> list[EncodedFrame]:
    """Each frame of ``image`` with its JPEG bytes in ``marker``, the ``encode_jpeg`` result of ``image`` at
    ``quality``."""
    frames = image.reshape((-1, *image.shape[-3:]))
    return [EncodedFrame(_snapshot(frame), jpeg, quality) for frame, jpeg in zip(frames, marker[_FRAMES], strict=True)]


def _snapshot(frame: np.ndarray) -> np.ndarray:
    copy = frame.copy()
    copy.flags.writeable = False
    return copy


def frame_digest(data: bytes) -> bytes:
    """The name of a frame in a reference: a hash of its JPEG bytes."""
    return hashlib.blake2b(data, digest_size=16).digest()


def _is_jpeg(value: Any) -> bool:
    return isinstance(value, cabc.Mapping) and _JPEG in value


def frame_digests(value: Any) -> set[bytes]:
    """The digests of the frames of every JPEG marker in ``value``, at any depth."""
    if _is_jpeg(value):
        return {frame_digest(buf) for buf in value[_FRAMES]}
    if isinstance(value, cabc.Mapping):
        return set().union(*(frame_digests(v) for v in value.values()))
    if isinstance(value, list | tuple):
        return set().union(*(frame_digests(v) for v in value))
    return set()


def reference_frames(value: Any, held: cabc.Container[bytes]) -> tuple[Any, int]:
    """``value`` with each frame that the receiver ``held`` sent as its digest, and the count of those frames.

    Only a frame of a ``(T, H, W, 3)`` stack is sent as a digest, and never the last frame of a stack: the newest
    frame always travels in full. ``value`` does not change.
    """
    if _is_jpeg(value):
        frames = list(value[_FRAMES])
        refs: list[bytes | None] = [None] * len(frames)
        if value.get(_NDIM) == 4:
            for i in range(len(frames) - 1):
                digest = frame_digest(frames[i])
                if digest in held:
                    frames[i], refs[i] = b'', digest
        count = sum(ref is not None for ref in refs)
        return ({**value, _FRAMES: frames, _REFS: refs} if count else value), count
    if isinstance(value, cabc.Mapping):
        items = [(k, *reference_frames(v, held)) for k, v in value.items()]
        return {k: v for k, v, _ in items}, sum(n for _, _, n in items)
    if isinstance(value, list | tuple):
        items = [reference_frames(v, held) for v in value]
        return type(value)(v for v, _ in items), sum(n for _, n in items)
    return value, 0


class UnknownFrame(KeyError):
    """A JPEG marker references a frame that the decoder does not hold."""


def _decode_jpeg(
    marker: dict, held: cabc.Mapping[bytes, np.ndarray] | None = None, decoded: dict[bytes, np.ndarray] | None = None
) -> np.ndarray:
    """Inverse of ``encode_jpeg``: decode per-frame JPEGs and restore the original shape.

    A frame sent as a digest comes from ``held``. Each frame decoded from its bytes goes into ``decoded`` under its
    digest.
    """
    if _SHAPE in marker and not marker[_FRAMES]:
        shape = tuple(marker[_SHAPE])
        if len(shape) < 4 or shape[-1] != 3 or min(shape[-3:-1]) < 1 or 0 not in shape[:-3]:
            raise ValueError('An empty JPEG batch must have an empty leading dimension')
        return np.empty(shape, dtype=np.uint8)
    refs = marker.get(_REFS) or [None] * len(marker[_FRAMES])
    decoded_frames = []
    for buf, ref in zip(marker[_FRAMES], refs, strict=True):
        if ref is not None:
            if held is None or ref not in held:
                raise UnknownFrame(ref)
            decoded_frames.append(held[ref])
            continue
        frame = np.asarray(PilImage.open(io.BytesIO(buf)))
        if decoded is not None:
            decoded[frame_digest(buf)] = frame
        decoded_frames.append(frame)
    frames = np.stack(decoded_frames)
    if _SHAPE in marker:
        return frames.reshape(marker[_SHAPE])
    return frames if marker[_NDIM] == 4 else frames[0]


def pack(obj):
    """msgpack's ``default`` hook: one value in its wire form, or unchanged when msgpack handles it."""
    if isinstance(obj, cabc.Mapping):
        return dict(obj)
    if isinstance(obj, np.ndarray | np.generic) and obj.dtype.kind in ('V', 'O', 'c'):
        raise ValueError(f'Unsupported dtype: {obj.dtype}')
    if isinstance(obj, np.ndarray):
        return {_NDARRAY: True, _DATA: obj.tobytes(), _DTYPE: obj.dtype.str, _SHAPE: obj.shape}
    if isinstance(obj, np.generic):
        return {_NPGENERIC: True, _DATA: obj.item(), _DTYPE: obj.dtype.str}
    return obj


def unpack(obj, held: cabc.Mapping[bytes, np.ndarray] | None = None, decoded: dict[bytes, np.ndarray] | None = None):
    """msgpack's ``object_hook``: one decoded mapping restored to the value it encodes.

    A JPEG frame sent as a digest comes from ``held``, and raises ``UnknownFrame`` when ``held`` does not hold it.
    Each JPEG frame decoded from its bytes goes into ``decoded`` under its digest.
    """
    if _NDARRAY in obj:
        return np.ndarray(buffer=obj[_DATA], dtype=np.dtype(obj[_DTYPE]), shape=obj[_SHAPE])
    if _NPGENERIC in obj:
        return np.dtype(obj[_DTYPE]).type(obj[_DATA])
    if _JPEG in obj:
        return _decode_jpeg(obj, held, decoded)
    return obj


def serialise(obj: Any) -> bytes:
    packed = msgpack.packb(obj, default=pack)
    assert packed is not None
    return packed


deserialise = functools.partial(msgpack.unpackb, object_hook=unpack)

# Aliases for consistency
serialize = serialise
deserialize = deserialise
