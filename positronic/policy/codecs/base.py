import collections.abc as cabc
from typing import Any, ClassVar, final, overload

import numpy as np
from positronic_model_server.spec import PAR, SEQ

from positronic import telemetry, telemetry_keys
from positronic.dataset.transforms.episode import Derive, EpisodeTransform
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.policy.base import Obs, ProcessorRun, Step
from positronic.utils import merge_dicts


class Codec:
    """Base class for observation/action codecs.

    Subclasses override ``encode`` (observation encoding) and/or ``_decode_single``
    (action decoding). The ``training_encoder`` property
    returns an ``EpisodeTransform`` used by the training pipeline to derive dataset columns.

    Reserved ``meta`` key:

    ``image_sizes``
        The image dimensions this codec encodes to. Either a ``(width, height)`` tuple (same
        size for all images) or a dict mapping raw input keys to ``(width, height)`` tuples.
    """

    IMAGE_SIZES = 'image_sizes'
    WIRE_NAME: ClassVar[str]
    WIRE_VERSION: ClassVar[int] = 1

    def encode(self, data: dict) -> dict:
        return {}

    def decode(self, data: Any) -> Any:
        if isinstance(data, list):
            return [self.decode(d) for d in data]
        return self._decode_single(data)

    def _decode_single(self, data: dict) -> dict:
        return {}

    @property
    def training_encoder(self) -> EpisodeTransform:
        # TODO: Remove this interface and its composition after all model recipes build training independently.
        return Derive()

    @property
    def meta(self) -> dict:
        return {}

    @overload
    def wrap(self, function: cabc.Callable[[dict], Any]) -> cabc.Callable[[Obs], Any]: ...

    @overload
    def wrap(self, function: ProcessorRun[Obs, Any]) -> ProcessorRun[Obs, Any]: ...

    def wrap(
        self, function: cabc.Callable[[dict], Any] | ProcessorRun[Obs, Any]
    ) -> cabc.Callable[[Obs], Any] | ProcessorRun[Obs, Any]:
        """Encode inputs and decode outputs around a callable or a primed processor run.

        Steps retain their wake-up time; only nonempty commands are decoded. The caller owns the
        wrapped dependency, including closing it when it is a generator.
        """
        if isinstance(function, cabc.Generator):
            run = self._wrap_run(function)
            next(run)
            return run

        encode = telemetry.traced(
            telemetry_keys.SPAN_POLICY_ENCODE, **{telemetry_keys.ATTR_CODEC: type(self).__name__}
        )(self.encode)

        @telemetry.traced(telemetry.component_name(self))
        def call(obs: Obs) -> Any:
            encoded = encode(dict(obs))
            result = function(encoded)
            if isinstance(result, Step):
                return Step(self.decode(dict(result.commands)), result.resume_at_ns) if result.commands else result
            return self.decode(result) if result is not None else None

        return call

    def _wrap_run(self, inner: ProcessorRun[Obs, Any]) -> ProcessorRun[Obs, Any]:
        call = self.wrap(inner.send)
        obs = yield
        while True:
            try:
                result = call(obs)
            except StopIteration:
                return
            obs = yield result

    def to_spec(self) -> dict[str, Any]:
        raise NotImplementedError(f'{type(self).__name__} has no wire spec')

    @final
    def __or__(self, other) -> Any:
        if isinstance(other, Codec):
            return _ComposedCodec(self, other)
        return NotImplemented

    @final
    def __and__(self, other):
        if isinstance(other, Codec):
            return _ParallelCodec(self, other)
        return NotImplemented


def _meta_conflicts(left: dict, right: dict, prefix: str = '') -> list[str]:
    """``key: left != right`` for every leaf the two metas declare differently. Nested dicts merge per key,
    so only leaves can conflict."""
    found = []
    for key in left.keys() & right.keys():
        a, b = left[key], right[key]
        if isinstance(a, dict) and isinstance(b, dict):
            found += _meta_conflicts(a, b, f'{prefix}{key}.')
        elif isinstance(a, dict) or isinstance(b, dict) or not np.array_equal(a, b):
            found.append(f'{prefix}{key}: {a!r} != {b!r}')
    return found


def _merged_meta(left: dict, right: dict) -> dict:
    """Two codecs' metadata as one dict.

    A leaf both declare differently has no merged answer, so it raises rather than keeping one: the survivor
    would describe a pipeline neither codec implements. Declare the composition as a single value instead.
    """
    conflicts = _meta_conflicts(left, right)
    if conflicts:
        raise ValueError(f'composed codecs disagree on metadata — {"; ".join(sorted(conflicts))}')
    result: dict[str, Any] = {}
    merge_dicts(result, left)
    merge_dicts(result, right)
    return result


class _ComposedCodec(Codec):
    """Two codecs composed via ``|``. Encodes left-to-right, decodes right-to-left."""

    def __init__(self, left: Codec, right: Codec):
        self._left = left
        self._right = right

    def encode(self, data):
        return self._right.encode(self._left.encode(data))

    def decode(self, data):
        return self._left.decode(self._right.decode(data))

    @property
    def training_encoder(self):
        return self._left.training_encoder | self._right.training_encoder

    @property
    def meta(self):
        left, right = self._left.meta, self._right.meta
        # Sequential frame transforms compose, so either transform's frame metadata alone is incorrect.
        if roboarm_keys.EE_FRAME in left and roboarm_keys.EE_FRAME in right:
            raise ValueError(
                f'sequential codecs both declare {roboarm_keys.EE_FRAME}: poses come out at the product of both '
                f'moves, which neither {left[roboarm_keys.EE_FRAME]} nor {right[roboarm_keys.EE_FRAME]} names'
            )
        return _merged_meta(left, right)

    def to_spec(self):
        return {SEQ: [self._left.to_spec(), self._right.to_spec()]}


class _ParallelCodec(Codec):
    """Two codecs composed via ``&``. Both see the same input, outputs merged."""

    def __init__(self, left: Codec, right: Codec):
        self._left = left
        self._right = right

    def encode(self, data):
        return {**self._left.encode(data), **self._right.encode(data)}

    def decode(self, data):
        left_out = self._left.decode(data)
        right_out = self._right.decode(data)
        if isinstance(data, list):
            return [{**lf, **rt} for lf, rt in zip(left_out, right_out, strict=True)]
        return {**left_out, **right_out}

    @property
    def training_encoder(self):
        return self._left.training_encoder & self._right.training_encoder

    @property
    def meta(self):
        return _merged_meta(self._left.meta, self._right.meta)

    def to_spec(self):
        return {PAR: [self._left.to_spec(), self._right.to_spec()]}
