from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any

from positronic.dataset.transforms import signals
from positronic.dataset.transforms.signals import NpSignal
from positronic.utils import merge_dicts

from ..episode import Episode, EpisodeContainer
from ..signal import Signal


class EpisodeTransform(ABC):
    """Transform an episode into a new episode."""

    @abstractmethod
    def __call__(self, episode: Episode) -> Episode:
        """Transform an episode into a new episode."""
        ...

    @property
    def meta(self) -> dict[str, Any]:
        """Metadata for this transform. Transformed episodes can have different metadata from transform metadata."""
        return {}


class _LazyDeriveEpisode(Episode):
    """Episode that computes derived values lazily on first access."""

    def __init__(self, base: Episode, transforms: dict[str, Callable[[Episode], Any]]):
        self._base = base
        self._transforms = transforms
        self._computed: dict[str, Any] = {}

    def __iter__(self) -> Iterator[str]:
        yield from self._transforms.keys()

    def __len__(self) -> int:
        return len(self._transforms)

    def __getitem__(self, name: str) -> Any:
        if name not in self._transforms:
            raise KeyError(name)
        if name not in self._computed:
            fn = self._transforms[name]
            try:
                self._computed[name] = fn(self._base)
            except Exception as e:
                raise ValueError(f'Failed to apply transform {name} to episode {self._base.meta}.') from e
        return self._computed[name]

    @property
    def meta(self) -> dict:
        return self._base.meta


class Derive(EpisodeTransform):
    """Derive new episode keys by applying functions to the input episode.

    Each keyword argument maps an output key name to a function that takes an Episode
    and returns either a Signal or a static value. Values are computed lazily on first access.

    Example:
        Derive(
            state=Concat('joint_q', 'ee_pose'),
            label=FromValue('pick_place')
        )
    """

    def __init__(self, **transforms: Callable[[Episode], Any]):
        self._transforms = transforms

    def __call__(self, episode: Episode) -> Episode:
        return _LazyDeriveEpisode(episode, self._transforms)


class _LazyMergedEpisode(Episode):
    """Episode that lazily merges results from multiple episodes."""

    def __init__(self, episodes: tuple[Episode, ...], meta: dict):
        self._episodes = episodes  # First takes precedence
        self._meta = meta

    def __iter__(self) -> Iterator[str]:
        seen: set[str] = set()
        for ep in self._episodes:
            for k in ep:
                if k not in seen:
                    seen.add(k)
                    yield k

    def __len__(self) -> int:
        return len({k for ep in self._episodes for k in ep})

    def __getitem__(self, name: str) -> Any:
        for ep in self._episodes:
            try:
                return ep[name]
            except KeyError:
                continue
        raise KeyError(name)

    @property
    def meta(self) -> dict:
        return self._meta


class Group(EpisodeTransform):
    """Group multiple transforms into a single transform for parallel application.

    Applies all transforms to the same input episode and merges their results lazily.
    If transforms produce overlapping keys, the first transform in the list takes precedence.

    Example:
        Group(observation_encoder, action_encoder, task_labeler)
    """

    def __init__(self, *transforms: EpisodeTransform):
        self._transforms = tuple(transforms)

    def __call__(self, episode: Episode) -> Episode:
        results = tuple(tf(episode) for tf in self._transforms)
        return _LazyMergedEpisode(results, episode.meta)

    @property
    def meta(self) -> dict[str, Any]:
        result = {}
        for tf in reversed(self._transforms):
            result = merge_dicts(result, tf.meta)
        return result


class Eager(EpisodeTransform):
    """Force eager evaluation of a wrapped transform.

    Use this when you want to ensure all values are computed upfront,
    for example when debugging or when you know all values will be accessed.

    Example:
        Eager(Derive(task=FromValue('...'), units=calculate_units))
    """

    def __init__(self, transform: EpisodeTransform):
        self._transform = transform

    def __call__(self, episode: Episode) -> Episode:
        lazy_result = self._transform(episode)
        # Force evaluation by accessing all keys
        return EpisodeContainer({k: lazy_result[k] for k in lazy_result}, lazy_result.meta)

    @property
    def meta(self) -> dict[str, Any]:
        return self._transform.meta


class Rename(EpisodeTransform):
    """Rename keys in an episode.

    Only renamed keys are included in the output episode.

    Example:
        # Keyword form (only works for identifier-like names):
        Rename(state='robot_state.q', handcam='image.wrist')

        # For keys that are not valid Python identifiers (e.g. contain dots),
        # use dict expansion. The mapping is new_key='old_key' (output -> input):
        Rename(**{'robot_state.q': 'robot_state.joints', 'image.handcam': 'image.wrist'})
    """

    def __init__(self, **mapping: str):
        """Create a rename mapping for episode keys.

        Args:
            **mapping: Keyword arguments mapping ``new_key`` -> ``old_key`` (output key -> input key).
                For keys that are not valid Python identifiers (like names containing dots),
                pass them via dict expansion: ``Rename(**{'a.b': 'c.d'})``.
        """
        if not mapping:
            raise ValueError('Rename requires at least one mapping entry')
        self._mapping: dict[str, str] = dict(mapping)

    def __call__(self, episode: Episode) -> Episode:
        res = {k: episode[v] for k, v in self._mapping.items()}
        return EpisodeContainer(res, episode.meta)


class Identity(EpisodeTransform):
    """Select specific keys from an episode, or pass through unchanged if no keys specified.

    Example:
        Identity('robot_state', 'image')  # Only keep these two keys
        Identity()  # Pass through all keys unchanged
    """

    def __init__(self, select: list[str] = None, remove: list[str] = None):
        """
        Args:
            select: Keys to include. If empty, returns the original episode unchanged.
            remove: Keys to exclude.
        """
        self._select = set(select or [])
        self._remove = set(remove or [])

    def __call__(self, episode: Episode) -> Episode:
        if not self._select and not self._remove:
            return episode

        container = {}
        for k, v in episode.items():
            if k in self._remove:
                continue
            if self._select and k not in self._select:
                continue
            container[k] = v
        return EpisodeContainer(container, episode.meta)


class Concat:
    """Helper for concatenating multiple episode signals into a single array signal.

    This is a callable helper (not an EpisodeTransform) typically used within Derive.

    Example:
        Derive(ee_pose=Concat('ee_translation', 'ee_quaternion'))
    """

    def __init__(self, *features: str) -> None:
        """
        Args:
            *features: Episode keys to concatenate in order
        """
        self._features = features

    def __call__(self, episode: Episode) -> NpSignal:
        return signals.concat(*[episode[k] for k in self._features])


class FromValue:
    """Helper that returns a constant value, ignoring the episode.

    This is a callable helper (not an EpisodeTransform) typically used within Derive
    to add static/constant values to episodes.

    Example:
        Derive(task=FromValue('pick and place the green cube'))
    """

    def __init__(self, value: Any):
        """
        Args:
            value: The constant value to return
        """
        self._value = value

    def __call__(self, episode: Episode) -> Any:
        return self._value


class TransformedEpisode(Episode):
    """Lazily transform an episode into a new view of the episode.

    Applies transforms sequentially (chained), where each transform receives the output
    of the previous transform. The transforms are applied in reverse order to match the
    precedence behavior where the first transform in the list takes precedence for duplicate keys.

    The transformation is lazy - transforms are only executed on first access to any key,
    and the result is cached for subsequent accesses.
    """

    _MISSING = object()

    def __init__(self, episode: Episode, *transforms: EpisodeTransform):
        if not transforms:
            raise ValueError('TransformedEpisode requires at least one transform')
        self._episode = episode
        self._transforms = tuple(transforms)

        self._cache: Episode | None = None

    def _get_cache(self) -> Episode:
        if self._cache is None:
            self._cache = self._episode
            for tf in reversed(self._transforms):
                self._cache = tf(self._cache)
        return self._cache

    def __iter__(self) -> Iterator[str]:
        yield from self._get_cache().keys()

    def __len__(self) -> int:
        return len(self._get_cache())

    def __getitem__(self, name: str) -> Signal[Any] | Any:
        return self._get_cache()[name]

    @property
    def meta(self) -> dict[str, Any]:
        # TODO: Should we add metadata from the transforms?
        return self._episode.meta
