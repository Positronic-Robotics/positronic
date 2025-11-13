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


class Derive(EpisodeTransform):
    """Derive new episode keys by applying functions to the input episode.

    Each keyword argument maps an output key name to a function that takes an Episode
    and returns either a Signal or a static value.

    Example:
        Derive(
            state=Concat('joint_q', 'ee_pose'),
            label=FromValue('pick_place')
        )
    """

    def __init__(self, **transforms: Callable[[Episode], Any]):
        self._transforms = transforms

    def __call__(self, episode: Episode) -> Episode:
        # TODO: Should we lazy-fy this?
        data = {name: fn(episode) for name, fn in self._transforms.items()}
        return EpisodeContainer(data, episode.meta)


class Group(EpisodeTransform):
    """Group multiple transforms into a single transform for parallel application.

    Applies all transforms to the same input episode and merges their results.
    If transforms produce overlapping keys, the first transform in the list takes precedence.

    Example:
        Group(observation_encoder, action_encoder, task_labeler)
    """

    def __init__(self, *transforms: EpisodeTransform):
        self._transforms = tuple(transforms)

    def __call__(self, episode: Episode) -> Episode:
        res = {}
        for tf in reversed(self._transforms):
            res.update(tf(episode))
        return EpisodeContainer(res, episode.meta)

    @property
    def meta(self) -> dict[str, Any]:
        result = {}
        for tf in reversed(self._transforms):
            result = merge_dicts(result, tf.meta)
        return result


class Rename(EpisodeTransform):
    """Rename keys in an episode.

    Only renamed keys are included in the output episode.

    Example:
        # Rename 'robot_state.q' to 'robot_joints' and 'image.wrist' to 'image.handcam'
        Rename({'robot_joints': 'robot_state.q', 'image.handcam': 'image.wrist'})
    """

    def __init__(self, mapping: dict[str, str]):
        """
        Args:
            mapping: Dictionary mapping new_key -> old_key (output key -> input key)
        """
        self._mapping = mapping

    def __call__(self, episode: Episode) -> Episode:
        res = {k: episode[v] for k, v in self._mapping.items()}
        return EpisodeContainer(res, episode.meta)


class Identity(EpisodeTransform):
    """Select specific keys from an episode, or pass through unchanged if no keys specified.

    Example:
        Identity('robot_state', 'image')  # Only keep these two keys
        Identity()  # Pass through all keys unchanged
    """

    def __init__(self, *features: str):
        """
        Args:
            *features: Keys to include. If empty, returns the original episode unchanged.
        """
        self._features = set(features)

    def __call__(self, episode: Episode) -> Episode:
        if not self._features:
            return episode
        return EpisodeContainer({k: episode[k] for k in self._features}, episode.meta)


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
