from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

from ..episode import Episode
from ..signal import Signal


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


class KeyFuncEpisodeTransform(EpisodeTransform):
    """Transform an episode using a dictionary of key-function pairs."""

    def __init__(self, **transforms: Callable[[Episode], Any]):
        self._transform_fns = transforms

    @property
    def keys(self) -> Sequence[str]:
        return self._transform_fns.keys()

    def transform(self, name: str, episode: Episode) -> Signal[Any] | Any:
        if name not in self._transform_fns:
            raise KeyError(f'Unknown key: {name}, expected one of {self._transform_fns.keys()}')
        return self._transform_fns[name](episode)


class TransformedEpisode(Episode):
    """Transform an episode into a new view of the episode.

    Supports one or more transforms. When multiple transforms are provided,
    their keys are concatenated in the given order. For duplicate keys across
    transforms, the first transform providing the key takes precedence.

    If ``pass_through`` is True, any keys from the underlying episode that are
    not provided by the transforms are appended after the transformed keys.
    Alternatively, a list of key names can be provided to selectively pass
    through only the listed keys from the original episode.
    """

    def __init__(self, episode: Episode, *transforms: EpisodeTransform, pass_through: bool | list[str] = False):
        if not transforms:
            raise ValueError('TransformedEpisode requires at least one transform')
        self._episode = episode
        self._transforms = tuple(transforms)
        if isinstance(pass_through, bool):
            self._pass_through_all = pass_through
            self._pass_through_keys = set()
        else:
            self._pass_through_all = False
            self._pass_through_keys = set(pass_through)

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
        for k in self._episode.keys:
            if k not in seen and (self._pass_through_all or k in self._pass_through_keys):
                ordered.append(k)
                seen.add(k)
        return ordered

    def __getitem__(self, name: str) -> Signal[Any] | Any:
        # If any transform defines this key, the first one takes precedence.
        for tf in self._transforms:
            if name in tf.keys:
                return tf.transform(name, self._episode)
        if self._pass_through_all or (self._pass_through_keys and name in self._pass_through_keys):
            return self._episode[name]
        raise KeyError(name)

    @property
    def meta(self) -> dict[str, Any]:
        return self._episode.meta
