from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from ..episode import Episode, EpisodeContainer
from ..signal import Signal


class EpisodeTransform(ABC):
    """Transform an episode into a new episode."""

    @property
    @abstractmethod
    def keys(self) -> Sequence[str]:
        """Keys that this transform generates."""
        ...

    @abstractmethod
    def transform(self, episode: Episode) -> Episode:
        """Transform an episode and return a new episode with transformed signals and static values."""
        ...

    @property
    def meta(self) -> dict[str, Any]:
        """Metadata for this transform. Transformed episodes can have different metadata from transform metadata."""
        return {}


class KeyFuncEpisodeTransform(EpisodeTransform):
    """Transform an episode using a dictionary of key-function pairs."""

    def __init__(self, **transforms: Callable[[Episode], Any]):
        self._transform_fns = transforms

    @property
    def keys(self) -> Sequence[str]:
        return self._transform_fns.keys()

    def transform(self, episode: Episode) -> Episode:
        # TODO: Should we lazy-fy this?
        data = {name: fn(episode) for name, fn in self._transform_fns.items()}
        return EpisodeContainer(data, episode.meta)


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

    _MISSING = object()

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

        self._cache: dict[str, Any] = {}
        seen: set[str] = set()
        for tf in self._transforms:
            for k in tf.keys:
                if k not in seen:
                    self._cache[k] = self._MISSING
                    seen.add(k)
        for k in self._episode:
            if k not in seen and (self._pass_through_all or k in self._pass_through_keys):
                self._cache[k] = self._MISSING
                seen.add(k)

    def __iter__(self) -> Iterator[str]:
        yield from self._cache.keys()

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, name: str) -> Signal[Any] | Any:
        if name not in self._cache:
            raise KeyError(f'Key {name} not found in transformed episode. Available keys: {", ".join(self.keys())}')
        if self._cache[name] is not self._MISSING:
            return self._cache[name]

        for tf in self._transforms:
            if name in tf.keys:
                episode = tf.transform(self._episode)
                for k in episode:
                    self._cache[k] = episode[k]
                return self._cache[name]

        if self._pass_through_all or (self._pass_through_keys and name in self._pass_through_keys):
            return self._episode[name]
        raise KeyError(name)  # Should never happen

    @property
    def meta(self) -> dict[str, Any]:
        # TODO: Should we add metadata from the transforms?
        return self._episode.meta
