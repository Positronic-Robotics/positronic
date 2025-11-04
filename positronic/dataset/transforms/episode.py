from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any

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


class KeyFuncEpisodeTransform(EpisodeTransform):
    """Transform an episode using a dictionary of key-function pairs."""

    def __init__(
        self,
        add: dict[str, Callable[[Episode], Any]],
        remove: bool | list[str] = False,
        pass_through: bool | list[str] = True,
    ):
        self._transform_fns = add
        self.pass_keys = set()
        if pass_through is True:
            assert remove is not True, 'If pass_through is True, remove must be False or a list of keys to remove.'
            self.remove = set(remove) if isinstance(remove, list) else set()
            self.pass_all = True
        elif pass_through is False:
            assert remove is False, "When we don't pass anything through, there's nothing to remove."
            self.remove = set()
            self.pass_all = False
        else:
            assert remove is False, "When we pass through specific keys, to remove keys just don't pass them through."
            self.pass_keys = set(pass_through)
            self.remove = set()
            self.pass_all = False

    def _pass_key(self, key: str) -> bool:
        return (key not in self.remove) and (self.pass_all or key in self.pass_keys)

    def __call__(self, episode: Episode) -> Episode:
        # TODO: Should we lazy-fy this?
        data = {k: v for k, v in episode.items() if self._pass_key(k)}
        data.update({name: fn(episode) for name, fn in self._transform_fns.items()})
        return EpisodeContainer(data, episode.meta)


class Concatenate(EpisodeTransform):
    """Concatenate multiple transforms into a single transform.

    Applies all transforms to the same input episode and merges their results.
    If transforms produce overlapping keys, the first transform in the list takes precedence.
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
        return {k: v for tf in reversed(self._transforms) for k, v in tf.meta.items()}


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
