import logging
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Hashable, Sequence
from typing import Any

logger = logging.getLogger(__name__)


class EpisodeCounter:
    """Per-key tally of completed episodes, grouped by context fields.

    Owns the counting state that a balancing ``Sampler`` reads. The harness
    bumps it on each successful episode completion via :meth:`record`; a sampler
    queries the current tallies via :meth:`counts` when choosing the next
    sub-policy. :meth:`seed_from` rebuilds the tallies from already-recorded
    episodes so balancing survives a process restart.

    Keys are read from ``session.meta[key_field]`` (the identifier of the
    sub-policy that ran, e.g. a checkpoint path). Episodes are grouped by
    ``group_fields`` looked up in the episode context, so each group is balanced
    independently (e.g. one tally per task).
    """

    def __init__(self, key_field: str, group_fields: Sequence[str] | None = None):
        self._key_field = key_field
        self._group_fields = tuple(group_fields) if group_fields is not None else None
        self._counts: dict[Hashable, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def _group(self, context: dict[str, Any]) -> Hashable:
        if self._group_fields is None:
            return ()
        return tuple(context.get(f) for f in self._group_fields)

    def record(self, session: Any, context: dict[str, Any]) -> None:
        """Tally one completed episode, reading its key from ``session.meta``."""
        key = session.meta.get(self._key_field)
        if key is not None:
            self._counts[self._group(context)][key] += 1

    def counts(self, keys: Sequence[str], context: dict[str, Any]) -> dict[str, int]:
        """Current completed-episode count for each key in ``keys`` within ``context``'s group."""
        group = self._counts[self._group(context)]
        return {k: group[k] for k in keys}

    def seed_from(self, dataset: Any, meta_prefix: str = 'inference.policy.') -> int:
        """Rebuild tallies from recorded episodes in ``dataset``. Returns the number counted.

        The episode static stores the key under ``{meta_prefix}{key_field}`` and the
        grouping fields under their raw names (the context the harness recorded).
        """
        meta_key = f'{meta_prefix}{self._key_field}'
        fields = self._group_fields or ()
        seeded = 0
        for i in range(len(dataset)):
            static = dataset[i].static
            key = static.get(meta_key)
            if key is None:
                continue
            self._counts[tuple(static.get(f) for f in fields)][key] += 1
            seeded += 1
        return seeded


class Sampler(ABC):
    """Strategy for selecting which sub-policy to run in a SampledPolicy."""

    @abstractmethod
    def sample(self, keys: Sequence[str], context: dict[str, Any], counts: dict[str, int]) -> str:
        """Pick a policy key from ``keys`` given the episode ``context`` and per-key ``counts``."""


class UniformSampler(Sampler):
    """Samples policies uniformly at random, optionally with fixed weights per key."""

    def __init__(self, weights: dict[str, float] | None = None):
        self._weights = weights

    def sample(self, keys: Sequence[str], context: dict[str, Any], counts: dict[str, int]) -> str:
        w = [self._weights[k] for k in keys] if self._weights else None
        return random.choices(keys, w)[0]


class BalancedSampler(Sampler):
    """Samples policies to balance completed-episode counts across checkpoints.

    Sampling probability for policy i is proportional to:
        max(counts) + balance - count_i

    Counts are supplied per call (the harness maintains them in an
    ``EpisodeCounter``), so this strategy is stateless.
    """

    def __init__(self, *, balance: int = 5):
        self._balance = balance

    def sample(self, keys: Sequence[str], context: dict[str, Any], counts: dict[str, int]) -> str:
        c = [counts[k] for k in keys]
        max_count = max(c) if c else 0
        weights = [max_count + self._balance - x for x in c]
        chosen = random.choices(list(keys), weights)[0]
        lines = ['BalancedSampler']
        for k, x, w in zip(keys, c, weights, strict=True):
            marker = ' ←' if k == chosen else ''
            lines.append(f'  {k}: count={x} weight={w}{marker}')
        logger.info('\n'.join(lines))
        return chosen
