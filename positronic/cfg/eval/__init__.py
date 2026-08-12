import random
from collections.abc import Callable
from typing import Any

import configuronic as cfn

from positronic import keys
from positronic.eval import Rollout


@cfn.config()
def placeholder():
    # Lets ``--eval=.sim.positronic.stack_cubes`` resolve relative to this package; never instantiated.
    raise SystemExit(
        '--eval is required: a config to run here (--eval=.sim.positronic.stack_cubes), '
        'or the name of one the platform offers (--eval=robolab.public_subset, with --policy-image)'
    )


def build_rollouts(
    instruction: str | Callable[[], str] | None,
    timeout: float,
    seed: int | None,
    rollout_count: int,
    scenes: list[dict[str, Any]] | None = None,
) -> list[Rollout]:
    """The rollout plan a self-driving eval sweeps: one task per (scene, seed) pair.

    Each ``scenes`` entry is a scene-spec base (e.g. ``{'eval.suite': ..., 'eval.task_id': ...}``) swept over
    the seed set; ``None`` sweeps the seed alone, for an eval with no scene axis. ``seed`` ``None`` draws an
    independent random seed per rollout; an int runs ``seed .. seed + rollout_count - 1`` for every scene.
    Every rollout carries the same ``instruction`` and ``timeout``.
    """

    def draw(i: int) -> int:
        return seed + i if seed is not None else random.randrange(2**31)

    return [
        Rollout(instruction, timeout, {**scene, keys.EVAL_SEED: draw(i)})
        for scene in (scenes if scenes is not None else [{}])
        for i in range(rollout_count)
    ]
