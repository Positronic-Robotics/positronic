import random
from dataclasses import replace
from typing import Any

import configuronic as cfn

from positronic.eval import EVAL_SEED, EVAL_TRIAL_COUNT, EVAL_TRIAL_INDEX, SCENE, Task


@cfn.config()
def placeholder():
    # Lets ``--eval=.sim.positronic.stack_cubes`` resolve relative to this package; never instantiated.
    raise SystemExit(
        '--eval is required: a config to run here (--eval=.sim.positronic.stack_cubes), '
        'or the name of one the platform offers (--eval=robolab.public_subset, with --policy-image)'
    )


def spec(**selection) -> dict[str, Any]:
    """The spec an env resolves: the selection arguments an eval binds, an unbound (``None``) one absent."""
    return {name: value for name, value in selection.items() if value is not None}


def number_trials(trials: list[tuple[Task, dict]]) -> list[Task]:
    """One trial per ``(task, params)`` pair: its scene prepare is asked with the params, and its episode records
    them beside the trial's place in the sweep."""
    return [
        replace(
            task,
            prepare_args={**task.prepare_args, SCENE: params},
            meta={**task.meta, **params, EVAL_TRIAL_INDEX: i, EVAL_TRIAL_COUNT: len(trials)},
        )
        for i, (task, params) in enumerate(trials)
    ]


def build_tasks(task: Task, seed: int | None, trial_count: int, scenes: list[dict] | None = None) -> list[Task]:
    """The sweep an eval runs: one copy of ``task`` per (scene, seed) pair.

    Each ``scenes`` entry is a scene-spec base (LIBERO's suite and task id, say) swept over the seed set;
    ``None`` sweeps the seed alone (an eval with no scene axis). ``seed`` ``None`` draws an independent
    random seed per trial; an int runs ``seed .. seed + trial_count - 1`` for every scene.
    """
    return number_trials([
        (task, {**scene, EVAL_SEED: seed + s if seed is not None else random.randrange(2**31)})
        for scene in (scenes if scenes is not None else [{}])
        for s in range(trial_count)
    ])
