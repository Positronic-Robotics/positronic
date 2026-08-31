import random
from dataclasses import replace

import configuronic as cfn

from positronic import keys
from positronic.eval import Task


@cfn.config()
def placeholder():
    # Lets ``--eval=.sim.positronic.stack_cubes`` resolve relative to this package; never instantiated.
    raise SystemExit(
        '--eval is required: a config to run here (--eval=.sim.positronic.stack_cubes), '
        'or the name of one the platform offers (--eval=robolab.public_subset, with --policy-image)'
    )


def number_trials(trials: list[tuple[Task, dict]]) -> list[Task]:
    """One trial per ``(task, params)`` pair: each task's scene prepare is asked with its own params, and its
    episode records them beside the trial's place in the whole sweep."""
    return [
        replace(
            task,
            prepare_args={**task.prepare_args, keys.SCENE: params},
            meta={**task.meta, **params, keys.EVAL_TRIAL_INDEX: i, keys.EVAL_TRIAL_COUNT: len(trials)},
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
        (task, {**scene, keys.EVAL_SEED: seed + s if seed is not None else random.randrange(2**31)})
        for scene in (scenes if scenes is not None else [{}])
        for s in range(trial_count)
    ])
