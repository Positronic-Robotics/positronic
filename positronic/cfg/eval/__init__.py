import random
from dataclasses import replace

import configuronic as cfn

from positronic.eval import Task


@cfn.config()
def placeholder():
    # Lets ``--eval=.sim.positronic.stack_cubes`` resolve relative to this package; never instantiated.
    raise SystemExit(
        '--eval is required: a config to run here (--eval=.sim.positronic.stack_cubes), '
        'or the name of one the platform offers (--eval=robolab.public_subset, with --policy-image)'
    )


def number_trials(task: Task, params: list[dict]) -> list[Task]:
    """One copy of ``task`` per entry in ``params``, each also carrying its place in the sweep."""
    return [
        replace(task, params={**p, 'eval.trial_index': i, 'eval.trial_count': len(params)})
        for i, p in enumerate(params)
    ]


def build_tasks(task: Task, seed: int | None, trial_count: int, scenes: list[dict] | None = None) -> list[Task]:
    """The sweep an eval runs: one copy of ``task`` per (scene, seed) pair.

    Each ``scenes`` entry is a scene-spec base (e.g. ``{'eval.suite': ..., 'eval.task_id': ...}``) swept over
    the seed set; ``None`` sweeps the seed alone (an eval with no scene axis). ``seed`` ``None`` draws an
    independent random seed per trial; an int runs ``seed .. seed + trial_count - 1`` for every scene.
    """
    return number_trials(
        task,
        [
            {**scene, 'eval.seed': seed + s if seed is not None else random.randrange(2**31)}
            for scene in scenes or [{}]
            for s in range(trial_count)
        ],
    )
