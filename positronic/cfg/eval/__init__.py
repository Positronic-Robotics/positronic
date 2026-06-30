import random

import configuronic as cfn


@cfn.config()
def placeholder():
    # Lets ``--eval=.sim.positronic.stack_cubes`` resolve relative to this package; never instantiated.
    raise SystemExit('--eval is required, e.g. --eval=.sim.positronic.stack_cubes')


def build_trials(seed: int | None, trial_count: int, task_ids: list[int] | None = None) -> list[dict]:
    """The per-trial RUN contexts a self-driving eval sweeps: one per (task_id, seed) pair.

    ``task_ids`` ``None`` sweeps the seed alone (an eval with no task axis); a list sweeps each task_id over the
    same seed set. ``seed`` ``None`` draws an independent random seed per trial; an int runs
    ``seed .. seed + trial_count - 1`` for every task. Each context also carries its flat ``eval.trial_index``
    and the total ``eval.trial_count``.
    """
    axes = [None] if task_ids is None else task_ids
    trials = []
    for task_id in axes:
        for s in range(trial_count):
            ctx = {'eval.seed': seed + s if seed is not None else random.randrange(2**31)}
            if task_id is not None:
                ctx['eval.task_id'] = task_id
            trials.append(ctx)
    for i, ctx in enumerate(trials):
        ctx['eval.trial_index'] = i
        ctx['eval.trial_count'] = len(trials)
    return trials
