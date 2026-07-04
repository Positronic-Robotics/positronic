import random

import configuronic as cfn


@cfn.config()
def placeholder():
    # Lets ``--eval=.sim.positronic.stack_cubes`` resolve relative to this package; never instantiated.
    raise SystemExit('--eval is required, e.g. --eval=.sim.positronic.stack_cubes')


def build_trials(seed: int | None, trial_count: int, scenes: list[dict] | None = None) -> list[dict]:
    """The per-trial RUN contexts a self-driving eval sweeps: one per (scene, seed) pair.

    Each ``scenes`` entry is a scene-spec context base (e.g. ``{'eval.suite': ..., 'eval.task_id': ...}``)
    swept over the seed set; ``None`` sweeps the seed alone (an eval with no scene axis). ``seed`` ``None``
    draws an independent random seed per trial; an int runs ``seed .. seed + trial_count - 1`` for every
    scene. Each context also carries its flat ``eval.trial_index`` and the total ``eval.trial_count``.
    """
    trials = []
    for scene in scenes if scenes is not None else [{}]:
        for s in range(trial_count):
            trials.append({**scene, 'eval.seed': seed + s if seed is not None else random.randrange(2**31)})
    for i, ctx in enumerate(trials):
        ctx['eval.trial_index'] = i
        ctx['eval.trial_count'] = len(trials)
    return trials
