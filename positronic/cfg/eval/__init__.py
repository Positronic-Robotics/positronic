import random

import configuronic as cfn

import positronic.cfg.policy as policy_cfg
from positronic import inference
from positronic.eval import Eval
from positronic.policy.harness import default_wrappers


@cfn.config()
def placeholder():
    # Lets ``--eval=.sim.positronic.stack_cubes`` resolve relative to this package; never instantiated.
    raise SystemExit('--eval is required, e.g. --eval=.sim.positronic.stack_cubes')


@cfn.config(eval=placeholder, policy=policy_cfg.placeholder, trial_count=1, show_gui=False, wrap=default_wrappers)
def run(eval: Eval, policy, trial_count, show_gui, output_dir=None, inference_latency=False, wrap=default_wrappers):
    """Run a selected eval (embodiment + task) through the shared inference harness."""
    # The trial plan: one RUN context per trial, consumed by the self-driving Harness. Per-trial seeds
    # are known upfront — ``--eval.seed`` + trial index, or an independent random draw per trial when
    # unset — and ride the RUN context, so the seed used always lands in episode meta.
    base = eval.task.seed
    trials = [
        {
            'inference_latency': inference_latency,
            'eval.seed': base + i if base is not None else random.randrange(2**31),
            'eval.trial_index': i,
            'eval.trial_count': trial_count,
        }
        for i in range(trial_count)
    ]
    inference.main(
        embodiment=eval.embodiment,
        task=eval.task,
        policy=policy,
        trials=trials,
        show_gui=show_gui,
        output_dir=output_dir,
        wrap=wrap,
    )


@cfn.config(eval=placeholder, policy=policy_cfg.placeholder, driver=inference.eval_ui, wrap=default_wrappers)
def attended(eval: Eval, policy, driver, output_dir=None, wrap=default_wrappers):
    """Drive a selected eval interactively: the operator surface owns the lifecycle instead of a trial plan."""
    inference.main(
        embodiment=eval.embodiment, task=eval.task, policy=policy, driver=driver, output_dir=output_dir, wrap=wrap
    )
