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
    # The trial plan: one RUN context per trial, consumed by the self-driving Harness.
    # (Step 5 adds the per-trial seed `base + i` here.)
    trials = [
        {
            'timeout': eval.task.timeout,
            'inference_latency': inference_latency,
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
