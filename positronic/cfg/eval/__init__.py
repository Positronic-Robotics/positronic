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
    driver = inference.sequencer(
        trial_count=trial_count, timeout=eval.task.timeout, inference_latency=inference_latency, show_gui=show_gui
    )
    inference.main(
        embodiment=eval.embodiment, task=eval.task, policy=policy, driver=driver, output_dir=output_dir, wrap=wrap
    )
