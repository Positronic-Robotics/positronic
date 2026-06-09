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
def run(
    eval: Eval,  # noqa: A002 — the CLI flag is `--eval`, so the parameter must be named `eval`
    policy,
    trial_count,
    show_gui,
    output_dir=None,
    simulate_inference=False,
    wrap=default_wrappers,
):
    """Run a selected eval (embodiment + task) through the shared inference harness.

    The per-trial time budget is the task's ``timeout`` (override with ``--eval.timeout``).
    """
    driver = inference.timed(num_iterations=trial_count, simulation_time=eval.task.timeout, show_gui=show_gui)
    inference.main(
        embodiment=eval.embodiment,
        task=eval.task,
        policy=policy,
        driver=driver,
        output_dir=output_dir,
        simulate_inference=simulate_inference,
        wrap=wrap,
    )
