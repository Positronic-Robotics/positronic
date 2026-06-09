import configuronic as cfn

import positronic.cfg.policy as policy_cfg
from positronic import inference
from positronic.policy.harness import default_wrappers
from positronic.task import Eval


def _eval_required():
    raise SystemExit('--eval is required, e.g. --eval=.sim.positronic.stack_cubes')


# Sentinel default defined in this package so its base path is ``positronic.eval``:
# ``--eval=.sim.positronic.stack_cubes`` resolves to ``positronic.eval.sim.positronic.stack_cubes``,
# lazily (only the selected eval is imported) — there is no central catalog to keep in sync.
EVAL_SENTINEL = cfn.Config(_eval_required)


def _run(
    eval: Eval,  # noqa: A002 — the CLI flag is `--eval`, so the parameter must be named `eval`
    policy,
    trial_count,
    simulation_time,
    show_gui,
    output_dir=None,
    simulate_inference=False,
    wrap=default_wrappers,
):
    """Run a selected eval (embodiment + task) through the shared inference harness."""
    driver = inference.timed(num_iterations=trial_count, simulation_time=simulation_time, show_gui=show_gui)
    inference.main(
        embodiment=eval.embodiment,
        task=eval.task,
        policy=policy,
        driver=driver,
        output_dir=output_dir,
        simulate_inference=simulate_inference,
        wrap=wrap,
    )


run_cfg = cfn.Config(
    _run,
    eval=EVAL_SENTINEL,
    policy=policy_cfg.placeholder,
    trial_count=1,
    simulation_time=15,
    show_gui=False,
    wrap=default_wrappers,
)
