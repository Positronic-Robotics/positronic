import logging
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import configuronic as cfn
import pos3

import pimm
import positronic.cfg.policy as policy_cfg
import positronic.cfg.wrappers as wrappers_cfg
from positronic import eval_timing, utils, wire
from positronic.cfg.eval import placeholder
from positronic.dataset.ds_writer_agent import TimeMode
from positronic.dataset.local_dataset import LocalDatasetWriter, load_all_datasets
from positronic.eval import Embodiment, Eval, Task
from positronic.gui.dpg import DearpyguiUi
from positronic.policy.base import SampledPolicy
from positronic.policy.harness import Harness

logger = logging.getLogger(__name__)


@dataclass
class Driver:
    """An attended operator surface: the directive source ``main`` wires into the Harness.

    Driver configs produce a factory called with the resolved local output directory, since
    the directory exists only after ``pos3.sync`` inside ``main``.
    """

    gui: pimm.ControlSystem | None
    directives: pimm.SignalEmitter
    directive_wrapper: Callable
    control_systems: list[pimm.ControlSystem]
    manual_commands: pimm.SignalEmitter | None = None


def _seed_counter(policy, output_dir: Path):
    """If policy is a SampledPolicy, seed its episode counter from existing episodes in output_dir."""
    if not isinstance(policy, SampledPolicy):
        return
    try:
        dataset = load_all_datasets(output_dir)
    except ValueError:
        return
    if len(dataset) == 0:
        return
    seeded = policy.counter.seed_from(dataset)
    logger.info(f'Seeded counter from {seeded} existing episodes')


def _completion_sink(policy):
    """Harness ``on_episode_complete`` callback that tallies completed episodes.

    Returns the ``SampledPolicy``'s counter ``record`` (which reads the sampled
    key from the session and bumps its tally), or ``None`` for non-sampled
    policies. The harness fires it on each clean episode completion.
    """
    return policy.counter.record if isinstance(policy, SampledPolicy) else None


def _run_world(
    policy,
    embodiment: Embodiment,
    task: Task | None,
    trials: list[dict] | None,
    driver: Driver | None,
    output_dir: Path | None,
    show_gui: bool,
    on_complete,
    *,
    wrap,
):
    """Wire one embodiment under a fresh Harness + World and run it to completion.

    ``driver`` (attended) and ``trials`` (unattended self-driving) are the two lifecycle sources, mutually
    exclusive per the caller. The shared ``policy`` is wrapped here per run, but its lifetime stays with ``main``.
    """
    harness = Harness(policy, embodiment, task=task, trials=trials, wrap=wrap, on_episode_complete=on_complete)
    gui = driver.gui if driver is not None else (DearpyguiUi() if show_gui else None)

    time_mode = TimeMode.MESSAGE if embodiment.simulated else TimeMode.CLOCK
    writer_cm = LocalDatasetWriter(output_dir) if output_dir is not None else nullcontext(None)
    with writer_cm as dataset_writer, pimm.World(virtual_time=embodiment.simulated) as world:
        privileged = task.privileged if task is not None else {}
        done = task.done if task is not None else None
        ds_agent = wire.wire_embodiment(
            world, harness, embodiment, dataset_writer, time_mode, privileged=privileged, done=done
        )
        if gui is not None:
            # HACK: GUI cameras are matched to observations by the `image.` name prefix, which
            # hard-binds GUI wiring to the observation naming convention. TODO: resolve this
            # coupling (the right binding is still open).
            for name, obs in embodiment.observations.items():
                if name.startswith('image.'):
                    world.connect(obs.source, gui.cameras[name])
        if driver is not None:
            world.connect(driver.directives, harness.directive, emitter_wrapper=driver.directive_wrapper)
            if driver.manual_commands is not None:
                world.connect(driver.manual_commands, harness.manual_command)
        if ds_agent is not None:
            world.connect(harness.ds_command, ds_agent.command)

        # Sim schedules harness, recorder, then producers (the simulator) in-process under the virtual clock,
        # in that order; each scheduler round is one control period. The harness decides the round's action
        # (a reset, a policy command off the last round's observation, or finish); the recorder logs that
        # observation with the command; the producer applies the command and publishes the next observation.
        # A reset arms the producer to publish frame-0 after the harness (last in the round); the recorder
        # drains its channels the turn it opens, dropping the pre-reset frame, so its first recorded sample
        # is the post-reset scene. Real runs the producers + recorder as background subprocesses; harness,
        # driver, and GUI placement is otherwise identical.
        producers = [cs for cs in embodiment.control_systems if cs is not None]
        foreground = driver.control_systems if driver is not None else []
        if embodiment.simulated:
            world.run([*foreground, harness, ds_agent, *producers], gui)
        else:
            world.run([harness, *foreground], [*producers, ds_agent, gui])


def main(
    policy,
    *,
    wrap,
    evals: list[Eval] | None = None,
    embodiment: Embodiment | None = None,
    driver: Callable[[Path | None], Driver] | None = None,
    output_dir: str | Path | None = None,
    show_gui: bool = False,
    timing: bool = False,
):
    """Run inference for an embodiment, real or simulated.

    Exactly one of ``driver`` (attended: a factory producing the operator surface that emits the directives
    over a single ``embodiment``) or ``evals`` (unattended: the harness self-drives each eval's trial plan,
    rebuilding the World per eval) must be given; ``show_gui`` applies to the unattended path (attended surfaces
    bring their own). ``main`` owns the policy lifetime: it warms the policy once up front and closes it once
    after the last World, so a multi-eval sweep reuses one live policy across the rebuilds.
    """
    assert (driver is None) != (evals is None), 'Provide exactly one of driver or evals'
    # Timing is validated up front, before the policy warmup and the bind: a rejected sweep must fail
    # before it spends anything.
    if timing:
        if output_dir is None:
            raise ValueError('--timing needs --output_dir: the per-rollout telemetry is recorded into the dataset')
        embodiments = [ev.embodiment for ev in evals] if evals is not None else [embodiment]
        if not all(e.simulated for e in embodiments):
            raise ValueError(
                'eval timing is sim-only: a real embodiment runs the recorder and producers as separate '
                'processes that do not inherit the timer context, so the episodes would carry no timing signal '
                '(record_io_s silently reads 0). Drop --timing for a real run.'
            )

    # Drive the policy's remote endpoints through their cold start before hardware and the operator
    # surface come up: opening a session blocks on the server handshake, which returns only once the
    # model is loaded, and a SampledPolicy reaches every sub-policy. The first episode then begins
    # warm instead of stalling on an on-request endpoint's model load while the robot waits.
    # TODO: a policy with recording taps (recording_dir set) records this throwaway warmup session —
    # an empty .rrd plus a bump to the recorder's episode counter — but warmup is not a real episode.
    logger.info('Warming up policy endpoints')
    policy.new_session().close()

    if output_dir is not None:
        output_dir = pos3.sync(output_dir, sync_on_error=True)
        utils.save_run_metadata(output_dir, patterns=['*.py', '*.toml'])
        _seed_counter(policy, output_dir)

    # One completion sink — so one ``SampledPolicy`` counter — across every eval, keeping sampling balanced
    # over the whole sweep.
    on_complete = _completion_sink(policy)
    # Bind one collector (and GPU sampler) around the whole sweep, not per eval: a per-eval bind would
    # restart the ``gpu_dmon.log`` sampler each time, dropping earlier evals' GPU samples.
    timing_cm = eval_timing.bind(output_dir) if timing and output_dir is not None else nullcontext()
    try:
        with timing_cm:
            if driver is not None:
                _run_world(
                    policy, embodiment, None, None, driver(output_dir), output_dir, show_gui, on_complete, wrap=wrap
                )
            else:
                for ev in evals:
                    _run_world(
                        policy, ev.embodiment, ev.task, ev.trials, None, output_dir, show_gui, on_complete, wrap=wrap
                    )
    finally:
        policy.close()


@cfn.config(eval=placeholder, policy=policy_cfg.placeholder, show_gui=False, wrap=wrappers_cfg.default_wrappers)
def run(eval: Eval, policy, show_gui, output_dir=None, inference_latency=False, timing=False, *, wrap):
    """Run a selected eval (embodiment + task + its trial sweep) through the shared inference harness.

    ``timing`` records per-rollout wall-clock telemetry into the dataset under ``output_dir`` (a ``timing.*``
    signal + statics per episode); it needs an ``output_dir`` and applies to sim evals only. Reduce it with
    ``positronic eval timing-report``.
    """
    # The eval config owns the trial sweep (seed, task range); ``inference_latency`` is the CLI's per-run knob
    # (sim inference-cost simulation). Overlay it onto every trial context, then self-drive the eval.
    eval = replace(eval, trials=[{**trial, 'inference_latency': inference_latency} for trial in eval.trials])
    main(policy=policy, evals=[eval], show_gui=show_gui, output_dir=output_dir, wrap=wrap, timing=timing)
