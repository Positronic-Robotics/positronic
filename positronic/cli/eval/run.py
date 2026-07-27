import logging
import os
import uuid
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import configuronic as cfn
import pos3

import pimm
import positronic.cfg.policy as policy_cfg
from positronic import telemetry, utils, wire
from positronic.cfg.eval import placeholder
from positronic.dataset.ds_writer_agent import TimeMode
from positronic.dataset.local_dataset import LocalDatasetWriter, load_all_datasets
from positronic.eval import Embodiment, Eval, Task
from positronic.gui import dpg_ui
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
):
    """Wire one embodiment under a fresh Harness + World and run it to completion.

    ``driver`` (attended) and ``trials`` (unattended self-driving) are the two lifecycle sources, mutually
    exclusive per the caller. The shared ``policy``'s lifetime stays with ``main``.
    """
    harness = Harness(policy, embodiment, task=task, trials=trials, on_episode_complete=on_complete)
    gui = driver.gui if driver is not None else (dpg_ui() if show_gui else None)

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

    ``timing`` records wall-clock telemetry sidecars under ``output_dir`` (spans + a machine-load stats
    stream). It applies to simulated embodiments — a real run schedules the recorder and producers as
    separate processes that never see the harness-process tracer — and needs an ``output_dir``.
    """
    assert (driver is None) != (evals is None), 'Provide exactly one of driver or evals'
    # Validate timing up front, before the policy warmup, so a rejected sweep fails before it spends anything.
    if timing:
        if output_dir is None:
            raise ValueError('--timing needs --output_dir: the telemetry sidecars are written under it')
        if evals is not None:
            simulated = [ev.embodiment.simulated for ev in evals]
        else:
            assert embodiment is not None, 'the attended (driver) path runs a single embodiment'
            simulated = [embodiment.simulated]
        if not any(simulated):
            raise ValueError(
                'eval timing needs a simulated embodiment: a real embodiment runs the recorder and producers '
                'as separate processes with no shared tracer, so nothing here is timed. Drop --timing.'
            )
        if not all(simulated):
            logger.warning(
                'eval timing is sim-only: %d real embodiment(s) in this sweep run untimed.',
                sum(not s for s in simulated),
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

    # Bind the harness-process telemetry (and the machine-load sampler) around the whole sweep, and bracket
    # the trial loop in one ``eval.pass`` span; a mixed sweep's sim evals share them. All inert off --timing.
    run_id = uuid.uuid4().hex
    timed = timing and output_dir is not None
    if timing and output_dir is not None:
        # A launched env server (e.g. RoboLab, positronic-free in its own interpreter) writes its own
        # telemetry sidecar; it reads these from the environment its launcher forwards to the subprocess.
        os.environ['POSITRONIC_ENV_TELEMETRY_DIR'] = str(Path(output_dir) / 'telemetry')
        os.environ['POSITRONIC_RUN_ID'] = run_id
    telemetry_cm = telemetry.bind(output_dir, 'harness', run_id) if timing and output_dir is not None else nullcontext()
    stats_cm = (
        telemetry.StatsSampler(Path(output_dir) / 'telemetry' / 'harness.stats.jsonl')
        if timing and output_dir is not None
        else nullcontext()
    )
    pass_cm = telemetry.eval_pass(**{'run.id': run_id, 'policy': type(policy).__name__}) if timed else nullcontext()
    try:
        with telemetry_cm, stats_cm, pass_cm:
            if driver is not None:
                assert embodiment is not None, 'the attended (driver) path runs a single embodiment'
                _run_world(policy, embodiment, None, None, driver(output_dir), output_dir, show_gui, on_complete)
            else:
                assert evals is not None  # driver/evals XOR asserted up front
                for ev in evals:
                    _run_world(policy, ev.embodiment, ev.task, ev.trials, None, output_dir, show_gui, on_complete)
    finally:
        policy.close()


@cfn.config(eval=placeholder, policy=policy_cfg.placeholder, show_gui=False)
def run(eval: Eval, policy, show_gui, output_dir=None, inference_latency=False, timing=False):
    """Run a selected eval (embodiment + task + its trial sweep) through the shared inference harness.

    ``timing`` records wall-clock telemetry sidecars under ``output_dir`` (spans + machine-load stats) for a
    simulated eval; reduce them with ``positronic eval timing-report``.
    """
    # The eval config owns the trial sweep (seed, task range); ``inference_latency`` is the CLI's per-run knob
    # (sim inference-cost simulation). Overlay it onto every trial context, then self-drive the eval.
    eval = replace(eval, trials=[{**trial, 'inference_latency': inference_latency} for trial in eval.trials])
    main(policy=policy, evals=[eval], show_gui=show_gui, output_dir=output_dir, timing=timing)
