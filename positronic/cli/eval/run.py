import logging
import os
import uuid
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import configuronic as cfn
import pos3

import pimm
import positronic.cfg.policy as policy_cfg
from positronic import telemetry, telemetry_keys, utils, wire
from positronic.cfg.eval import placeholder
from positronic.dataset.ds_writer_agent import TimeMode
from positronic.dataset.local_dataset import LocalDatasetWriter, load_all_datasets
from positronic.eval import Embodiment, Eval, Task
from positronic.gui import dpg_ui
from positronic.policy.base import SampledPolicy
from positronic.policy.harness import Harness
from positronic.simulator.env_server.telemetry import ATTR_RUN_ID, ENV_RUN_ID, ENV_TELEMETRY_DIR

logger = logging.getLogger(__name__)

# The environment a timed run hands a launched env server (read by ``env_server.telemetry.bind_from_env``);
# snapshotted before a run and restored after it, so a later run in the same process inherits nothing.
_ENV_TELEMETRY_VARS = (ENV_TELEMETRY_DIR, ENV_RUN_ID)


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
            # Feed live harness status to an operator surface that wants it (e.g. WebEvalUI's badge).
            # The status sink is optional: a GUI without one has no such attribute.
            gui_status = getattr(driver.gui, 'status', None) if driver.gui is not None else None
            if gui_status is not None:
                world.connect(harness.status, gui_status)
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


def _validate_timing(embodiments: Iterable[Embodiment], output_dir: str | Path | None) -> None:
    """Reject a ``--timing`` sweep the report cannot describe: one without an output dir to write the
    sidecars under, or one carrying a real embodiment. Everything under the bound tracer enters the report,
    so a real eval's episodes, wall time and machine samples would silently corrupt it."""
    if output_dir is None:
        raise ValueError('--timing needs --output_dir: the telemetry sidecars are written under it')
    real = sum(not embodiment.simulated for embodiment in embodiments)
    if real:
        raise ValueError(
            f'eval timing needs an all-simulated sweep, got {real} real '
            'embodiment(s): a real eval would run inside the same bound tracer and pass span, so its '
            'episodes, wall time and machine samples would silently corrupt the report. Run real '
            'embodiments in a separate untimed invocation.'
        )


@contextmanager
def _pass_span(**attrs) -> Generator[None, None, None]:
    """The span bracketing a whole eval sweep; the harness's episode spans parent to it, so it is anchored
    rather than entered as the OTel-current span — it must not become the parent of the rollout's phase spans. A
    sweep that exits with an exception still exports its pass — the partial window is real recorded data —
    stamped ``pass.failed`` so a reduce can see (and report) that it did not run to completion."""
    span = telemetry.start_span(telemetry_keys.SPAN_EVAL_PASS, **attrs)
    telemetry.push_anchor(span)
    try:
        yield
    except BaseException:
        telemetry.set_attrs(span, **{telemetry_keys.ATTR_PASS_FAILED: True})
        raise
    finally:
        span.end()
        telemetry.pop_anchor(span)


@contextmanager
def _timed_pass(output_dir: str | Path | None, timing: bool, policy):
    """Bracket a sweep in the harness-process telemetry: the bound tracer, the machine-load sampler and one
    ``eval.pass`` span, with the environment a launched env server reads set around them. Inert without
    ``timing``."""
    if not timing or output_dir is None:
        yield
        return
    timed_dir = Path(output_dir)
    run_id = uuid.uuid4().hex
    env_snapshot = {name: os.environ.get(name) for name in _ENV_TELEMETRY_VARS}
    # Set before any world comes up: a launched env server reads them off the environment its launcher
    # forwards to the subprocess, and writes its own sidecar under the same directory.
    os.environ[ENV_TELEMETRY_DIR] = str(timed_dir / telemetry.TELEMETRY_SUBDIR)
    os.environ[ENV_RUN_ID] = run_id
    try:
        # Built outside the pass: the constructor initialises NVML, enumerates its handles and primes the CPU
        # counters, and that setup is not eval wall — charging it to W_pass depresses the real-time factor.
        sampler = telemetry.StatsSampler(telemetry.stats_path(timed_dir, telemetry_keys.HARNESS_PROCESS))
        # Order is the contract twice over: the pass span closes before the tracer it is bound to shuts its
        # provider down, and the sampler's SAMPLING runs strictly INSIDE the pass span, so every sample it
        # stamps falls in the window the reduce counts. Sampling around the span instead drops its first and
        # last samples — and a run shorter than the sampling interval loses its only one, reporting a GPU box
        # as CPU-only.
        with (
            telemetry.bind(timed_dir, telemetry_keys.HARNESS_PROCESS, run_id),
            _pass_span(**{ATTR_RUN_ID: run_id, 'policy': type(policy).__name__}),
            sampler,
        ):
            yield
    finally:
        for name, value in env_snapshot.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


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
    stream). It needs an ``output_dir`` and an all-simulated sweep: everything under the bound tracer enters
    the report, so a real embodiment in a timed sweep is rejected rather than allowed to pollute it.
    """
    assert (driver is None) != (evals is None), 'Provide exactly one of driver or evals'
    # Validate timing up front, before the policy warmup, so a rejected sweep fails before it spends anything.
    if timing:
        if evals is not None:
            embodiments = [ev.embodiment for ev in evals]
        else:
            assert embodiment is not None, 'the attended (driver) path runs a single embodiment'
            embodiments = [embodiment]
        _validate_timing(embodiments, output_dir)

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

    try:
        with _timed_pass(output_dir, timing, policy):
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
