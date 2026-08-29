import logging
import os
import uuid
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import configuronic as cfn
import pos3
from platform_client.responses import SubmissionCreateResponse

import pimm
import positronic.cfg.policy as policy_cfg
from positronic import telemetry, telemetry_keys, utils, wire
from positronic.cfg.eval import placeholder
from positronic.cli.eval.submit import submit
from positronic.dataset.ds_writer_agent import TimeMode
from positronic.eval import Embodiment, Eval, Observation, Task
from positronic.policy import Policy
from positronic.policy.executor import blocking
from positronic.policy.harness import Harness, Rollout
from positronic.simulator.env_server.telemetry import ATTR_RUN_ID, ENV_RUN_ID, ENV_TELEMETRY_DIR

logger = logging.getLogger(__name__)

# The environment a timed run hands a launched env server (read by ``env_server.telemetry.bind_from_env``);
# snapshotted before a run and restored after it, so a later run in the same process inherits nothing.
_ENV_TELEMETRY_VARS = (ENV_TELEMETRY_DIR, ENV_RUN_ID)


def prepare_output_dir(output_dir: str | Path | None) -> Path | None:
    """Resolve where a run records: sync the directory and snapshot the sources into it.

    Returns the local path each episode records into, or ``None`` when the run records nothing.

    TODO(Positronic-Robotics/internal#378): take ``Path | None``. configuronic parses a CLI value with
    ``ast.literal_eval`` and falls back to ``str``, so ``--output_dir=s3://…`` arrives here as a ``str``
    whatever this says; narrowing before it coerces would make the annotation lie about what arrives.
    """
    if output_dir is None:
        return None
    local_dir = pos3.sync(str(output_dir), sync_on_error=True)
    utils.save_run_metadata(local_dir, patterns=['*.py', '*.toml'])
    return local_dir


class TaskDriver(pimm.ControlSystem):
    """Walks a plan of tasks, asking for each as an episode through ``perform_task``, and returns —
    stopping the world — once the last has ended.

    It makes the plan on its first turn, not when it is built. It opens a session per task, and asks for the
    episode that runs it, recording into ``dataset`` — the whole plan lands in one. One task is in flight at
    a time: the next is asked for only when the previous episode's terminal comes back, so the plan never
    overlaps two episodes, and each session opens on a model the last episode has let go of.
    """

    def __init__(self, tasks: Callable[[], Iterable[Task]], policy: Policy, dataset: Path | None):
        self._tasks = tasks
        self._policy = policy
        self._dataset = dataset
        self.perform_task = pimm.calls.ControlSystemCaller[Rollout, dict[str, Any]](self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        for task in self._tasks():
            answer = self.perform_task(Rollout(task, self._policy, self._dataset))
            while not answer.done():
                if should_stop.value:
                    return
                yield pimm.Yield()  # A sleep here would step the virtual clock on the driver's account.
            answer.result()  # raises if the episode failed
        # Let the recorder commit the final episode before this return brings the world down.
        yield pimm.Sleep(0.5)


def run_world(
    embodiment: Embodiment,
    driver,
    *,
    record: bool = True,
    privileged: dict[str, Observation] | None = None,
    done: pimm.ControlSystemEmitter | None = None,
) -> None:
    """Wire one embodiment under a fresh Harness + World, and run it until a control system returns.

    Every trial runs here, whoever asks for it: the driver is what an attended run and an unattended one
    differ by. A driver is any control system with a ``perform_task`` caller — a plan walked to its end, a
    person at a keyboard, a console of somebody's own — and it reads what it decides from itself, so the
    runner wires nothing of it but that call. The driver brings the policy and the dataset: it opens the
    session each episode runs on, and names where each episode records. ``record`` off keeps the recorder
    out of the world, so a run that writes nothing costs the producers nothing. ``done`` is what ends an
    episode from outside the policy: the env's terminal in a sim eval, the operator in an attended run.
    """
    harness = Harness(embodiment)
    time_mode = TimeMode.MESSAGE if embodiment.simulated else TimeMode.CLOCK
    with pimm.World(virtual_time=embodiment.simulated) as world:
        ds_agent = wire.wire_embodiment(
            world, harness, embodiment, time_mode, record=record, privileged=privileged, done=done
        )
        world.connect(driver.perform_task, harness.perform_task)
        if ds_agent is not None:
            world.connect(harness.ds_command, ds_agent.command)

        producers = [cs for cs in embodiment.control_systems if cs is not None]
        if embodiment.simulated:
            # Why this order:
            # - Each scheduler pass is one instant: everything emitted in it shares a timestamp.
            # - The harness tells the recorder when an episode opens, so the recorder runs after it and opens
            #   in that same pass.
            # - The producers run last, so what the recorder finds on the channels is the frame the reset
            #   published, with no step in between.
            world.run([driver, harness, ds_agent, *producers])
        else:
            world.run([driver, harness], [*producers, ds_agent])


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
def timed_pass(output_dir: str | Path | None, timing: bool, policy):
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


def main(policy, *, evals: list[Eval], output_dir: str | Path | None = None, timing: bool = False):
    """Run an unattended sweep: a driver walks each eval's tasks, rebuilding the World per eval.

    ``main`` owns the policy lifetime: it warms the policy once up front and closes it once after the last
    World, so a multi-eval sweep reuses one live policy across the rebuilds.

    ``timing`` records wall-clock telemetry sidecars under ``output_dir`` (spans + a machine-load stats
    stream). It needs an ``output_dir`` and an all-simulated sweep: everything under the bound tracer enters
    the report, so a real embodiment in a timed sweep is rejected rather than allowed to pollute it.
    """
    # Validate timing up front, before the policy warmup, so a rejected sweep fails before it spends anything.
    if timing:
        _validate_timing([ev.embodiment for ev in evals], output_dir)

    # A handshake returns only once the model is loaded, so this pays the cold start here, not in episode 1.
    # TODO: a policy with recording taps (recording_dir set) records this throwaway warmup session — an
    # empty .rrd plus a bump to the recorder's episode counter — but warmup is not a real episode.
    logger.info('Warming up policy endpoints')
    # The session runs no inference, but a session that serves its model on a runtime needs one to open.
    blocking(policy).new_session().close()
    dataset = prepare_output_dir(output_dir)

    try:
        with timed_pass(dataset, timing, policy):
            for ev in evals:
                driver = TaskDriver(ev.tasks, policy, dataset)
                run_world(ev.embodiment, driver, record=dataset is not None, privileged=ev.privileged, done=ev.done)
    finally:
        policy.close()


def _refuse(inapplicable: dict[str, object], where: str) -> None:
    """Stop on an argument the chosen half of `run` cannot honour.

    Dropping one silently hands back a run the caller believes they shaped. Every "not asked for"
    value is falsy, which is what makes the test the value itself.
    """
    asked = sorted(flag for flag, value in inapplicable.items() if value)
    if asked:
        raise SystemExit(f'a {where} run has no {", ".join(asked)}')


def _charged(tasks: Callable[[], Iterable[Task]], charge: bool) -> Iterator[Task]:
    """Every task the source makes, stamped with the run's inference-time policy."""
    return (replace(task, charge_inference_time=charge) for task in tasks())


@cfn.config(eval=placeholder, policy=policy_cfg.unset)
def run(
    eval: Eval | str,
    policy,
    output_dir=None,
    charge_inference_time: bool = False,
    timing=False,
    policy_image: str | None = None,
    alias: str | None = None,
    transaction_key: str | None = None,
    platform_url: str | None = None,
) -> SubmissionCreateResponse | None:
    """Run a selected eval (an embodiment and the tasks to run on it) through the shared inference harness.

    Here by default: ``--eval`` is an eval config and ``--policy`` the policy that drives it.
    ``--policy-image`` instead sends the run to the platform, which pulls that image and runs the
    eval of that NAME on the embodiment the eval names — ``--eval=robolab.public_subset``, not a
    config, since the platform owns the evals it offers. What comes back is a submission id, which
    ``positronic eval status`` reads.

    ``timing`` records wall-clock telemetry sidecars under ``output_dir`` (spans + machine-load stats) for a
    simulated eval; reduce them with ``positronic eval timing-report``.

    A platform run returns what the platform created, so a caller holding this function has the
    submission id without parsing what was printed. A local run's result is the dataset it wrote.
    """
    if policy_image is None:
        if policy is None:
            raise SystemExit('--policy is required to run here; --policy-image runs it on the platform instead')
        _refuse({'--alias': alias, '--transaction-key': transaction_key, '--platform-url': platform_url}, 'local')
        if not isinstance(eval, Eval):
            raise SystemExit(f'--eval={eval!r} is a name, not a config: pass --policy-image to run it on the platform')
        eval = replace(eval, tasks=partial(_charged, eval.tasks, charge_inference_time))
        main(policy=policy, evals=[eval], output_dir=output_dir, timing=timing)
        return None

    if policy is not None:
        raise SystemExit('--policy runs the eval here and --policy-image runs it on the platform; pass one')
    if not isinstance(eval, str):
        raise SystemExit('the platform names its own evals: pass --eval=<name>, e.g. --eval=robolab.public_subset')
    # The platform owns its own trial sweep, its own output and its own telemetry.
    _refuse(
        {'--output-dir': output_dir, '--charge-inference-time': charge_inference_time, '--timing': timing}, 'platform'
    )
    return submit(eval, policy_image, alias=alias, transaction_key=transaction_key, platform_url=platform_url)
