import logging
import os
import uuid
from collections.abc import Generator, Iterable
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path

import configuronic as cfn
import pos3
from platform_client.responses import SubmissionCreateResponse

import pimm
import positronic.cfg.policy as policy_cfg
from positronic import keys, telemetry, telemetry_keys, utils, wire
from positronic.cfg.eval import placeholder
from positronic.cli.eval.submit import submit
from positronic.dataset.ds_writer_agent import TimeMode
from positronic.dataset.local_dataset import LocalDatasetWriter
from positronic.eval import Embodiment, Eval
from positronic.policy.harness import Harness
from positronic.simulator.env_server.telemetry import ATTR_RUN_ID, ENV_RUN_ID, ENV_TELEMETRY_DIR

logger = logging.getLogger(__name__)

# The environment a timed run hands a launched env server (read by ``env_server.telemetry.bind_from_env``);
# snapshotted before a run and restored after it, so a later run in the same process inherits nothing.
_ENV_TELEMETRY_VARS = (ENV_TELEMETRY_DIR, ENV_RUN_ID)


def prepare_output_dir(output_dir: str | Path | None) -> Path | None:
    """Resolve where a run records: sync the directory and snapshot the sources into it.

    Returns the local path a ``LocalDatasetWriter`` opens, or ``None`` when the run records nothing.

    TODO(Positronic-Robotics/internal#378): take ``Path | None``. configuronic parses a CLI value with
    ``ast.literal_eval`` and falls back to ``str``, so ``--output_dir=s3://…`` arrives here as a ``str``
    whatever this says; narrowing before it coerces would make the annotation lie about what arrives.
    """
    if output_dir is None:
        return None
    local_dir = pos3.sync(str(output_dir), sync_on_error=True)
    utils.save_run_metadata(local_dir, patterns=['*.py', '*.toml'])
    return local_dir


def _run_world(policy, ev: Eval, output_dir: Path | None):
    """Wire one eval's embodiment under a fresh Harness + World and run it to completion.

    The harness self-drives ``ev.trials``; the shared ``policy``'s lifetime stays with ``main``.
    """
    embodiment = ev.embodiment
    harness = Harness(policy, embodiment, task=ev.task, trials=ev.trials, reset=ev.reset)

    time_mode = TimeMode.MESSAGE if embodiment.simulated else TimeMode.CLOCK
    writer_cm = LocalDatasetWriter(output_dir) if output_dir is not None else nullcontext(None)
    with writer_cm as dataset_writer, pimm.World(virtual_time=embodiment.simulated) as world:
        ds_agent = wire.wire_embodiment(
            world, harness, embodiment, dataset_writer, time_mode, privileged=ev.privileged, done=ev.done
        )
        if ds_agent is not None:
            world.connect(harness.ds_command, ds_agent.command)

        # Sim schedules harness, recorder, then producers (the simulator) in-process under the virtual clock,
        # in that order; each scheduler round is one control period. The harness decides the round's action
        # (a reset, a policy command off the last round's observation, or finish); the recorder logs that
        # observation with the command; the producer applies the command and publishes the next observation.
        # A reset arms the producer to publish frame-0 after the harness (last in the round); the recorder
        # drains its channels the turn it opens, dropping the pre-reset frame, so its first recorded sample
        # is the post-reset scene. Real runs the producers + recorder as background subprocesses; the
        # harness's placement is otherwise identical.
        producers = [cs for cs in embodiment.control_systems if cs is not None]
        if embodiment.simulated:
            world.run([harness, ds_agent, *producers])
        else:
            world.run([harness], [*producers, ds_agent])


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
    """Run an unattended sweep: the harness self-drives each eval's trial plan, rebuilding the World per eval.

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
    policy.new_session().close()
    output_dir = prepare_output_dir(output_dir)

    try:
        with timed_pass(output_dir, timing, policy):
            for ev in evals:
                _run_world(policy, ev, output_dir)
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
    """Run a selected eval (embodiment + task + its trial sweep) through the shared inference harness.

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
        # The eval config owns the trial sweep (seed, task range); ``charge_inference_time`` is the CLI's
        # per-run knob (whether a sim charges each call the time it took). Overlay it onto every trial
        # context, then self-drive the eval.
        eval = replace(
            eval, trials=[{**trial, keys.CHARGE_INFERENCE_TIME: charge_inference_time} for trial in eval.trials]
        )
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
