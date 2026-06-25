import logging
import random
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import configuronic as cfn
import pos3

import pimm
import positronic.cfg.policy as policy_cfg
import positronic.cfg.wrappers as wrappers_cfg
from positronic import utils, wire
from positronic.cfg.eval import placeholder
from positronic.dataset.ds_writer_agent import TimeMode
from positronic.dataset.local_dataset import LocalDatasetWriter, load_all_datasets
from positronic.eval import Embodiment, Eval, Task
from positronic.gui.dpg import DearpyguiUi
from positronic.policy.base import SampledPolicy
from positronic.policy.harness import Harness, default_wrappers

logger = logging.getLogger(__name__)


@dataclass
class Driver:
    """An attended operator surface: the directive source ``main`` wires into the Harness.

    Driver configs produce a factory called with the resolved local output directory, since
    the directory exists only after ``pos3.sync`` inside ``main``.
    """

    gui: DearpyguiUi | None
    directives: pimm.SignalEmitter
    directive_wrapper: Callable
    control_systems: list[pimm.ControlSystem]


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


def main(
    embodiment: Embodiment,
    policy,
    driver: Callable[[Path | None], Driver] | None = None,
    output_dir: str | Path | None = None,
    task: Task | None = None,
    trials: list[dict] | None = None,
    show_gui: bool = False,
    wrap=default_wrappers,
):
    """Run inference for an embodiment, real or simulated.

    ``task`` (when given) supplies the policy-facing instruction, the per-trial ``timeout``
    bounding self-driven trials, the privileged ground-truth signals to record, and the seeded
    scene reset run at each trial start; without it the instruction rides the driver. Exactly
    one of ``driver`` (attended: a factory producing the operator surface that emits the
    directives) or ``trials`` (unattended: the harness runs the plan itself) must be given;
    ``show_gui`` applies to the unattended path (attended surfaces bring their own).
    """
    assert (driver is None) != (trials is None), 'Provide exactly one of driver or trials'

    # Drive the policy's remote endpoints through their cold start before hardware and the operator
    # surface come up: opening a session blocks on the server handshake, which returns only once the
    # model is loaded, and a SampledPolicy reaches every sub-policy. The first episode then begins
    # warm instead of stalling on an on-request endpoint's model load while the robot waits.
    # TODO: a policy with recording taps (recording_dir set) records this throwaway warmup session —
    # an empty .rrd plus a bump to the recorder's episode counter — but warmup is not a real episode.
    logger.info('Warming up policy endpoints')
    policy.new_session().close()

    harness = Harness(
        policy, embodiment, task=task, trials=trials, wrap=wrap, on_episode_complete=_completion_sink(policy)
    )

    if output_dir is not None:
        output_dir = pos3.sync(output_dir, sync_on_error=True)
        utils.save_run_metadata(output_dir, patterns=['*.py', '*.toml'])
        _seed_counter(policy, output_dir)

    driver = driver(output_dir) if driver is not None else None
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


@cfn.config(
    eval=placeholder, policy=policy_cfg.placeholder, trial_count=1, show_gui=False, wrap=wrappers_cfg.default_wrappers
)
def run(
    eval: Eval,
    policy,
    trial_count,
    show_gui,
    output_dir=None,
    inference_latency=False,
    wrap=wrappers_cfg.default_wrappers,
):
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
    main(
        embodiment=eval.embodiment,
        task=eval.task,
        policy=policy,
        trials=trials,
        show_gui=show_gui,
        output_dir=output_dir,
        wrap=wrap,
    )
