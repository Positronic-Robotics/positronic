"""Legacy ``positronic-inference`` CLI: the attended ``real`` (keyboard) and ``web`` (browser console) paths, plus
the ``sim`` and ``stats`` aliases over ``cli.eval.run``."""

import logging
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import configuronic as cfn
import pos3

import pimm
import positronic.cfg.embodiment
import positronic.cfg.eval.real.droid
import positronic.cfg.policy as policy_cfg
from pimm.logging import init_logging
from positronic import keys, wire
from positronic.cfg.eval.sim.positronic import stack_cubes
from positronic.cli.eval.run import prepare_output_dir, run, run_world, scoped_env_var
from positronic.dataset.ds_writer_agent import DsWriterCommand, DsWriterCommandType
from positronic.dataset.local_dataset import load_all_datasets
from positronic.drivers.keyboard import KeyboardControl
from positronic.eval import Embodiment, Task
from positronic.eval import keys as eval_keys
from positronic.gui.web import EndTrial, StationConsole
from positronic.policy import Policy
from positronic.policy import keys as policy_keys
from positronic.policy.harness import Harness, Rollout
from positronic.simulator.env_server.telemetry import ENV_TELEMETRY_DIR

logger = logging.getLogger(__name__)


class KeyboardOperator(KeyboardControl):
    """The keyboard that runs episodes: ``s`` asks for one through ``perform_task``, ``p`` ends the live one,
    ``q`` ends the run.

    One episode is in flight at a time: a press while one runs is declined here, with a warning. It holds
    the pending answer because that is where the episode's terminal, its error or a refused ask arrives, and
    it logs that as it lands. ``next_task`` makes the trial and the policy opens its session, once per accepted
    press. Every episode records into ``output_path``, and none records when that is ``None``.
    """

    def __init__(self, next_task: Callable[[], Task], policy: Policy, output_path: Path | None):
        super().__init__(quit_key='q')
        self._next_task = next_task
        self._policy = policy
        self._output_path = output_path
        self._pending: pimm.calls.Answer[dict[str, Any]] | None = None
        self.perform_task = pimm.calls.ControlSystemCaller[Rollout, dict[str, Any]](self)
        self.done = pimm.ControlSystemEmitter[dict[str, Any]](self)

    def _each_round(self, key: str | None) -> None:
        super()._each_round(key)
        if self._pending is not None and self._pending.done():
            try:  # rules-allow: swallowed-error — the operator is who this failure is for
                logger.info(f'Episode ended: {self._pending.result()}')
            except Exception as e:
                logger.error(f'Episode failed: {e}')
            self._pending = None
        match key:
            case 's' if self._pending is not None:
                logger.warning('An episode is already running: press [p] to stop it')
            case 's':
                try:
                    rollout = Rollout(self._next_task(), self._policy, self._output_path)
                    self._pending = self.perform_task(rollout)
                except Exception as e:  # rules-allow: swallowed-error — the operator is who this failure is for
                    logger.error(f'Episode failed to open: {e}')
            case 'p':
                self.done.emit({eval_keys.ENDED_BY: eval_keys.ENDED_BY_OPERATOR})


def real(policy, embodiment: Embodiment, next_task: Callable[[], Task], output_dir=None):
    """Run one hardware embodiment attended and headless, the keyboard deciding when an episode starts and
    finishes.

    The keyboard shows nothing; ``web`` shows the cameras. A run ends when the operator returns — on ``q``, or on
    a stdin that is not a terminal — since a control system returning stops the world.
    """
    if embodiment.simulated:
        raise ValueError('the keyboard path drives hardware in real time; run a simulated embodiment as `sim`')

    with scoped_env_var(ENV_TELEMETRY_DIR):
        output_path = prepare_output_dir(output_dir)
        operator = KeyboardOperator(next_task, policy, output_path)
        logger.info('Keyboard controls: [s]tart, sto[p], [q]uit')
        run_world(embodiment, operator, record=output_path is not None, done=operator.done)


real_cfg = cfn.Config(
    real,
    embodiment=positronic.cfg.embodiment.droid,
    next_task=positronic.cfg.eval.real.droid.attended_trials,
    policy=policy_cfg.placeholder,
)


# The lag between a Start and the harness taking the trial, and between the harness's answer and the console.
FORWARD_POLL_S = 0.02

TrialCall = pimm.calls.Call[Task | EndTrial, dict[str, Any]]


@dataclass
class _OpenTrial:
    """The trial the harness holds, and the verdict that waits for the harness to start it."""

    call: TrialCall
    answer: pimm.calls.Answer[dict[str, Any]]
    started: bool = False
    stopped: bool = False
    verdict: dict[str, Any] | None = None
    delivered: bool = False
    # The calls that ended this trial. Each gets the trial's answer.
    ends: list[TrialCall] = field(default_factory=list)

    def take_verdict(self, call: TrialCall, payload: dict[str, Any]) -> None:
        self.ends.append(call)
        if self.verdict is None and not self.delivered:
            self.verdict = payload

    def observe(self, command: DsWriterCommand) -> None:
        if command.type is DsWriterCommandType.START_EPISODE:
            self.started = True
        else:
            self.stopped = True

    def deliver(self, done: pimm.SignalEmitter[dict[str, Any]]) -> None:
        if self.started and not self.stopped and self.verdict is not None:
            done.emit(self.verdict)
            self.verdict, self.delivered = None, True

    def close(self) -> None:
        for call in (self.call, *self.ends):
            with pimm.calls.raise_to(call):
                call.set_result(self.answer.result())


class TrialForwarder(pimm.ControlSystem):
    """Runs the trials and verdicts that ``trials`` receives: a ``Task`` as the rollout ``rollout_of`` makes, and an
    ``EndTrial`` as one ``done`` to the harness.

    It runs beside the harness for a caller in another process, which cannot hand the harness a policy. One trial
    runs at a time. A verdict reaches the harness once, after the harness's recorder command shows that the harness
    started that trial, and never after the trial stops: the harness drops a ``done`` that reaches it while idle.
    Connect ``recorder`` to the harness's ``ds_command`` and ``done`` to its ``done``.
    """

    def __init__(self, rollout_of: Callable[[Task], Rollout]):
        self._rollout_of = rollout_of
        self.trials = pimm.calls.ControlSystemHandler[Task | EndTrial, dict[str, Any]](self)
        self.perform_task = pimm.calls.ControlSystemCaller[Rollout, dict[str, Any]](self)
        self.recorder = pimm.ControlSystemReceiver[DsWriterCommand](self)
        self.done = pimm.ControlSystemEmitter[dict[str, Any]](self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        trial: _OpenTrial | None = None
        while not should_stop.value:
            # Read before the new calls: a recorder command read here belongs to a trial taken in an earlier round.
            command = pimm.value_updated(self.recorder)
            if trial is not None and command is not None:
                trial.observe(command)
            for call in self.trials.incoming():
                trial = self._receive(call, trial)
            if trial is not None:
                trial.deliver(self.done)
                if trial.answer.done():
                    trial.close()
                    trial = None
            yield pimm.Sleep(FORWARD_POLL_S)

    def _receive(self, call: TrialCall, trial: _OpenTrial | None) -> _OpenTrial | None:
        request = call.request
        if isinstance(request, Task):
            if trial is not None:
                call.set_exception(RuntimeError('a trial is already running'))
                return trial
            return _OpenTrial(call, self.perform_task(self._rollout_of(request)))
        if trial is None:
            call.set_exception(RuntimeError('the trial ended before its verdict arrived'))
            return None
        trial.take_verdict(call, request.payload)
        return trial


@scoped_env_var(ENV_TELEMETRY_DIR)
def web(
    policy,
    embodiment: Embodiment,
    next_task: Callable[[], Task],
    output_dir=None,
    host: str = '127.0.0.1',
    port: int = 8080,
):
    """Run one hardware embodiment attended from a browser, at ``http://{host}:{port}/``.

    The page shows each camera, starts an episode on the trial ``next_task`` makes, and ends it with a pass or fail
    verdict. The run ends on the page's End run or on Ctrl-C. The policy's metadata is read once before the page opens,
    so a policy that cannot answer stops the run there.
    """
    if embodiment.simulated:
        raise ValueError('the web console drives hardware in real time; run a simulated embodiment as `sim`')

    logger.info('Reading the policy metadata')
    label = policy.meta().get(policy_keys.TYPE, type(policy).__name__)
    output_path = prepare_output_dir(output_dir)
    console = StationConsole(next_task, policy=label, host=host, port=port)
    forwarder = TrialForwarder(partial(Rollout, policy=policy, output_path=output_path))
    harness = Harness(embodiment)
    with pimm.World() as world:
        ds_agent = wire.wire_embodiment(world, harness, embodiment, record=output_path is not None, done=forwarder.done)
        world.connect(console.trials, forwarder.trials)
        world.connect(forwarder.perform_task, harness.perform_task)
        world.connect(harness.ds_command, forwarder.recorder)
        for name, observation in embodiment.observations.items():
            if name.startswith(keys.IMAGE_PREFIX):
                world.connect(observation.source, console.cameras[name])
        if ds_agent is not None:
            world.connect(harness.ds_command, ds_agent.command)
        producers = [cs for cs in embodiment.control_systems if cs is not None]
        world.run([forwarder, harness], [*producers, ds_agent, console])


web_cfg = cfn.Config(
    web,
    embodiment=positronic.cfg.embodiment.droid,
    next_task=positronic.cfg.eval.real.droid.attended_trials,
    policy=policy_cfg.placeholder,
)


# Console entry point for [project.scripts].
@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli({
        'run': real_cfg,
        'real': real_cfg,  # `real` is the documented name for the hardware path
        'web': web_cfg,
        'sim': run.override(eval=stack_cubes),
        'stats': stats,
    })


@cfn.config(fields=['eval.object', 'eval.external_camera', 'eval.tote_placement'])
def stats(output_dir: str, fields: list[str]):
    dataset = load_all_datasets(pos3.sync(output_dir))
    counts = Counter()
    for i in range(len(dataset)):
        static = dataset[i].static
        counts[tuple(static.get(f, 'N/A') for f in fields)] += 1

    n = len(fields)
    subtotals = [0] * n
    prev_key = None

    def _print_subtotal(level):
        row = list(prev_key[:level]) + ['Total'] + [''] * (n - level - 1) + [str(subtotals[level])]
        print('\t'.join(row))
        subtotals[level] = 0

    print('\t'.join(fields + ['count']))
    for key, count in sorted(counts.items()):
        if prev_key is not None:
            change_level = next((i for i in range(n) if key[i] != prev_key[i]), n)
            for level in range(n - 1, change_level, -1):
                _print_subtotal(level)

        print('\t'.join([*key, str(count)]))
        for level in range(n):
            subtotals[level] += count
        prev_key = key

    if prev_key is not None:
        for level in range(n - 1, -1, -1):
            _print_subtotal(level)


if __name__ == '__main__':
    _internal_main()
