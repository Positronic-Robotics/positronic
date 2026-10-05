"""Legacy ``positronic-inference`` CLI: the attended ``real`` (keyboard) and ``web`` (browser console) paths, plus
the ``sim`` and ``stats`` aliases over ``cli.eval.run``."""

import logging
from collections import Counter
from collections.abc import Callable, Iterator
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
from positronic.dataset.local_dataset import load_all_datasets
from positronic.drivers.keyboard import KeyboardControl
from positronic.eval import Embodiment, Task
from positronic.eval import keys as eval_keys
from positronic.gui.web import StationConsole
from positronic.policy import Policy
from positronic.policy import keys as policy_keys
from positronic.policy.harness import Harness, Rollout
from positronic.simulator.env_server.telemetry import ENV_TELEMETRY_DIR

logger = logging.getLogger(__name__)


class KeyboardOperator(KeyboardControl):
    """The keyboard that runs episodes: ``s`` asks for one through ``perform_task``, ``p`` ends the live one,
    ``q`` ends the run.

    One episode is in flight at a time: a press while one runs is declined here, with a warning. It holds
    the pending answer because that is where the episode's terminal — or a refused ask — arrives, and it
    logs that as it lands. ``next_task`` makes the trial and the policy opens its session, once per accepted
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

    The keyboard shows nothing; ``web`` is the attended path that shows the cameras. A run ends when the
    operator returns — on ``q``, or on a stdin that is not a terminal — since a control system returning stops
    the world.
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


class TrialForwarder(pimm.ControlSystem):
    """Asks the harness to perform each trial that ``run_trial`` receives, as the rollout ``rollout_of`` makes, and
    answers the call with the harness's answer.

    It runs beside the harness for a caller in another process, which cannot hand the harness a policy.
    """

    def __init__(self, rollout_of: Callable[[Task], Rollout]):
        self._rollout_of = rollout_of
        self.run_trial = pimm.calls.ControlSystemHandler[Task, dict[str, Any]](self)
        self.perform_task = pimm.calls.ControlSystemCaller[Rollout, dict[str, Any]](self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        pending: list[tuple[pimm.calls.Call[Task, dict[str, Any]], pimm.calls.Answer[dict[str, Any]]]] = []
        while not should_stop.value:
            for call in self.run_trial.incoming():
                pending.append((call, self.perform_task(self._rollout_of(call.request))))
            for call, answer in pending:
                if answer.done():
                    with pimm.calls.raise_to(call):
                        call.set_result(answer.result())
            pending = [(call, answer) for call, answer in pending if not answer.done()]
            yield pimm.Sleep(FORWARD_POLL_S)


def web(
    policy,
    embodiment: Embodiment,
    next_task: Callable[[], Task],
    output_dir=None,
    host: str = '127.0.0.1',
    port: int = 8080,
):
    """Run one hardware embodiment attended from a browser, at ``http://{host}:{port}/``.

    The page shows each camera, starts an episode on the trial ``next_task`` makes, and ends it with a verdict or a
    discard. The run ends on Ctrl-C. The policy's metadata is read once before the page opens, so a policy that
    cannot answer stops the run there.
    """
    if embodiment.simulated:
        raise ValueError('the web console drives hardware in real time; run a simulated embodiment as `sim`')

    logger.info('Reading the policy metadata')
    label = policy.meta().get(policy_keys.TYPE, type(policy).__name__)
    with scoped_env_var(ENV_TELEMETRY_DIR):
        output_path = prepare_output_dir(output_dir)
        console = StationConsole(next_task, policy=label, host=host, port=port)
        forwarder = TrialForwarder(partial(Rollout, policy=policy, output_path=output_path))
        harness = Harness(embodiment)
        with pimm.World() as world:
            ds_agent = wire.wire_embodiment(
                world, harness, embodiment, record=output_path is not None, done=console.done
            )
            world.connect(console.run_trial, forwarder.run_trial)
            world.connect(forwarder.perform_task, harness.perform_task)
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
