"""The round driver: a command that runs DAgger rounds, and the driver of round 0, ``RoundZeroPolicy``::

    run(
        rig=rig,
        tasks=tasks,
        model=model,
        rounds=3,
        episodes=20,
        output_dir='~/dagger/1',
        round_0_license='internal',
        training_licenses=['internal', 'apache-2.0'],
    )

Each round ``k`` writes to the folder ``output_dir/round_k``: ``records``, ``training_data``, ``converted`` and
``checkpoint``.
"""

import logging
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import configuronic as cfn
import numpy as np
from positronic_wire.wire import SESSION_PATH, HostPortAddress

from positronic import keys
from positronic.dagger.training_data import write_training_data
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.drivers.roboarm.command import JointPosition
from positronic.eval import Task
from positronic.eval import keys as eval_keys
from positronic.policy.base import Commands, Obs, Policy, PolicyRun, Runtime, Step
from positronic.policy.dagger import DaggerDataset
from positronic.policy.harness import Rollout
from positronic.policy.keys import LICENSE, NO_LICENSE, TYPE
from positronic.policy.license import LicenseRule
from positronic.policy.remote import RemotePolicy
from positronic.simulator.env_server.launcher import serve_subprocess

# The time between two steps of ``RoundZeroPolicy``.
PERIOD_NS = 20_000_000
# The folders of a DAgger round. The converter and the trainer get them by these names.
ROUND_FOLDERS = ('records', 'training_data', 'converted', 'checkpoint')


def _hold(commands: Collection[str], obs: Obs) -> Commands:
    """The commands on the channels ``commands`` that keep each arm at the joints and the grip that ``obs`` gives."""
    held: Commands = {}
    for channel in commands:
        base, _, name = channel.partition('.')
        arm = name or None
        if base == keys.ROBOT_COMMAND:
            joints = obs[f'{keys.arm_channel(keys.ROBOT_STATE, arm)}{keys.JOINTS_SUFFIX}']
            held[channel] = JointPosition(positions=np.asarray(joints))
        elif base == keys.TARGET_GRIP:
            held[channel] = float(obs[keys.arm_channel(keys.GRIP, arm)])
        else:
            raise ValueError(
                f'The round 0 policy commands only {keys.ROBOT_COMMAND} and {keys.TARGET_GRIP}, not {channel}'
            )
    return held


class RoundZeroPolicy(Policy):
    """The driver of DAgger round 0. At each step it commands the joints and the grip of the first observation, on
    the command channels ``commands`` of the rig."""

    WIRE_NAME = 'dagger_round_0'

    def __init__(self, license: str, commands: Collection[str]):
        self._license = license
        self._commands = tuple(commands)

    def meta(self) -> dict[str, Any]:
        return {TYPE: self.WIRE_NAME, LICENSE: self._license}

    def run(self, runtime: Runtime) -> PolicyRun:
        obs = yield
        commands = _hold(self._commands, obs)
        while True:
            yield Step(dict(commands), runtime.time_ns + PERIOD_NS)


class Rig(ABC):
    """The rig of the Collect step and its teleoperator. In simulation, an env server and a scripted expert. On a rig
    with no platform, the drivers and a teleop device."""

    @property
    @abstractmethod
    def observations(self) -> Collection[str]:
        """The observation signals of the rig. The DAgger reader keeps them as the observation of a step."""

    @property
    @abstractmethod
    def commands(self) -> Collection[str]:
        """The command channels of the rig."""

    @abstractmethod
    def harness(
        self, *, takes_over_at_start: bool
    ) -> AbstractContextManager[tuple[Callable[[Rollout], dict[str, Any]], Policy]]:
        """Run the harness of the rig for the context. The value is a pair: a function that runs one rollout and returns
        its answer, and the teleoperator. ``takes_over_at_start`` is true in round 0, where the teleoperator takes over
        at the start of each episode."""


@dataclass(frozen=True)
class ModelCommands:
    """The model that the DAgger rounds train: the labels that it learns, and the commands of its converter, its
    trainer and its policy server. Each command is a list of arguments, which ``str.format`` fills. The converter and
    the trainer get the ``ROUND_FOLDERS`` of the round. The server gets the ``checkpoint`` of the round before, the
    ``host`` and the ``port`` that it binds, and ``license``, the license of each checkpoint."""

    labels: tuple[str, ...]
    convert: tuple[str, ...]
    train: tuple[str, ...]
    serve: tuple[str, ...]
    license: str
    # The wire of the policy server, as ``RemotePolicy`` names it.
    wire: str = 'websocket'

    def __post_init__(self):
        if not self.labels:
            raise ValueError('The model learns at least one label')
        folders = _folders(Path('output'), 0)
        server = _server_values(folders['checkpoint'], 'localhost', 0, self.license)
        for name, command, values in (
            ('convert', self.convert, folders),
            ('train', self.train, folders),
            ('serve', self.serve, server),
        ):
            try:
                _arguments(command, values)
            except (KeyError, AttributeError, IndexError, ValueError) as error:
                raise ValueError(f'The {name} command gets only {", ".join(values)}, but: {error!r}') from error


def _folders(output: Path, index: int) -> dict[str, Path]:
    """The folders of DAgger round ``index``, by the names that the commands of the model give them."""
    return {name: output / f'round_{index}' / name for name in ROUND_FOLDERS}


def _server_values(checkpoint: Path, host: str, port: int, license: str) -> dict[str, Any]:
    return {'checkpoint': checkpoint, 'host': host, 'port': port, 'license': license}


def _arguments(command: Sequence[str], values: Mapping[str, Any]) -> list[str]:
    return [argument.format(**values) for argument in command]


def _run(command: Sequence[str], values: Mapping[str, Any]) -> None:
    arguments = _arguments(command, values)
    logging.info('Running %s', ' '.join(arguments))
    subprocess.run(arguments, check=True)


@contextmanager
def _served(model: ModelCommands, checkpoint: Path) -> Iterator[RemotePolicy]:
    """The policy server of ``model`` for ``checkpoint``, for the context, and a remote policy that it serves."""

    def spawn(host: str, port: int) -> subprocess.Popen:
        arguments = _arguments(model.serve, _server_values(checkpoint, host, port, model.license))
        logging.info('Starting %s', ' '.join(arguments))
        return subprocess.Popen(arguments)

    with serve_subprocess(spawn, 'localhost') as (host, port):
        policy = RemotePolicy(model.wire, HostPortAddress(host, port, SESSION_PATH, ''))
        policy.wait_for_server()
        yield policy


def _collect(
    rig: Rig,
    tasks: Sequence[Task],
    policy: Policy,
    records: Path,
    license_rule: LicenseRule,
    *,
    takes_over_at_start: bool,
) -> list[dict[str, Any]]:
    with rig.harness(takes_over_at_start=takes_over_at_start) as (perform, teleoperator):
        _check_license(license_rule, 'the teleoperator', teleoperator.meta().get(LICENSE, NO_LICENSE))
        writer = LocalDatasetWriter(records)
        return [perform(Rollout(task, policy, writer, teleoperator)) for task in tasks]


def _check_license(rule: LicenseRule, source: str, license: str) -> None:
    """Raise when ``rule`` does not allow ``license``, the license of ``source``."""
    if license == NO_LICENSE:
        raise ValueError(
            f'{source.capitalize()} states no license, so the license rule drops each record in which it drives. Give '
            f'{source} a license'
        )
    if not rule.allows(license):
        raise ValueError(
            f'The licenses that allow training do not have {license!r}, the license of {source}. The license rule '
            f'drops each record in which {source} drives, so the round gives no training data. Add it to '
            'training_licenses'
        )


def _round(
    index: int,
    rig: Rig,
    tasks: Sequence[Task],
    driver: AbstractContextManager[Policy],
    model: ModelCommands,
    folders: Mapping[str, Path],
    license_rule: LicenseRule,
    data: Sequence[DaggerDataset],
) -> list[DaggerDataset]:
    """DAgger round ``index`` in ``folders``: the Collect step of ``tasks`` with the policy of ``driver``, the training
    data of ``data`` and of this round, and the convert and the train steps of ``model``. The value is ``data`` with the
    data of this round."""
    with driver as policy:
        results = _collect(rig, tasks, policy, folders['records'], license_rule, takes_over_at_start=index == 0)
    this_round = DaggerDataset(LocalDataset(folders['records']), rig.observations, model.labels, license_rule)
    data = [*data, this_round]
    write_training_data(data, folders['training_data'])
    logging.info(
        'Round %d: %d episodes, %d did the task, the license rule drops %d; %d teleoperator runs of %d steps in '
        'all; steps with no shadow action: %s. The training data has %d episodes of rounds 0 to %d.',
        index,
        len(results),
        sum(bool(result.get(eval_keys.SUCCESS)) for result in results),
        this_round.dropped_records,
        len(this_round),
        sum(len(each.episode[model.labels[0]]) for each in this_round.runs),
        dict(this_round.steps_without_shadow),
        sum(len(each) for each in data),
        index,
    )
    _run(model.convert, folders)
    _run(model.train, folders)
    return data


@cfn.config()
def run(
    rig: Rig,
    tasks: Iterable[Task],
    model: ModelCommands,
    rounds: int,
    episodes: int,
    output_dir: str,
    round_0_license: str,
    training_licenses: Collection[str],
) -> None:
    """Run ``rounds`` DAgger rounds of ``episodes`` episodes each on ``rig``, with the teleoperator of ``rig`` as the
    helper. Each episode takes the next task of ``tasks``. ``round_0_license`` is the license of ``RoundZeroPolicy``.
    The training data holds the teleoperator runs of each record that the license rule keeps with
    ``training_licenses``."""
    if rounds < 1 or episodes < 1:
        raise ValueError(f'A run has at least one round of at least one episode, not {rounds} of {episodes}')
    license_rule = LicenseRule(training_licenses)
    _check_license(license_rule, 'the round 0 policy', round_0_license)
    if rounds > 1:
        _check_license(license_rule, 'the model', model.license)
    output = Path(output_dir).expanduser()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'The DAgger rounds go to a new folder, but {output} has files')
    plan = iter(tasks)
    data: list[DaggerDataset] = []
    for index in range(rounds):
        round_tasks = list(islice(plan, episodes))
        if len(round_tasks) < episodes:
            raise ValueError(f'Round {index} has {episodes} episodes, but the tasks gave only {len(round_tasks)}')
        driver: AbstractContextManager[Policy] = (
            nullcontext(RoundZeroPolicy(round_0_license, rig.commands))
            if index == 0
            else _served(model, _folders(output, index - 1)['checkpoint'])
        )
        data = _round(index, rig, round_tasks, driver, model, _folders(output, index), license_rule, data)
    logging.info(
        'The checkpoint of the last round, the result of the run, is in %s', _folders(output, rounds - 1)['checkpoint']
    )
