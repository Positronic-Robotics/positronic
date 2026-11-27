import json
import logging
import socket
import sys
from collections.abc import Collection
from contextlib import contextmanager
from dataclasses import replace
from itertools import islice
from pathlib import Path

import configuronic as cfn
import numpy as np
import pytest

import pimm
from pimm.tests.testing import Passive
from positronic import keys
from positronic.cfg import dagger
from positronic.dagger import META_RECORD_UID, META_ROUND
from positronic.dagger.env_server import EnvServerRig
from positronic.dagger.rounds import ModelCommands, Rig, RoundZeroPolicy, run
from positronic.dagger.tests import stub_model
from positronic.dataset.episode import META_UID
from positronic.dataset.local_dataset import LocalDataset
from positronic.drivers.roboarm.command import JointPosition
from positronic.drivers.roboarm.tests.fakes import make_robot_state
from positronic.eval import Task
from positronic.eval import keys as eval_keys
from positronic.policy import keys as policy_keys
from positronic.policy.base import Policy
from positronic.policy.executor import Executor
from positronic.policy.teleoperator import Teleoperator
from positronic.policy.tests import test_dagger as dagger_tests
from positronic.simulator.robolab import keys as robolab_keys
from positronic.simulator.robolab.expert import BananaInBowlExpert, layout

_LABELS = (keys.TARGET_EE_POSE, keys.TARGET_GRIP)
_STUB = (sys.executable, '-m', 'positronic.dagger.tests.stub_model')
_STUB_MODEL = ModelCommands(
    labels=_LABELS,
    convert=(*_STUB, 'convert', '--training_data={training_data}', '--output_dir={converted}'),
    train=(*_STUB, 'train', '--input_path={converted}', '--output_dir={checkpoint}'),
    serve=(
        *(*_STUB, 'serve', '--model.checkpoint={checkpoint}', '--model.license={license}'),
        *('--websocket.served_address.host={host}', '--websocket.served_address.port={port}'),
    ),
    license='test',
)
_NO_MODEL = ModelCommands(labels=_LABELS, convert=('true',), train=('true',), serve=('true',), license='test')
_SHADOW = f'{policy_keys.SHADOW}.{policy_keys.POLICY_SOURCE}.{_LABELS[0]}'


def _joints(arm: str) -> str:
    return f'{keys.arm_channel(keys.ROBOT_STATE, arm)}{keys.JOINTS_SUFFIX}'


def test_the_round_0_policy_commands_at_each_step_the_joints_and_the_grip_of_the_first_observation():
    arms = keys.BIMANUAL_ARMS
    first = {
        _joints(keys.LEFT_ARM): np.array([0.1, 0.2]),
        f'{keys.arm_channel(keys.ROBOT_STATE, keys.LEFT_ARM)}{keys.EE_POSE_SUFFIX}': np.zeros(7),
        keys.arm_channel(keys.GRIP, keys.LEFT_ARM): 0.3,
        _joints(keys.RIGHT_ARM): np.array([0.4, 0.5]),
        keys.arm_channel(keys.GRIP, keys.RIGHT_ARM): 0.6,
        keys.EXTERIOR_IMAGE: np.zeros((2, 2, 3)),
    }
    later = {name: value + 1 for name, value in first.items()}
    commands = [keys.arm_channel(channel, arm) for channel in (keys.ROBOT_COMMAND, keys.TARGET_GRIP) for arm in arms]
    runtime = Executor(lambda: 0, simulated=True, charge_inference_time=False)
    policy_run = runtime.start(RoundZeroPolicy('test', commands))

    for obs in (first, later, later):
        step = policy_run.send(obs)
        assert step is not None
        assert sorted(step.commands) == sorted(commands)
        for arm in arms:
            command = step.commands[keys.arm_channel(keys.ROBOT_COMMAND, arm)]
            assert isinstance(command, JointPosition)
            np.testing.assert_array_equal(command.positions, first[_joints(arm)])
            assert step.commands[keys.arm_channel(keys.TARGET_GRIP, arm)] == first[keys.arm_channel(keys.GRIP, arm)]


def test_the_round_0_policy_refuses_a_channel_that_is_not_an_arm_or_a_grip():
    runtime = Executor(lambda: 0, simulated=True, charge_inference_time=False)
    policy_run = runtime.start(RoundZeroPolicy('test', ['wheels']))

    with pytest.raises(ValueError, match='not wheels'):
        policy_run.send({})


def test_a_model_command_that_does_not_fill_fails_when_the_config_is_made():
    with pytest.raises(
        ValueError, match=r"The serve command gets only checkpoint, host, port, license, but: KeyError\('checkpoints'\)"
    ):
        ModelCommands(labels=_LABELS, convert=(), train=(), serve=('--model={checkpoints}',), license='test')
    with pytest.raises(ValueError, match=r'The train command .* KeyError\(\'port\'\)'):
        ModelCommands(labels=_LABELS, convert=(), train=('--port={port}',), serve=(), license='test')
    with pytest.raises(ValueError, match=r"The train command .* AttributeError.*'parnt'"):
        ModelCommands(labels=_LABELS, convert=(), train=('--output_dir={checkpoint.parnt}',), serve=(), license='test')


_TELEOPERATOR = RoundZeroPolicy('test', ())


class _NoSignals(Rig):
    """A rig with no observation and no command channel."""

    @property
    def observations(self) -> Collection[str]:
        return ()

    @property
    def commands(self) -> Collection[str]:
        return ()


class _Rig(_NoSignals):
    """A rig with no simulator and the teleoperator ``teleoperator``: each rollout of a Collect step opens its writer,
    records nothing and calls ``perform``."""

    def __init__(self, perform, teleoperator: Policy = _TELEOPERATOR):
        self._perform = perform
        self._teleoperator = teleoperator

    @contextmanager
    def harness(self, *, takes_over_at_start: bool):
        def perform(rollout):
            with rollout.writer:
                pass
            return self._perform(rollout)

        yield perform, self._teleoperator


def _run(
    tmp_path: Path,
    rig: Rig,
    *,
    tasks: int = 2,
    rounds: int = 2,
    episodes: int = 1,
    model: ModelCommands = _NO_MODEL,
    round_0_license: str = 'test',
    training_licenses=('test',),
) -> Path:
    output = tmp_path / 'rounds'
    run.override(
        rig=rig,
        tasks=[Task('lift the cube', 1.0)] * tasks,
        model=model,
        rounds=rounds,
        episodes=episodes,
        output_dir=str(output),
        round_0_license=round_0_license,
        training_licenses=training_licenses,
    ).instantiate()
    return output


def test_a_run_refuses_a_folder_that_has_files(tmp_path):
    rollouts = []
    (tmp_path / 'rounds').mkdir()
    (tmp_path / 'rounds' / 'notes.txt').touch()

    with pytest.raises(ValueError, match='has files'):
        _run(tmp_path, _Rig(rollouts.append))
    assert rollouts == []


@pytest.mark.parametrize(
    ('teleoperator', 'change', 'error'),
    [
        (_TELEOPERATOR, {'training_licenses': ('test', policy_keys.NO_LICENSE)}, 'missing license'),
        (_TELEOPERATOR, {'training_licenses': ()}, 'No license allows training'),
        (_TELEOPERATOR, {'round_0_license': 'gpl-3.0'}, "not have 'gpl-3.0', the license of the round 0 policy"),
        (RoundZeroPolicy('gpl-3.0', ()), {}, "not have 'gpl-3.0', the license of the teleoperator"),
        (
            Teleoperator(pimm.ControlSystemReceiver[dict](Passive()), policy_keys.NO_LICENSE),
            {},
            'The teleoperator states no license',
        ),
        (_TELEOPERATOR, {'model': replace(_NO_MODEL, license='gpl-3.0')}, "not have 'gpl-3.0', the license of the mod"),
    ],
)
def test_a_run_refuses_licenses_that_allow_no_training_data_before_the_first_episode(
    tmp_path, teleoperator, change, error
):
    rollouts = []

    with pytest.raises(ValueError, match=error):
        _run(tmp_path, _Rig(rollouts.append, teleoperator), **change)
    assert rollouts == []


def test_the_training_data_leaves_out_each_record_that_the_license_rule_drops_and_the_run_logs_it(tmp_path, caplog):
    licenses = iter([dagger_tests.ALLOWED, 'gpl-3.0'])
    model = replace(_NO_MODEL, labels=(dagger_tests.ARM,))

    def perform(rollout):
        static = {**dagger_tests.LICENSED, f'{policy_keys.POLICY_META}.{policy_keys.LICENSE}': next(licenses)}
        dagger_tests._record(rollout.writer.root, [dagger_tests._policy(), dagger_tests._teleoperator()], static=static)
        return {}

    with caplog.at_level(logging.INFO):
        output = _run(
            tmp_path, _Rig(perform), rounds=1, episodes=2, model=model, training_licenses=('test', dagger_tests.ALLOWED)
        )

    kept, dropped = LocalDataset(output / 'round_0' / 'records')
    training = LocalDataset(output / 'round_0' / 'training_data')
    assert [episode.meta[META_RECORD_UID] for episode in training] == [kept.meta[META_UID]]
    messages = [record.message for record in caplog.records]
    assert [message for message in messages if 'The license rule drops the record' in message] == [
        f'The license rule drops the record {dropped.meta[META_UID]}: { ({policy_keys.POLICY_SOURCE: "gpl-3.0"}) }'
    ]
    assert 'Round 0: 2 episodes, 0 did the task, the license rule drops 1; 1 teleoperator runs' in ' '.join(messages)


def test_too_few_tasks_stop_the_run_before_the_collect_step_of_that_round(tmp_path):
    rollouts = []

    with pytest.raises(ValueError, match='Round 1 has 1 episodes, but the tasks gave only 0'):
        _run(tmp_path, _Rig(lambda rollout: rollouts.append(rollout) or {}), tasks=1)
    assert len(rollouts) == 1
    assert not (tmp_path / 'rounds' / 'round_1').exists()


def test_an_episode_error_ends_the_run(tmp_path):
    def perform(rollout):
        raise RuntimeError('the rig stopped')

    with pytest.raises(RuntimeError, match='the rig stopped'):
        _run(tmp_path, _Rig(perform))
    assert not (tmp_path / 'rounds' / 'round_0' / 'training_data').exists()


def _accepts(port: int) -> bool:
    try:
        socket.create_connection(('localhost', port), timeout=1.0).close()
    except OSError:
        return False
    return True


# Write to the file ``argv[2]`` whether the port in the file ``argv[1]`` accepts, when that file exists, then run the
# command of the rest of the arguments.
_PROBE_THEN_RUN = """
import os, socket, sys
from pathlib import Path
port_file, probes, *command = sys.argv[1:]
if Path(port_file).exists():
    try:
        socket.create_connection(('localhost', int(Path(port_file).read_text())), timeout=1.0).close()
        state = 'up'
    except OSError:
        state = 'down'
    with open(probes, 'a') as file:
        file.write(state + chr(10))
os.execvp(command[0], command)
"""


@pytest.mark.timeout(300.0)
def test_the_policy_server_runs_for_the_collect_step_of_round_1_and_stops_before_its_convert_step(tmp_path):
    port_file, probes = tmp_path / 'port', tmp_path / 'probes'
    serve = ('sh', '-c', f'echo {{port}} > {port_file}; exec "$@"', 'sh', *_STUB_MODEL.serve)
    convert = (sys.executable, '-c', _PROBE_THEN_RUN, str(port_file), str(probes), *_STUB_MODEL.convert)
    model = replace(_STUB_MODEL, convert=convert, serve=serve)
    up_in_collect = []

    def perform(rollout):
        if port_file.exists():
            up_in_collect.append(_accepts(int(port_file.read_text())))
        return {}

    _run(tmp_path, _Rig(perform), model=model)

    assert up_in_collect == [True]
    assert probes.read_text().split() == ['down']


class _Arm(pimm.ControlSystem):
    """An arm that stands at the joints ``joints``."""

    def __init__(self, joints: list[float]):
        self.state = pimm.ControlSystemEmitter(self)
        self._joints = joints

    def run(self, should_stop, clock):
        while not should_stop.value:
            self.state.emit(make_robot_state([0.0, 0.0, 0.0], self._joints))
            yield pimm.Sleep(0.01)


class _TeleopDevice(pimm.ControlSystem):
    """A teleop device that reads the arm state. Its message commands the joints that it read last."""

    def __init__(self):
        self.arm_state = pimm.ControlSystemReceiver(self)
        self.inputs = pimm.ControlSystemEmitter(self)

    def run(self, should_stop, clock):
        while not should_stop.value:
            if (state := self.arm_state.read()) is not None:
                self.inputs.emit({keys.ROBOT_COMMAND: JointPosition(positions=state.data.q)})
            yield pimm.Sleep(0.01)


class _TeleopRig(_NoSignals):
    """A rig with no platform and a teleop device. Each Collect step opens a world with a new arm and a new device.
    The arm of Collect step ``k`` stands at the joints ``[k, k]``. Each rollout opens its writer, records nothing, and
    runs the teleoperator until its element has the message of the device. ``elements`` keeps that element."""

    def __init__(self):
        self.takes_over_at_start: list[bool] = []
        self.elements: list[dict] = []

    @contextmanager
    def harness(self, *, takes_over_at_start: bool):
        self.takes_over_at_start.append(takes_over_at_start)
        with pimm.World(virtual_time=True) as world:
            arm = _Arm([float(len(self.takes_over_at_start) - 1)] * 2)
            device = _TeleopDevice()
            # `Teleoperator` is not a control system, so the receiver that it reads needs an owner in the world.
            owner = Passive()
            inputs = pimm.ControlSystemReceiver[dict](owner)
            world.connect(arm.state, device.arm_state)
            world.connect(device.inputs, inputs)
            loop = world.start([arm, device, owner])

            def perform(rollout):
                with rollout.writer:
                    pass
                executor = Executor(world.clock.now_ns, simulated=True, charge_inference_time=False)
                teleoperator_run = executor.start(rollout.helper)
                for _ in range(100):
                    next(loop)
                    step = teleoperator_run.send({})
                    assert step is not None
                    if element := step.commands:
                        self.elements.append(element)
                        return {}
                raise AssertionError('The teleoperator got no message of the device')

            try:
                yield perform, Teleoperator(inputs, 'test')
            finally:
                world.request_stop()
                list(loop)


@pytest.mark.timeout(300.0)
def test_the_teleop_device_of_the_rig_reaches_the_teleoperator_in_the_world_of_each_collect_step(tmp_path):
    rig = _TeleopRig()

    _run(tmp_path, rig, model=_STUB_MODEL)

    assert rig.takes_over_at_start == [True, False]
    assert [list(element[keys.ROBOT_COMMAND].positions) for element in rig.elements] == [[0.0, 0.0], [1.0, 1.0]]


@pytest.fixture(scope='module')
def rounds(tmp_path_factory) -> Path:
    """Two DAgger rounds of one ``lift_cube`` episode each, on the Franka of ``MujocoSim`` with no cameras, with the
    stub model, from the config ``mujoco_lift_cube``. The folder of the rounds."""
    output = tmp_path_factory.mktemp('dagger') / 'rounds'
    overrides = {'rig.license': 'test', 'rig.cameras': None, 'tasks.timeout_sec': 6.0}
    dagger.mujoco_lift_cube.override(
        model=_STUB_MODEL,
        rounds=2,
        episodes=1,
        output_dir=str(output),
        round_0_license='test',
        training_licenses=['test'],
        **overrides,
    ).instantiate()
    return output


def _record(rounds: Path, index: int):
    [record] = LocalDataset(rounds / f'round_{index}' / 'records')
    return record


def _checkpoint(rounds: Path, index: int) -> str:
    return json.loads((rounds / f'round_{index}' / 'checkpoint' / stub_model.CHECKPOINT).read_text())['id']


@pytest.mark.timeout(600.0)
def test_in_round_0_the_round_0_policy_drives_and_the_expert_takes_over_at_the_start_and_lifts_the_cube(rounds):
    record = _record(rounds, 0)

    assert record[f'{policy_keys.POLICY_META}.{policy_keys.TYPE}'] == RoundZeroPolicy.WIRE_NAME
    assert record[f'{policy_keys.POLICY_META}.{policy_keys.LICENSE}'] == 'test'
    sources = list(record[policy_keys.CONTROL_SOURCE].values())
    assert sources == [policy_keys.POLICY_SOURCE, policy_keys.TELEOPERATOR]
    assert record[eval_keys.SUCCESS] is True


@pytest.mark.timeout(600.0)
def test_round_1_runs_the_checkpoint_that_trained_on_the_training_data_of_round_0(rounds):
    trained_on = [{'round': 0, 'record_uid': _record(rounds, 0).meta[META_UID]}]
    record = _record(rounds, 1)

    assert _checkpoint(rounds, 0) == stub_model.checkpoint_id(trained_on)
    assert record[f'{policy_keys.SERVER_META}.{policy_keys.TYPE}'] == stub_model.WIRE_NAME
    assert record[f'{policy_keys.SERVER_META}.{policy_keys.CHECKPOINT_ID}'] == _checkpoint(rounds, 0)


@pytest.mark.timeout(600.0)
def test_the_training_data_of_round_1_holds_the_teleoperator_runs_of_round_0_and_the_corrections_of_round_1(rounds):
    first, second = _record(rounds, 0), _record(rounds, 1)
    corrections = list(second[policy_keys.CONTROL_SOURCE].values()).count(policy_keys.TELEOPERATOR)
    [round_0] = LocalDataset(rounds / 'round_0' / 'training_data')
    training = list(LocalDataset(rounds / 'round_1' / 'training_data'))

    assert corrections > 1
    listed = [(episode.meta[META_ROUND], episode.meta[META_RECORD_UID]) for episode in training]
    assert listed == [(0, first.meta[META_UID])] + [(1, second.meta[META_UID])] * corrections
    for label in _LABELS:
        np.testing.assert_array_equal(list(training[0][label].values()), list(round_0[label].values()))
    for episode in training[1:]:
        # While the expert drives, the policy of the checkpoint runs in shadow, and its actions stay beside the labels.
        assert len(episode[_LABELS[0]]) > 0
        assert len(episode[_SHADOW]) > 0
    converted = stub_model.episodes(rounds / 'round_1' / 'converted')
    assert [(each['round'], each['record_uid']) for each in converted] == listed
    assert _checkpoint(rounds, 1) == stub_model.checkpoint_id(converted)


def test_the_robolab_config_builds_the_rig_the_tasks_with_a_layout_each_and_the_expert_of_banana_in_bowl():
    config = dagger.robolab_banana_in_bowl.override(
        output_dir='unused', round_0_license='test', **{'rig.teleoperator.license': 'test'}
    )
    built = cfn.Config(dict, **config.kwargs).instantiate()

    assert isinstance(built['rig'], EnvServerRig)
    observations = set(built['rig'].observations)
    for arm in keys.BIMANUAL_ARMS:
        assert keys.arm_channel(keys.ROBOT_STATE, arm) in observations
        assert keys.arm_channel(keys.GRIP, arm) in observations
    assert {keys.EXTERIOR_IMAGE, keys.WRIST_LEFT_IMAGE, keys.WRIST_RIGHT_IMAGE} <= observations
    first, second = islice(built['tasks'], 2)
    scenes = [task.prepare_args[eval_keys.SCENE] for task in (first, second)]
    assert [scene[eval_keys.TASK] for scene in scenes] == ['BananaInBowlTask'] * 2
    assert [scene[robolab_keys.CAMERAS] for scene in scenes] == [robolab_keys.TOP_AND_WRISTS] * 2
    assert [task.meta[eval_keys.SEED] for task in (first, second)] == [0, 1]
    assert scenes[0][robolab_keys.STATE] == layout(np.random.default_rng(0), 0.05, np.radians(30.0))
    assert scenes[0][robolab_keys.STATE] != scenes[1][robolab_keys.STATE]
    # The env server starts at the first request of the proxy, so the harness of the rig starts no RoboLab here.
    with built['rig'].harness(takes_over_at_start=True) as (_, teleoperator):
        pass
    assert isinstance(teleoperator, BananaInBowlExpert)
    meta = teleoperator.meta()
    assert (meta[policy_keys.TYPE], meta[policy_keys.LICENSE]) == (BananaInBowlExpert.WIRE_NAME, 'test')
    for statics in teleoperator.arms.values():
        assert (statics.reach, statics.rest_yaw, statics.wrist_turn) == (0.48, np.pi / 2, np.radians(100.0))
        assert np.array_equal(statics.park, dagger.YAM_PARK)
    # positronic's LeRobot ACT has no two-arm recipe, so the config binds no model.
    assert 'model' not in built
    assert (built['rounds'], built['episodes']) == (3, 20)
