import numpy as np
import pytest
from eval_vocabulary.episode import GT_STATE_GRIPPER_CONTACT

from positronic import geom, keys
from positronic.cfg.simulator import lift_cube_loaders
from positronic.dagger.mujoco import MujocoRig
from positronic.dagger.rounds import RoundZeroPolicy
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.drivers.roboarm.command import CartesianPosition
from positronic.drivers.roboarm.models import ee_frames_entry
from positronic.eval import Task
from positronic.eval import keys as eval_keys
from positronic.policy import keys as policy_keys
from positronic.policy.base import Policy, PolicyRun, Runtime, Step
from positronic.policy.harness import Rollout
from positronic.simulator.expert import object_key
from positronic.simulator.mujoco.expert import FRAMES, LiftCubeExpert
from positronic.simulator.mujoco.lift_cube import CUBE_NAME
from positronic.utils import package_assets_path

MODEL = package_assets_path('assets/mujoco/franka_table.xml')
RISE = 0.01  # m
_TELEOPERATOR_META = f'{policy_keys.SOURCES}.{policy_keys.TELEOPERATOR}'


class _Rises(Policy):
    """A policy that commands the measured pose ``RISE`` higher, with the measured grip, as the stub model does."""

    def meta(self):
        return {policy_keys.TYPE: 'rises', policy_keys.LICENSE: 'test', **ee_frames_entry(FRAMES)}

    def run(self, runtime: Runtime) -> PolicyRun:
        obs = yield
        while True:
            pose = geom.Transform3D.from_vector(np.asarray(obs[keys.EE_POSE]), geom.Rotation.Representation.QUAT)
            up = geom.Transform3D(pose.translation + [0.0, 0.0, RISE], pose.rotation)
            commands = {keys.ROBOT_COMMAND: CartesianPosition(up), keys.TARGET_GRIP: float(obs[keys.GRIP])}
            obs = yield Step(commands, runtime.time_ns + 66_000_000)


def _collect(tmp_path, rig: MujocoRig, policy: Policy, *, takes_over_at_start: bool, timeout_sec: float):
    task = Task(
        'lift the cube', timeout_sec, prepare_args={eval_keys.SCENE: {eval_keys.SEED: 0}}, charge_inference_time=False
    )
    with rig.harness(takes_over_at_start=takes_over_at_start) as (perform, expert):
        answer = perform(Rollout(task, policy, LocalDatasetWriter(tmp_path), expert))
    [record] = LocalDataset(tmp_path)
    return answer, record


@pytest.fixture(scope='module')
def rig() -> MujocoRig:
    return MujocoRig(MODEL, lift_cube_loaders.instantiate(), 'test')


def test_the_rig_has_the_observations_and_the_commands_of_the_mujoco_franka(rig):
    assert set(rig.observations) == {keys.ROBOT_STATE, keys.GRIP}
    assert set(rig.commands) == {keys.ROBOT_COMMAND, keys.TARGET_GRIP}


@pytest.mark.timeout(120.0)
def test_in_round_0_the_expert_takes_over_at_its_second_element_and_lifts_the_cube(tmp_path, rig):
    answer, record = _collect(
        tmp_path, rig, RoundZeroPolicy('test', rig.commands), takes_over_at_start=True, timeout_sec=10.0
    )

    assert answer == {eval_keys.SUCCESS: True, eval_keys.TERMINATED: True}
    assert list(record[policy_keys.CONTROL_SOURCE].values()) == [policy_keys.POLICY_SOURCE, policy_keys.TELEOPERATOR]
    assert record[f'{_TELEOPERATOR_META}.{policy_keys.TYPE}'] == LiftCubeExpert.WIRE_NAME
    assert record[f'{_TELEOPERATOR_META}.{policy_keys.LICENSE}'] == 'test'
    assert any(record[object_key(CUBE_NAME, GT_STATE_GRIPPER_CONTACT)].values())


@pytest.mark.timeout(120.0)
def test_the_expert_takes_over_from_a_policy_that_leaves_its_path_and_hands_back(tmp_path, rig):
    _, record = _collect(tmp_path, rig, _Rises(), takes_over_at_start=False, timeout_sec=6.0)

    sources = list(record[policy_keys.CONTROL_SOURCE].values())
    assert sources.count(policy_keys.TELEOPERATOR) >= 2
    assert sources.count(policy_keys.POLICY_SOURCE) >= 2
    assert len(record[f'{policy_keys.SHADOW}.{policy_keys.POLICY_SOURCE}.{keys.TARGET_EE_POSE}']) > 0
