"""A joint delta holds its target through a gap between commands, as the arm's driver holds ``q + delta``."""

from typing import Any

import mujoco
import numpy as np
import pytest

import pimm
from positronic import geom, keys
from positronic.drivers.roboarm import command as roboarm_command
from positronic.simulator.env_server import arm_action, protocol
from positronic.simulator.env_server.adapter import WireCommandAdapter
from positronic.simulator.env_server.tests.mujoco_chain import hinge_chain_xml

JOINTS = 7
KP = 50.0  # Nm/rad
BIAS = 0.5  # Nm on every joint, which no friction holds
CONTROL_DT = 1 / 15


class _JointsAdapter(WireCommandAdapter):
    """The command side of a one-arm adapter whose observation is the joints."""

    def _reset_token(self, params: dict) -> None:
        return None

    def task_params(self, records):
        return []

    def observations(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        self._measure(geom.Transform3D.identity, raw_obs[protocol.OBS_JOINT_POS])
        return {}

    def privileged(self, raw_obs):
        return {}

    def terminal(self, result):
        return None


def _biased_arm() -> tuple[mujoco.MjModel, mujoco.MjData]:
    actuators = ''.join(f'<position joint="arm_joint{i}" kp="{KP}" kv="5"/>' for i in range(1, JOINTS + 1))
    xml = (
        '<mujoco><compiler angle="radian"/><option gravity="0 0 0" timestep="0.002"/>'
        '<default><joint armature="0.1"/></default>'
        f'<worldbody>{hinge_chain_xml("arm", 0.0, JOINTS)}</worldbody>'
        f'<actuator>{actuators}</actuator></mujoco>'
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qfrc_applied[:] = BIAS
    return model, data


def _no_ik(*_args):
    raise AssertionError('a joint command needs no pose')


def _tick(model, data, adapter: WireCommandAdapter, cmd: roboarm_command.CommandType | None) -> None:
    """One control period: the adapter's payload, the server's joint targets, then the physics."""
    adapter.observations({protocol.OBS_JOINT_POS: data.qpos.copy()})
    wire = adapter.action({keys.ROBOT_COMMAND: None if cmd is None else pimm.Message(cmd)})[keys.ROBOT_COMMAND]
    data.ctrl[:] = arm_action.arm_targets(wire, data.qpos.copy(), ik=_no_ik, read_eef=_no_ik).q
    for _ in range(round(CONTROL_DT / model.opt.timestep)):
        mujoco.mj_step(model, data)


@pytest.mark.timeout(60.0)
def test_the_joints_hold_through_a_gap_under_a_torque_the_target_does_not_balance():
    model, data = _biased_arm()
    adapter = _JointsAdapter()
    _tick(model, data, adapter, roboarm_command.JointDelta(np.full(JOINTS, 0.05)))
    for _ in range(5):
        _tick(model, data, adapter, None)
    settled = data.qpos.copy()
    for _ in range(30):
        _tick(model, data, adapter, None)
    # Re-commanding the measured joints walks the arm by about 0.005 rad per period, 0.15 rad over 30 periods.
    np.testing.assert_allclose(data.qpos, settled, atol=1e-3)


def test_a_joint_delta_is_held_as_its_target_at_the_measured_joints():
    adapter = _JointsAdapter()
    q = np.linspace(-0.3, 0.3, JOINTS)
    adapter.observations({protocol.OBS_JOINT_POS: q})
    first = adapter.action({keys.ROBOT_COMMAND: pimm.Message(roboarm_command.JointDelta(np.full(JOINTS, 0.1)))})
    adapter.observations({protocol.OBS_JOINT_POS: q + 0.02})
    later = adapter.action({keys.ROBOT_COMMAND: None})
    for action in (first, later):
        assert action[keys.ROBOT_COMMAND][protocol.COMMAND_TYPE] == protocol.JOINT_POS
        np.testing.assert_allclose(action[keys.ROBOT_COMMAND][protocol.COMMAND_JOINT_POS], q + 0.1)


def test_a_joint_delta_keeps_its_control_mode():
    adapter = _JointsAdapter()
    adapter.observations({protocol.OBS_JOINT_POS: np.zeros(JOINTS)})
    mode = roboarm_command.Impedance(kq=(40.0,) * 7, kqd=(4.0,) * 7, kx=(750.0,) * 6, kxd=(37.0,) * 6)
    cmd = roboarm_command.JointDelta(np.zeros(JOINTS), mode=mode)
    wire = adapter.action({keys.ROBOT_COMMAND: pimm.Message(cmd)})[keys.ROBOT_COMMAND]
    assert wire[protocol.COMMAND_MODE] == roboarm_command.to_wire(mode)


def test_a_joint_delta_before_any_measurement_is_refused():
    with pytest.raises(ValueError, match='measured joints'):
        _JointsAdapter().action({keys.ROBOT_COMMAND: pimm.Message(roboarm_command.JointDelta(np.zeros(JOINTS)))})


def test_a_joint_delta_that_names_other_joints_than_the_measured_is_refused():
    adapter = _JointsAdapter()
    adapter.observations({protocol.OBS_JOINT_POS: np.zeros(JOINTS)})
    with pytest.raises(ValueError, match='measured joints'):
        adapter.action({keys.ROBOT_COMMAND: pimm.Message(roboarm_command.JointDelta(np.zeros(JOINTS - 1)))})
