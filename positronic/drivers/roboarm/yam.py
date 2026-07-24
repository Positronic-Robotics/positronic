"""Driver for the real i2rt YAM arm — one CAN chain carrying six joints plus the gripper.

i2rt exposes joint-space position-PD with gravity compensation only (its own ~100 Hz control thread), so this
driver solves FK/IK itself against the vendored MJCF (``assets/mujoco/i2rt_yam/yam.xml``) at ``grasp_site`` —
the control frame the training data is expressed in. The gripper is the chain's 7th DOF, normalized
0=closed/1=open — the inverse of positronic's grip convention — so grip values are inverted in both directions.

Station bring-up is not verifiable off-hardware and must be re-checked on the rig: CAN interface up
(``ip link set can0 up type can bitrate 1000000``), motor zero calibration, kp/kd gains, physical gripper
polarity and joint-range check, mount pose survey (``base_pose``), teleop latency, and the chain going limp
on close (``zero_torque_mode``).
"""

import logging
from collections.abc import Callable, Iterator
from typing import Any

import mujoco as mj
import numpy as np

try:
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType
except ImportError:
    # i2rt is an optional extra (`pip install "positronic[yam]"`); fake and kinematics-only paths must import
    # without it, so degrade to None here and raise from ``_connect`` only when a real arm is requested.
    get_yam_robot = GripperType = None

import pimm
from positronic import geom
from positronic.utils import package_assets_path

from . import RobotStatus, State, command
from .ik import qpos_from_site_pose

# Duplicated from the yam-bimanual branch's simulator/mujoco/yam.py so driver and sim agree on frames without
# a code dependency. TODO: converge on one module when the YAM sim lands on this branch.
_JOINT_NAMES = ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')
# The menagerie "home" keyframe: arm folded up and back, out of the workspace.
HOME_Q = np.array([0.0, 1.047, 1.047, 0.0, 0.0, 0.0])
# Sim world-frame mount geometry the training data uses: table top z=0.30, arms at (0.30, ±0.305) facing +x,
# base plate lifted 0.011 above the tabletop.
YAM_MOUNTS = {'left': (0.30, 0.305), 'right': (0.30, -0.305)}
TABLE_Z = 0.30
YAM_MOUNT_LIFT = 0.011

_MJCF_PATH = 'assets/mujoco/i2rt_yam/yam.xml'
_IK_POS_TOL = 1e-3  # meters; FK-verify acceptance for an IK solution after limit clamping
_IK_ROT_TOL = 1e-2  # radians


def _reach_postures(x: float, y: float) -> list[np.ndarray]:
    """IK warm-start candidates for reaching toward arm-base-frame point (x, y): joint1 swung to the target's
    azimuth, elbow folded down at two heights. The 6-DoF wrist gives LM no null space to escape bad basins,
    so seeding near the goal is what makes limit-clamped IK reliable."""
    az = np.arctan2(y, x)
    return [np.array([az, 1.8, 2.2, 0.0, -0.9, 0.0]), np.array([az, 1.2, 1.2, 0.0, 0.6, 0.0])]


def _connect(channel: str, sim: bool):
    """Open the i2rt chain in position-PD mode; ``sim=True`` runs i2rt's own MuJoCo sim instead of hardware."""
    if get_yam_robot is None or GripperType is None:
        raise ImportError('YAM support is not installed. Install the yam extra:\n  pip install "positronic[yam]"\n')
    return get_yam_robot(channel, gripper_type=GripperType.LINEAR_4310, zero_gravity_mode=False, sim=sim)


class YamState(State, pimm.shared_memory.NumpySMAdapter):
    Q_OFFSET = 0
    DQ_OFFSET = Q_OFFSET + 6
    EE_POSE_OFFSET = DQ_OFFSET + 6
    STATUS_OFFSET = EE_POSE_OFFSET + 7
    TOTAL = STATUS_OFFSET + 1

    def __init__(self):
        super().__init__(shape=(YamState.TOTAL,), dtype=np.dtype(np.float32))

    def instantiation_params(self) -> tuple[Any, ...]:
        return ()

    @property
    def q(self) -> np.ndarray:
        return self.array[YamState.Q_OFFSET : YamState.Q_OFFSET + 6].copy()

    @property
    def dq(self) -> np.ndarray:
        return self.array[YamState.DQ_OFFSET : YamState.DQ_OFFSET + 6].copy()

    @property
    def ee_pose(self) -> geom.Transform3D:
        pose = self.array[YamState.EE_POSE_OFFSET : YamState.EE_POSE_OFFSET + 7].copy()
        return geom.Transform3D(pose[:3], geom.Rotation.from_quat(pose[3:7]))

    @property
    def status(self) -> RobotStatus:
        return RobotStatus(int(self.array[YamState.STATUS_OFFSET]))

    def _start_reset(self):
        self.array[YamState.STATUS_OFFSET] = RobotStatus.RESETTING.value

    def encode(self, q: np.ndarray, dq: np.ndarray, ee_pose: geom.Transform3D):
        self.array[YamState.Q_OFFSET : YamState.Q_OFFSET + 6] = q
        self.array[YamState.DQ_OFFSET : YamState.DQ_OFFSET + 6] = dq
        self.array[YamState.EE_POSE_OFFSET : YamState.EE_POSE_OFFSET + 3] = ee_pose.translation
        self.array[YamState.EE_POSE_OFFSET + 3 : YamState.EE_POSE_OFFSET + 7] = ee_pose.rotation.as_quat
        self.array[YamState.STATUS_OFFSET] = RobotStatus.AVAILABLE.value


class _Kinematics:
    """FK/IK on the vendored YAM MJCF at ``grasp_site``, in the arm-base frame."""

    def __init__(self):
        self._model = mj.MjModel.from_xml_path(package_assets_path(_MJCF_PATH))
        self._data = mj.MjData(self._model)
        self._site_id = mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_SITE, 'grasp_site')
        self._qpos_ids = np.array([self._model.joint(name).qposadr.item() for name in _JOINT_NAMES])
        self._dof_ids = np.array([self._model.joint(name).dofadr.item() for name in _JOINT_NAMES])
        ranges = np.array([self._model.joint(name).range for name in _JOINT_NAMES])
        self._lower, self._upper = ranges[:, 0], ranges[:, 1]

    def fk(self, q: np.ndarray) -> geom.Transform3D:
        self._data.qpos[self._qpos_ids] = q
        mj.mj_kinematics(self._model, self._data)
        quat = np.empty(4)
        mj.mju_mat2Quat(quat, self._data.site_xmat[self._site_id].copy())
        return geom.Transform3D(self._data.site_xpos[self._site_id].copy(), geom.Rotation.from_quat(quat))

    def ik(self, target: geom.Transform3D, current_q: np.ndarray) -> np.ndarray | None:
        """Multi-start LM IK: the live posture first, then the reach postures toward the target's azimuth.
        Solutions are wrapped and clamped into joint range, then FK-verified before acceptance."""
        for start in (current_q, *_reach_postures(*target.translation[:2])):
            self._data.qpos[:] = 0.0
            self._data.qpos[self._qpos_ids] = start
            qpos, _, success = qpos_from_site_pose(
                self._model,
                self._data,
                self._site_id,
                self._dof_ids,
                target.translation,
                target.rotation.as_quat,
                rot_weight=0.5,
            )
            if not success:
                continue
            q = qpos[self._qpos_ids].copy()
            # A revolute joint at q ± 2π is the same pose; wrap out-of-range entries back in when they fit.
            q = np.where(q > self._upper, q - 2 * np.pi, q)
            q = np.where(q < self._lower, q + 2 * np.pi, q)
            q = np.clip(q, self._lower, self._upper)
            reached = self.fk(q)
            rot_err = (reached.rotation.inv * target.rotation).angle
            rot_err = min(rot_err, 2 * np.pi - rot_err)
            if np.linalg.norm(reached.translation - target.translation) < _IK_POS_TOL and rot_err < _IK_ROT_TOL:
                return q
        return None


class Robot(pimm.ControlSystem):
    """Drives one YAM chain: FK/IK in the driver, joint-space position-PD on the arm.

    ``base_pose`` places the arm base in the world frame (identity = arm-base frame): IK targets are pulled
    back through it and the emitted ``ee_pose`` is pushed forward, so a bimanual embodiment can mount both
    arms in the training world frame. The gripper shares the CAN chain, so the arm driver carries the
    ``grip``/``target_grip`` ports (SO-101 precedent).
    """

    def __init__(
        self,
        channel: str = 'can0',
        *,
        home_joints: list[float] | None = None,
        base_pose: geom.Transform3D | None = None,
        sim: bool = False,
        connect: Callable = _connect,
    ) -> None:
        """
        :param channel: SocketCAN interface of the chain (e.g. ``can0``). Ignored in sim mode.
        :param home_joints: Joints of the "reset" position; defaults to ``HOME_Q``.
        :param base_pose: Arm-base mount pose in the world frame; None keeps everything in the arm-base frame.
        :param sim: Run against i2rt's own MuJoCo sim instead of hardware.
        :param connect: ``(channel, sim) -> i2rt Robot`` factory; the fake-mode smoke injects ``_FakeYam``.
        """
        self._channel = channel
        self._home_joints = np.asarray(home_joints if home_joints is not None else HOME_Q, dtype=np.float64)
        self._base_pose = base_pose if base_pose is not None else geom.Transform3D.identity
        self._sim = sim
        self._connect = connect

        self.commands: pimm.SignalReceiver[command.CommandType] = pimm.ControlSystemReceiver(self, default=None)
        self.target_grip: pimm.SignalReceiver[float] = pimm.ControlSystemReceiver(self, default=0.0)
        self.state: pimm.SignalEmitter[YamState] = pimm.ControlSystemEmitter(self)
        self.grip: pimm.SignalEmitter[float] = pimm.ControlSystemEmitter(self)
        self.robot_meta = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        arm = self._connect(self._channel, self._sim)
        try:
            kin = _Kinematics()
            meta = {'robot': 'i2rt_yam', 'joint_names': list(_JOINT_NAMES), 'control_frame': 'grasp_site'}
            self.robot_meta.emit(meta)

            robot_state = YamState()
            limiter = pimm.RateLimiter(clock, hz=100)
            player = command.TrajectoryPlayer(reduce=command.reduce)
            grip_player = command.TrajectoryPlayer()

            q_target = self._reset(arm, kin, robot_state)
            grip_target = 0.0

            while not should_stop.value:
                cmd_msg = self.commands.read()
                if cmd_msg.updated:
                    player.set(cmd_msg.data)
                grip_msg = self.target_grip.read()
                if grip_msg.updated:
                    grip_player.set(grip_msg.data)

                grip = grip_player.advance(clock.now_ns())
                if grip is not None:
                    grip_target = float(grip)

                obs = arm.get_observations()
                q = obs['joint_pos']

                cmd = player.advance(clock.now_ns())
                if cmd is not None:
                    match cmd:
                        case command.Reset():
                            q_target = self._reset(arm, kin, robot_state)
                            grip_target = 0.0
                            obs = arm.get_observations()
                            q = obs['joint_pos']
                        case command.JointPosition(positions):
                            q_target = np.asarray(positions, dtype=np.float64)
                        case command.JointDelta(velocities=delta):
                            q_target = q + np.asarray(delta, dtype=np.float64)
                        case command.CartesianPosition(pose):
                            q_target = self._ik_or_hold(kin, pose, q, q_target)
                        case command.CartesianDelta(delta):
                            target = command.apply_cartesian_delta(self._base_pose * kin.fk(q), delta)
                            q_target = self._ik_or_hold(kin, target, q, q_target)
                        case _:
                            raise NotImplementedError(f'Unsupported command {cmd}')

                arm.command_joint_pos(np.append(q_target, 1.0 - grip_target))

                robot_state.encode(q, obs['joint_vel'], self._base_pose * kin.fk(q))
                self.state.emit(robot_state)
                self.grip.emit(1.0 - float(obs['gripper_pos'][0]))
                yield limiter.wait()
        finally:
            arm.zero_torque_mode()
            arm.close()

    def _reset(self, arm, kin: _Kinematics, robot_state: YamState) -> np.ndarray:
        robot_state._start_reset()
        self.state.emit(robot_state)
        arm.move_joints(np.append(self._home_joints, 1.0), time_interval_s=2.0)  # chain gripper 1.0 = open
        obs = arm.get_observations()
        robot_state.encode(obs['joint_pos'], obs['joint_vel'], self._base_pose * kin.fk(obs['joint_pos']))
        self.state.emit(robot_state)
        return self._home_joints.copy()

    def _ik_or_hold(
        self, kin: _Kinematics, world_pose: geom.Transform3D, q: np.ndarray, hold: np.ndarray
    ) -> np.ndarray:
        """IK in the arm-base frame; on failure hold the previous joint target rather than jump."""
        solution = kin.ik(self._base_pose.inv * world_pose, q)
        if solution is None:
            logging.warning(f'IK failed for target {world_pose}, holding previous joint target')
            return hold
        return solution


class _FakeYam:
    """First-order-lag echo of the 7-DOF chain (6 joints + normalized gripper, 0=closed/1=open).

    Duck-types the slice of the runtime-checkable ``i2rt.robots.robot.Robot`` protocol the driver uses, so
    the ``--fake`` smoke runs without i2rt installed.
    """

    def __init__(self, alpha: float = 0.3):
        self._alpha = alpha
        self._pos = np.append(np.zeros(6), 1.0)  # the chain boots with the gripper open
        self._vel = np.zeros(7)
        self.last_command: np.ndarray | None = None

    def num_dofs(self) -> int:
        return 7

    def get_observations(self) -> dict[str, np.ndarray]:
        return {
            'joint_pos': self._pos[:6].copy(),
            'joint_vel': self._vel[:6].copy(),
            'gripper_pos': self._pos[6:7].copy(),
            'gripper_vel': self._vel[6:7].copy(),
        }

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self.last_command = np.asarray(joint_pos, dtype=np.float64).copy()
        step = self._alpha * (self.last_command - self._pos)
        self._vel = step * 100.0  # commands arrive at the driver's 100 Hz
        self._pos = self._pos + step

    def move_joints(self, target_joint_positions: np.ndarray, time_interval_s: float = 2.0) -> None:
        self._pos = np.asarray(target_joint_positions, dtype=np.float64).copy()
        self._vel = np.zeros(7)

    def zero_torque_mode(self) -> None:
        self._vel = np.zeros(7)

    def close(self) -> None:
        pass


if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser(description='YAM driver smoke: drives a Cartesian square and checks round-trips.')
    parser.add_argument('--channel', default='can0')
    parser.add_argument('--fake', action='store_true', help='in-process first-order-lag echo; needs no i2rt/hardware')
    parser.add_argument('--sim', action='store_true', help="i2rt's own MuJoCo sim instead of the CAN chain")
    args = parser.parse_args()

    fake = _FakeYam() if args.fake else None
    robot = Robot(args.channel, sim=args.sim, connect=(lambda channel, sim: fake) if args.fake else _connect)

    with pimm.World() as world:
        commands = world.pair(robot.commands)
        target_grip = world.pair(robot.target_grip)
        state = world.pair(robot.state)
        grip = world.pair(robot.grip)

        loop = world.start([robot])

        def pump(seconds: float):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline and not world.should_stop:
                cmd = next(loop)
                time.sleep(cmd.seconds if isinstance(cmd, pimm.Sleep) else 0)

        pump(0.1)
        while state.read() is None:
            pump(0.1)
        assert state.value.status == RobotStatus.AVAILABLE, state.value.status

        kin = _Kinematics()

        if fake is not None:
            # State round-trip: the homed chain comes back through the driver's FK.
            assert np.allclose(state.value.q, HOME_Q, atol=1e-3), state.value.q
            home_err = np.linalg.norm(state.value.ee_pose.translation - kin.fk(HOME_Q).translation)
            assert home_err < 1e-4, home_err

            # Grip round-trip: polarity inverted on the way out (command) and on the way back (observation).
            target_grip.emit(0.8)
            pump(0.5)
            assert fake.last_command is not None
            assert abs(fake.last_command[6] - 0.2) < 1e-6, fake.last_command  # positronic 0.8 closed -> chain 0.2
            assert abs(grip.value - 0.8) < 0.02, grip.value
            target_grip.emit(0.0)
            pump(0.5)
            assert abs(fake.last_command[6] - 1.0) < 1e-6, fake.last_command
            assert abs(grip.value) < 0.02, grip.value

        # Unfold toward the workspace, then drive a Cartesian square through the driver's IK. The square
        # sits well inside the reach envelope, at the unfolded posture's wrist orientation.
        reach_q = np.array([0.0, 1.2, 1.2, 0.0, 0.6, 0.0])
        commands.emit(command.JointPosition(reach_q))
        pump(0.5)
        if fake is not None:
            assert np.allclose(state.value.q, reach_q, atol=0.02), state.value.q

        center = geom.Transform3D(np.array([0.30, 0.05, 0.20]), state.value.ee_pose.rotation)
        print(f'Square center: {center}')
        square = [(0.0, 0.05, 0.0), (0.0, 0.05, 0.05), (0.0, -0.05, 0.05), (0.0, -0.05, 0.0), (0.0, 0.0, 0.0)]
        for offset in square:
            target = geom.Transform3D(center.translation + np.asarray(offset), center.rotation)
            solution = kin.ik(target, state.value.q)
            assert solution is not None, f'IK failed for {target}'
            ik_err = np.linalg.norm(kin.fk(solution).translation - target.translation)
            assert ik_err < 5e-3, ik_err  # FK↔IK consistency
            commands.emit(command.CartesianPosition(target))
            pump(0.7)
            reached = np.linalg.norm(state.value.ee_pose.translation - target.translation)
            print(f'Moved to {target.translation}, error {reached * 1000:.2f} mm')
            if fake is not None:
                assert reached < 5e-3, reached

        print('YAM driver smoke passed')
