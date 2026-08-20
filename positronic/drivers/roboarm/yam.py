"""Driver for the real i2rt YAM arm — one CAN chain carrying six joints plus the gripper.

i2rt exposes joint-space position-PD with gravity compensation only (its own ~100 Hz control thread), so this
driver solves FK/IK itself against the vendored MJCF (``assets/mujoco/i2rt_yam/yam.xml``) at ``DEFAULT_FRAME`` —
the control frame the training data is expressed in. The upstream MJCF package is vendored whole
(``scene.xml`` and meshes included); the driver itself loads only ``yam.xml``. The gripper is the chain's 7th
DOF, normalized 0=closed/1=open — the inverse of positronic's grip convention — so grip values are inverted in
both directions.

Station bring-up is not verifiable off-hardware and must be re-checked on the rig: CAN interface up
(``ip link set can0 up type can bitrate 1000000``), motor zero calibration, kp/kd gains, physical gripper
polarity and joint-range check, mount pose survey (``base_pose``), teleop latency, and the chain going limp
on close (``zero_torque_mode``).
"""

import logging
from collections.abc import Callable, Generator, Iterator
from typing import Any

import mujoco as mj
import numpy as np

import pimm
from positronic import geom, keys
from positronic.drivers import vendor_import
from positronic.drivers.utils import DriverRun, MoveStatus
from positronic.utils import package_assets_path

from . import RobotStatus, State, command
from .ik import qpos_from_site_pose
from .models import DEFAULT_FRAME

# i2rt lives in the `yam` extra, which the type-check environment does not install.
with vendor_import('i2rt', 'YAM support', hint='Re-run with the yam extra:\n  uv run --locked --extra yam ...\n'):
    from i2rt.robots.get_robot import get_yam_robot  # pyright: ignore[reportMissingImports]
    from i2rt.robots.utils import GripperType  # pyright: ignore[reportMissingImports]

# The driver solves FK/IK itself, so its joint order and control frame must match the YAM sim's.
# TODO(#517): centralise driver kinematics so driver and sim share one module.
_JOINT_NAMES = ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')
# The menagerie "home" keyframe: arm folded up and back, out of the workspace.
HOME_Q = np.array([0.0, 1.047, 1.047, 0.0, 0.0, 0.0])

_MJCF_PATH = 'assets/mujoco/i2rt_yam/yam.xml'
_IK_POS_TOL = 1e-3  # meters; FK-verify acceptance for an IK solution after limit clamping
_IK_ROT_TOL = 1e-2  # radians
_MOVE_TIME_S = 2.0  # seconds the commanded position is ramped over on the way to a target
_SETTLE_S = 1.0  # seconds the chain is given to reach the last waypoint before the move gives up
_ARRIVED_TOL = 0.02  # radians; the chain has no goal to report, so arrival is judged from the joints it reads
_GRIP_ARRIVED_TOL = 0.05  # normalized; the fingers report width, so arrival is judged from that reading
# The vendor's observation contract
_JOINT_POS, _JOINT_VEL, _GRIPPER_POS = 'joint_pos', 'joint_vel', 'gripper_pos'


def _reach_postures(x: float, y: float) -> list[np.ndarray]:
    """IK warm-start candidates for reaching toward arm-base-frame point (x, y): joint1 swung to the target's
    azimuth, elbow folded down at two heights. The 6-DoF wrist gives LM no null space to escape bad basins,
    so seeding near the goal is what makes limit-clamped IK reliable."""
    az = np.arctan2(y, x)
    return [np.array([az, 1.8, 2.2, 0.0, -0.9, 0.0]), np.array([az, 1.2, 1.2, 0.0, 0.6, 0.0])]


def _connect(channel: str, sim: bool):
    """Open the i2rt chain in position-PD mode; ``sim=True`` runs i2rt's own MuJoCo sim instead of hardware."""
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

    def _set_moving(self):
        self.array[YamState.STATUS_OFFSET] = RobotStatus.MOVING.value

    def _set_error(self):
        self.array[YamState.STATUS_OFFSET] = RobotStatus.ERROR.value

    def encode(self, q: np.ndarray, dq: np.ndarray, ee_pose: geom.Transform3D):
        self.array[YamState.Q_OFFSET : YamState.Q_OFFSET + 6] = q
        self.array[YamState.DQ_OFFSET : YamState.DQ_OFFSET + 6] = dq
        self.array[YamState.EE_POSE_OFFSET : YamState.EE_POSE_OFFSET + 3] = ee_pose.translation
        self.array[YamState.EE_POSE_OFFSET + 3 : YamState.EE_POSE_OFFSET + 7] = ee_pose.rotation.as_quat
        self.array[YamState.STATUS_OFFSET] = RobotStatus.AVAILABLE.value


class _Kinematics:
    """FK/IK on the vendored YAM MJCF at ``DEFAULT_FRAME``, in the arm-base frame.

    ``mujoco`` exports every symbol below from a compiled extension, so a type checker cannot see them.
    """

    def __init__(self):
        model_path = package_assets_path(_MJCF_PATH)
        self._model = mj.MjModel.from_xml_path(model_path)
        self._data = mj.MjData(self._model)
        site = mj.mjtObj.mjOBJ_SITE
        self._site_id = mj.mj_name2id(self._model, site, DEFAULT_FRAME)
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


class _Chain(DriverRun):
    """The chain the driver drives: the vendor handle, the state published from it, and the moves made with it."""

    def __init__(
        self,
        vendor: Any,
        out: pimm.SignalEmitter[YamState],
        home_joints: np.ndarray,
        base_pose: geom.Transform3D,
        should_stop: pimm.SignalReceiver,
        clock: pimm.Clock,
    ):
        super().__init__(should_stop, clock, hz=100)
        self.vendor = vendor
        self.out = out
        self.state = YamState()
        self.home_joints = home_joints
        self._base_pose = base_pose
        self._kin = _Kinematics()

    def observations(self) -> dict[str, np.ndarray]:
        """What the chain reports right now."""
        return self.vendor.get_observations()

    def encode(self, obs: dict[str, np.ndarray]) -> None:
        q = obs[_JOINT_POS]
        self.state.encode(q, obs[_JOINT_VEL], self._base_pose * self._kin.fk(q))

    def publish(self, obs: dict[str, np.ndarray]) -> None:
        """Ship the chain as it reports itself, marked ERROR while it is not where the driver put it."""
        self.encode(obs)
        if self.errored:  # ``encode`` clears the status; the chain is still where it was left
            self.state._set_error()
        self.out.emit(self.state)

    def measured_joints(self) -> np.ndarray:
        """The joints the chain reports, as a target that holds it where it stands."""
        return np.asarray(self.observations()[_JOINT_POS], dtype=np.float64)

    def _ik(self, world_pose: geom.Transform3D, q: np.ndarray) -> np.ndarray:
        """IK in the arm-base frame."""
        solution = self._kin.ik(self._base_pose.inv * world_pose, q)
        if solution is None:
            raise ValueError(f'{world_pose} is out of reach')
        return solution

    def target_joints(self, cmd: command.CommandType, q: np.ndarray) -> np.ndarray:
        """The joints ``cmd`` asks the chain to hold; raises what the chain cannot be asked for."""
        match cmd:
            case command.JointPosition(positions):
                return np.asarray(positions, dtype=np.float64)
            case command.JointDelta(velocities=delta):
                return q + np.asarray(delta, dtype=np.float64)
            case command.CartesianPosition(pose):
                return self._ik(pose, q)
            case command.CartesianDelta() as delta_cmd:
                return self._ik(delta_cmd.apply(self._base_pose * self._kin.fk(q)), q)
            case other:
                raise NotImplementedError(f'Unsupported command {other}')

    def _publish_moving(self) -> None:
        """Publish the chain mid-move, where its pose is not the one it was commanded to track."""
        self.encode(self.observations())
        self.state._set_moving()  # ``encode`` clears the status; the chain has not arrived
        self.out.emit(self.state)

    def _arrived(self, target: np.ndarray, grip: float) -> bool:
        """Whether the chain is where it was sent. The fingers count as much as the joints: a caller told a
        move landed may act on the whole chain, gripper included."""
        obs = self.observations()
        return bool(np.all(np.abs(obs[_JOINT_POS] - target) < _ARRIVED_TOL)) and (
            abs((1.0 - float(obs[_GRIPPER_POS][0])) - grip) < _GRIP_ARRIVED_TOL
        )

    def move_to(self, target: np.ndarray, grip: float) -> Generator[pimm.Command, None, MoveStatus]:
        """Ramp the chain to ``target``, yielding until it reads back there.

        Drive with ``yield from``. The vendor's own ``move_joints`` is not used: it blocks the world for the
        whole ramp and returns without ever reading where the chain got to.
        """
        try:
            start = np.asarray(self.observations()[_JOINT_POS], dtype=np.float64)
            started = self.clock.now()
            while not self._arrived(target, grip):
                if self.should_stop.value:
                    return MoveStatus.GAVE_UP
                elapsed = self.clock.now() - started
                if elapsed > _MOVE_TIME_S + _SETTLE_S:
                    raise TimeoutError(f'the chain stopped short of {target} at grip {grip}')
                # Ramped rather than commanded outright, so the chain travels at a pace the joints can hold,
                # and held at the target afterwards while it settles the last of the way in.
                alpha = min(elapsed / _MOVE_TIME_S, 1.0)
                self.vendor.command_joint_pos(np.append((1 - alpha) * start + alpha * target, 1.0 - grip))
                self._publish_moving()
                yield self.limiter.wait()
        except Exception:
            self.errored = True
            raise

        self.errored = False
        self.encode(self.observations())
        self.out.emit(self.state)
        return MoveStatus.ARRIVED

    def home(self, grip: float) -> Generator[pimm.Command, None, np.ndarray]:
        """Ramp the chain home, and return the joints to hold: home if it got there, where it stopped if not."""
        home = np.asarray(self.home_joints, dtype=np.float64)
        try:
            if (yield from self.move_to(home, grip)) is MoveStatus.ARRIVED:
                return home
        # rules-allow: swallowed-error — a chain that will not home reads ERROR; it does not end the run
        except Exception as exc:
            logging.error(f'Homing failed, the chain is not where the driver put it: {exc}')
        return self.measured_joints()

    def serve_sync_move(
        self, call: pimm.calls.Call[command.CommandType, None], q: np.ndarray, grip: float
    ) -> Generator[pimm.Command, None, np.ndarray]:
        """Put the chain where ``call`` asks, answer it, and return the joints it now holds."""
        request = call.request
        with pimm.calls.forward_failure(call):
            target = self.home_joints if isinstance(request, command.Reset) else self.target_joints(request, q)
            if (yield from self.move_to(target, grip)) is MoveStatus.ARRIVED:
                call.set_result(None)
            return np.asarray(target, dtype=np.float64)
        # The move failed and its asker was told. ``q`` predates the ramp, so holding it would drive the chain
        # back the way it came; where it stopped is the only posture nobody has to be surprised by.
        return self.measured_joints()


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

        self.commands = pimm.ControlSystemReceiver[command.CommandType](self)
        self.sync_move = pimm.calls.ControlSystemHandler[command.CommandType, None](self)
        self.target_grip = pimm.ControlSystemReceiver[float](self)
        self.state = pimm.ControlSystemEmitter[YamState](self)
        self.grip = pimm.ControlSystemEmitter[float](self)
        self.robot_meta = pimm.ControlSystemEmitter[dict[str, Any]](self)

    def _chain(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> _Chain:
        """The chain this run drives, built from the driver's configuration.

        Built here, not in ``__init__``: a background control system is pickled before it runs, and a live
        CAN handle does not survive the trip.
        """
        vendor = self._connect(self._channel, self._sim)
        return _Chain(vendor, self.state, self._home_joints, self._base_pose, should_stop, clock)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        chain = self._chain(should_stop, clock)
        try:
            meta = {'robot': 'i2rt_yam', keys.JOINT_NAMES: list(_JOINT_NAMES), keys.CONTROL_FRAME: DEFAULT_FRAME}
            self.robot_meta.emit(meta)

            grip_target = 0.0
            q_target = yield from chain.home(grip_target)

            while not should_stop.value:
                if (grip := pimm.value_updated(self.target_grip)) is not None:
                    grip_target = float(grip)

                q = chain.observations()[_JOINT_POS]
                if (call := next(self.sync_move.incoming(), None)) is not None:
                    if isinstance(call.request, command.Reset):
                        grip_target = 0.0  # homing opens the gripper, as a ``Reset`` command does
                    q_target = yield from chain.serve_sync_move(call, q, grip_target)
                elif (cmd := pimm.value_updated(self.commands)) is not None:
                    try:
                        if isinstance(cmd, command.Reset):
                            grip_target = 0.0
                            q_target = yield from chain.home(grip_target)
                        else:
                            q_target = chain.target_joints(cmd, q)
                    # rules-allow: swallowed-error — a command stream cannot end the run; the next supersedes
                    except Exception as exc:
                        logging.warning(f'{cmd} not applied: {exc}')

                chain.vendor.command_joint_pos(np.append(q_target, 1.0 - grip_target))

                # Read afresh: a move above ran for seconds, so the reading taken before it is long stale.
                obs = chain.observations()
                chain.publish(obs)
                self.grip.emit(1.0 - float(obs[_GRIPPER_POS][0]))
                yield chain.limiter.wait()
        finally:
            chain.vendor.zero_torque_mode()
            chain.vendor.close()


class _FakeYam:
    """First-order-lag echo of the 7-DOF chain (6 joints + normalized gripper, 0=closed/1=open).

    Duck-types the slice of the runtime-checkable ``i2rt.robots.robot.Robot`` protocol the driver uses, so
    the ``--fake`` smoke runs without hardware.
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
            _JOINT_POS: self._pos[:6].copy(),
            _JOINT_VEL: self._vel[:6].copy(),
            _GRIPPER_POS: self._pos[6:7].copy(),
            'gripper_vel': self._vel[6:7].copy(),
        }

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self.last_command = np.asarray(joint_pos, dtype=np.float64).copy()
        step = self._alpha * (self.last_command - self._pos)
        self._vel = step * 100.0  # commands arrive at the driver's 100 Hz
        self._pos = self._pos + step

    def zero_torque_mode(self) -> None:
        self._vel = np.zeros(7)

    def close(self) -> None:
        pass


if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser(description='YAM driver smoke: drives a Cartesian square and checks round-trips.')
    parser.add_argument('--channel', default='can0')
    parser.add_argument('--fake', action='store_true', help='in-process first-order-lag echo; needs no hardware')
    parser.add_argument('--sim', action='store_true', help="i2rt's own MuJoCo sim instead of the CAN chain")
    args = parser.parse_args()

    fake = _FakeYam() if args.fake else None
    robot = Robot(args.channel, sim=args.sim, connect=(lambda channel, sim: fake) if args.fake else _connect)

    with pimm.World() as world:
        # `World.pair` cannot express that it returns the counterpart of the port it is given, so the four
        # payload types are named here.
        commands = world.pair(robot.commands)
        sync_move = world.pair(robot.sync_move)
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
        while state.read() is None or state.value.status == RobotStatus.MOVING:
            pump(0.1)  # the opening move ramps the chain home over a couple of seconds
        assert state.value.status == RobotStatus.AVAILABLE, state.value.status

        kin = _Kinematics()

        if fake is not None:
            # State round-trip: the homed chain comes back through the driver's FK.
            assert np.allclose(state.value.q, HOME_Q, atol=_ARRIVED_TOL), state.value.q
            home_err = np.linalg.norm(state.value.ee_pose.translation - kin.fk(HOME_Q).translation)
            assert home_err < 0.02, home_err  # the chain arrives within `_ARRIVED_TOL` of home, not onto it

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

        reach_q = np.array([0.0, 1.2, 1.2, 0.0, 0.6, 0.0])
        answer = sync_move(command.JointPosition(reach_q))
        for _ in range(100):
            if answer.done():
                break
            pump(0.1)
        answer.result()
        if fake is not None:
            assert np.allclose(state.value.q, reach_q, atol=_ARRIVED_TOL), state.value.q

        # Then a Cartesian square through the driver's IK, commanded rather than asked for. The square sits
        # well inside the reach envelope, at the unfolded posture's wrist orientation.

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
