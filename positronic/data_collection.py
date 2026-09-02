import logging
import time
from collections.abc import Callable, Iterator, Sequence
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any

import configuronic as cfn
import numpy as np
import pos3

import pimm
import positronic.cfg.hardware.camera
import positronic.cfg.hardware.gripper
import positronic.cfg.hardware.roboarm
import positronic.cfg.simulator
import positronic.cfg.sound
import positronic.cfg.webxr
from pimm.logging import init_logging
from positronic import geom, keys, utils, wire
from positronic.dataset.ds_writer_agent import DsWriterAgent, DsWriterCommand, TimeMode
from positronic.dataset.local_dataset import LocalDatasetWriter
from positronic.dataset.serializers import Serializers
from positronic.drivers import roboarm
from positronic.drivers.keyboard import KeyboardControl
from positronic.drivers.roboarm import State as RoboarmState
from positronic.drivers.webxr import WebXR
from positronic.gui import dpg_ui
from positronic.simulator.mujoco.sim import MujocoSim
from positronic.simulator.mujoco.transforms import MujocoSceneTransform
from positronic.utils import package_assets_path
from positronic.utils.buttons import ButtonHandler


def _parse_buttons(buttons: dict, button_handler: ButtonHandler) -> set[str]:
    """Feed the controllers' buttons to ``button_handler``, and answer with the hands that sent any.

    A controller the headset has lost sends nothing at all, and the handler raises when asked for a button
    it has never seen — so the hands named here are exactly the ones a caller may ask about.
    """
    hands = set()
    for side in ['left', 'right']:
        if buttons[side] is None:
            continue

        hands.add(side)
        mapping = {
            f'{side}_A': buttons[side][4],
            f'{side}_B': buttons[side][5],
            f'{side}_trigger': buttons[side][0],
            f'{side}_thumb': buttons[side][1],
            f'{side}_stick': buttons[side][3],
        }
        button_handler.update_buttons(mapping)
    return hands


def _check_error(is_error, was_error):
    return is_error, is_error and not was_error


# Where the hand ends and its shake begins. Measured over three minutes of teleoperation: 84% of the
# controller's movement sits below 1 Hz, 2% between 8 and 15 Hz — the tremor a hand always carries — and
# 8% above 15 Hz, which no hand does and the pose stream brings on its own. A cut here keeps 88% of the
# movement, and costs the operator 32 ms of lag.
_HAND_CUTOFF_HZ = 5.0


class _Tracker:
    on = False
    _offset = geom.Transform3D()
    _teleop_t = geom.Transform3D()

    def __init__(self, operator_position: geom.Transform3D | None):
        self._operator_position = operator_position
        self.on = self.umi_mode
        self._steady: geom.Transform3D | None = None
        self._steady_at = 0

    @property
    def umi_mode(self):
        return self._operator_position is None

    def turn_on(self, robot_pos: geom.Transform3D):
        if self.umi_mode:
            logging.info('Ignoring tracking on/off in UMI mode')
            return

        self.on = True
        logging.info('Starting tracking')
        self._offset = geom.Transform3D(
            -self._teleop_t.translation + robot_pos.translation, self._teleop_t.rotation.inv * robot_pos.rotation
        )

    def turn_off(self):
        if self.umi_mode:
            logging.info('Ignoring tracking on/off in UMI mode')
            return
        self.on = False
        logging.info('Stopped tracking')

    def _steadied(self, pose: geom.Transform3D, ts_ns: int) -> geom.Transform3D:
        """``pose`` with the hand's own shake taken out, as a first-order lag on both halves."""
        if self._steady is None:
            self._steady, self._steady_at = pose, ts_ns
            return pose
        step = max(ts_ns - self._steady_at, 0) / 1e9
        self._steady_at = ts_ns
        share = step / (step + 1.0 / (2.0 * np.pi * _HAND_CUTOFF_HZ))
        turn = (self._steady.rotation.inv * pose.rotation).as_rotvec
        self._steady = geom.Transform3D(
            self._steady.translation + share * (pose.translation - self._steady.translation),
            self._steady.rotation * geom.Rotation.from_rotvec(share * turn),
        )
        return self._steady

    def update(self, tracker_pos: geom.Transform3D, ts_ns: int):
        if self.umi_mode:
            return tracker_pos

        steady = self._steadied(tracker_pos, ts_ns)
        self._teleop_t = self._operator_position * steady * self._operator_position.inv
        return geom.Transform3D(
            self._teleop_t.translation + self._offset.translation, self._teleop_t.rotation * self._offset.rotation
        )


class Teleop(Enum):
    """What the operator moves to drive the arm."""

    # A hand tracked in space. The arm follows the pose of the controller in that hand, and the operator
    # starts and stops the tracking with that controller's own `A`.
    HAND = 'hand'
    # A leader arm the operator holds. The follower stands at the joints the leader reads — nothing is
    # solved in between — and starts following once the two arms have met.
    LEADER = 'leader'


# How closely the leader and the follower must stand before the follower starts copying it, per joint. The
# follower takes up whatever is left as one streamed step, so this is the largest jump engaging can make.
_MEET_RAD = 0.1


class _Follow:
    """The follower waiting for the leader it copies to come to it.

    Asking the operator to line the arms up and then press something takes their word for it, and a press
    with the arms apart moves the follower the whole way at once. This engages on the fact instead: the
    arms are together, or the follower is not following.
    """

    def __init__(self) -> None:
        self.on = False
        self._armed = False

    def arm(self) -> None:
        """Let the follower take up its leader again, once the session has put both where they stand."""
        self._armed = True

    def turn_off(self) -> None:
        if self.on:
            logging.info('The arm stopped following its leader')
        self.on = False
        self._armed = False

    def met(self, leader: np.ndarray, follower: np.ndarray) -> bool:
        """Whether the follower may copy the leader: it may once the session has armed it and every joint
        of the two is within ``_MEET_RAD`` of the other's, and goes on doing so until something turns it
        off. An arm the operator has not asked for holds still, however close the two stand."""
        if not self._armed:
            return False
        if not self.on and leader.shape == follower.shape and np.max(np.abs(leader - follower)) <= _MEET_RAD:
            self.on = True
            logging.info('The arm met its leader and follows it now')
        return self.on


class OperatorPosition(Enum):
    # map xyz -> zxy
    FRONT = geom.Transform3D(rotation=geom.Rotation.from_quat([0.5, 0.5, 0.5, 0.5]))
    # map xyz -> zxy + flip x and y
    BACK = geom.Transform3D(rotation=geom.Rotation.from_quat([-0.5, -0.5, 0.5, 0.5]))


class DataCollectionController(pimm.ControlSystem):
    def __init__(
        self,
        operator_position: geom.Transform3D | None,
        nominal_joints: Sequence[float] | np.ndarray,
        joints_spread: Sequence[float] | np.ndarray = (),
        park_joints: Sequence[float] | np.ndarray = (),
        *,
        teleop: Teleop = Teleop.HAND,
        output_path: Path | None = None,
        static_meta: dict | None = None,
        metadata_getter: Callable[[], dict] | None = None,
    ):
        self.operator_position = operator_position
        self.teleop = teleop
        self._output_path = output_path
        self._nominal_joints = np.asarray(nominal_joints, dtype=np.float64)
        # A station that measured no jitter sends the arm exactly to its nominal.
        spread = joints_spread if len(joints_spread) else np.zeros_like(self._nominal_joints)
        self._joints_spread = np.asarray(spread, dtype=np.float64)
        self._park_joints = np.asarray(park_joints, dtype=np.float64)
        self._static_meta = static_meta or {}
        self.metadata_getter = metadata_getter or (lambda: {})
        self.controller_positions = pimm.DefaultingReceiver(self, default={})
        # An arm driven by a leader has no controller to press, and the session runs off `session_events`
        # instead — so no buttons is a rig without them, not a rig whose operator has not pressed yet.
        self.buttons_receiver = pimm.DefaultingReceiver(self, default={'left': None, 'right': None})
        # A keyboard carries these on a rig whose operator holds the leader.
        self.session_events = pimm.ControlSystemReceiver[SessionEvent](self)
        # What the leader publishes, on a rig driven by one.
        self.leader_joints = pimm.ControlSystemReceiver[np.ndarray](self)
        self.leader_grip = pimm.ControlSystemReceiver[float](self)
        self.robot_state = pimm.ControlSystemReceiver(self)
        self.gripper_state = pimm.FakeReceiver(self)  # To make compatible with other "policy" control systems
        self.frames = pimm.ReceiverDict(self, fake=True)
        self.robot_meta_in = pimm.DefaultingReceiver(self, default={})

        self.robot_commands = pimm.ControlSystemEmitter(self)
        self.sync_move = pimm.calls.ControlSystemCaller[roboarm.command.CommandType, None](self)
        # The leader travels to the poses the follower is sent to, so the two arms stand together when the
        # operator takes over. A leader left behind hands the follower the whole gap as its first step.
        self.leader_move = pimm.calls.ControlSystemCaller[roboarm.command.CommandType, None](self)
        self.redraw_scene = pimm.calls.ControlSystemCaller[Any, None](self)
        self.target_grip = pimm.ControlSystemEmitter(self)

        self.ds_agent_commands = pimm.ControlSystemEmitter(self)
        self.sound = pimm.ControlSystemEmitter(self)

    @property
    def _umi(self) -> bool:
        """Whether a hand's poses reach the arm as they are, with no tracking to start or stop and no
        start pose to put anything back at. A leader is never that: it is an arm, and it has one."""
        return self.teleop is Teleop.HAND and self.operator_position is None

    def _travel(
        self, target: roboarm.command.CommandType, should_stop: pimm.SignalReceiver, asks: Sequence[Any] = ()
    ) -> Iterator[pimm.Sleep]:
        """Take every arm of the rig to ``target``, yielding until it and everything in ``asks`` is done.

        Raises whatever a device failed on: a call hands its handler's exception back, so the vocabulary is
        the driver's — a move abandoned, a target it cannot hold, a fault from the vendor.
        """
        asks = list(asks)
        for move in (self.sync_move, self.leader_move):
            if move.connected:  # a rig without the arm has none to put anywhere
                asks.append(move(target))
        ready = pimm.calls.all_of(asks)
        while not ready.done():
            if should_stop.value:
                return
            self._drop_readings()
            yield pimm.Sleep(0.001)
        ready.result()

    def _drop_readings(self) -> None:
        """Drop what reached the rig while it travels: those readings are of a rig that has since moved.

        A reading taken before a move says where an arm stood then, and the tick after the move takes what
        it reads for the present. A follower would meet a leader that has gone, and be sent to the pose the
        two of them stood at. What the operator asks for in front of a travelling arm goes the same way:
        the press is impatience with this move, and holding it would run the move again.
        """
        for port in (
            self.session_events,
            self.controller_positions,
            self.robot_state,
            self.leader_joints,
            self.leader_grip,
        ):
            port.read()

    def _ready(self, should_stop: pimm.SignalReceiver) -> Iterator[pimm.Sleep]:
        """Redraw the scene and put every arm at a start pose drawn around the nominal joints, yielding
        until all of them are done."""
        logging.info('Readying the rig for the next episode')
        # A real scene is a person's to set up, and nothing here is asked.
        scene = [self.redraw_scene(None)] if self.redraw_scene.connected else []
        yield from self._travel(
            roboarm.command.sampled_joints(self._nominal_joints, self._joints_spread), should_stop, scene
        )

    def _park(self, should_stop: pimm.SignalReceiver) -> Iterator[pimm.Sleep]:
        """Put every arm of the rig where it rests, yielding until all of them are there."""
        if not len(self._park_joints):
            logging.warning('This rig names no pose to rest at, so there is nowhere to park it')
            return
        logging.info('Taking the rig to rest')
        yield from self._travel(roboarm.command.JointPosition(self._park_joints), should_stop)

    def _abandon(self, recording: bool, abort_wav_path: Path) -> bool:
        """Give up the recording that runs, if one does, and answer that none runs now."""
        if recording:
            self.ds_agent_commands.emit(DsWriterCommand.ABORT())
            self.sound.emit(abort_wav_path)
            logging.info('The recording was abandoned')
        return False

    def _from_hand(self, tracker: _Tracker, hands: set[str], buttons: ButtonHandler) -> tuple[Any, float | None]:
        """What the hand asks the arm for, and what it holds the grip at.

        The tracker is fed whether or not the arm is tracking: what the hand did while it was not is how
        far the arm would jump the moment it starts, and the offset is measured off it.
        """
        grip = buttons.get_value('right_trigger') if 'right' in hands else None
        cp_msg = self.controller_positions.read()
        if cp_msg.updated and cp_msg.data.get('right') is not None:
            pose = tracker.update(cp_msg.data['right'], cp_msg.ts)
            return roboarm.command.CartesianPosition(pose), grip
        return None, grip

    def _from_leader(self, follow: _Follow, state: pimm.Message[RoboarmState] | None) -> tuple[Any, float | None]:
        """What the leader asks its follower for, and what its trigger holds the grip at.

        The joints go over as they read: a leader and its follower are the same arm, so there is nothing
        to solve between them.
        """
        held = self.leader_grip.read()
        grip = float(held.data) if held is not None and held.updated else None
        joints = self.leader_joints.read()
        # The arms are judged to have met on a reading the leader has just sent. A reading that repeats
        # says where the leader was when it sent it, and the arm may have been driven somewhere since.
        if joints is None or state is None or not joints.updated:
            return None, grip
        leader = np.asarray(joints.data, dtype=np.float64)
        if not follow.met(leader, np.asarray(state.data.q, dtype=np.float64)):
            return None, grip
        return roboarm.command.JointPosition(leader), grip

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:  # noqa: C901
        sounds = Path(package_assets_path('assets/sounds'))
        start_wav_path = sounds / 'recording-has-started.wav'
        end_wav_path = sounds / 'recording-has-stopped.wav'
        abort_wav_path = sounds / 'recording-has-been-aborted.wav'
        error_wav_path = sounds / 'error-occurred.wav'

        source = _Tracker(self.operator_position) if self.teleop is Teleop.HAND else _Follow()
        button_handler = ButtonHandler()

        recording = False
        in_error = False

        while not should_stop.value:
            try:
                hands = _parse_buttons(self.buttons_receiver.value, button_handler)
                state = self.robot_state.read()
                asked = pimm.value_updated(self.session_events)
                if button_handler.just_pressed('right_B') or asked is SessionEvent.RECORD:
                    if not recording:
                        meta = dict(self._static_meta)
                        meta.update(self.robot_meta_in.value)
                        meta.update(self.metadata_getter())
                        self.ds_agent_commands.emit(DsWriterCommand.START(self._output_path, meta))
                        self.sound.emit(start_wav_path)
                        logging.info('The recording started')
                    else:
                        self.ds_agent_commands.emit(DsWriterCommand.STOP())
                        self.sound.emit(end_wav_path)
                        logging.info('The recording stopped')
                    recording = not recording
                elif isinstance(source, _Tracker) and button_handler.just_pressed('right_A'):
                    if source.on:
                        source.turn_off()
                    elif state is None:
                        logging.warning('The arm has not said where it stands, so there is nothing to follow from')
                    else:
                        source.turn_on(state.data.ee_pose)
                    logging.info('Tracking is %s', 'on' if source.on else 'off')
                elif (button_handler.just_pressed('right_stick') or asked is SessionEvent.READY) and not self._umi:
                    recording = self._abandon(recording, abort_wav_path)
                    source.turn_off()
                    try:
                        yield from self._ready(should_stop)
                        if isinstance(source, _Follow):
                            source.arm()
                    # rules-allow: swallowed-error — the operator hears it and asks again, session goes on
                    except Exception as e:
                        logging.error(f'The rig was not readied: {e}')
                        self.sound.emit(error_wav_path)
                elif asked is SessionEvent.PARK:
                    recording = self._abandon(recording, abort_wav_path)
                    source.turn_off()
                    try:
                        yield from self._park(should_stop)
                    # rules-allow: swallowed-error — the operator hears it and asks again, session goes on
                    except Exception as e:
                        logging.error(f'The rig was not parked: {e}')
                        self.sound.emit(error_wav_path)

                if isinstance(source, _Tracker):
                    cmd, grip = self._from_hand(source, hands, button_handler)
                else:
                    cmd, grip = self._from_leader(source, state)
                if grip is not None:
                    self.target_grip.emit(grip)

                if source.on and state is not None:
                    in_error, entered_error = _check_error(state.data.status == roboarm.RobotStatus.ERROR, in_error)
                    if entered_error:
                        logging.error('The arm is in error. It holds still until a reset clears the error.')
                        self.sound.emit(error_wav_path)
                    if not in_error and cmd is not None:
                        self.robot_commands.emit(cmd)

                yield pimm.Sleep(0.001)

            except pimm.NoValueException:
                yield pimm.Sleep(0.001)
                continue


class SessionEvent(Enum):
    """What the operator asks of the session itself, by whatever device is to hand."""

    # Start the recording, or stop the one that runs.
    RECORD = 'record'
    # Put the arm back at its start pose, and abandon the recording that runs.
    READY = 'ready'
    # Put the arm where it rests, and abandon the recording that runs.
    PARK = 'park'


# The key that asks for each. An operator holding the leader has no hand free for these, so a second
# person presses them; a pedal under the operator's own foot would ask for the same ones.
_SESSION_KEYS = {'r': SessionEvent.RECORD, ' ': SessionEvent.READY, 'h': SessionEvent.PARK}


def _session_event(key: str) -> SessionEvent | None:
    """What a keystroke asks of the session, or nothing for a key that asks for nothing."""
    return _SESSION_KEYS.get(key)


def controller_positions_serializer(controller_positions: dict[str, geom.Transform3D]) -> dict[str, np.ndarray]:
    res = {}
    for side, pos in controller_positions.items():
        if pos is not None:
            res[f'.{side}'] = Serializers.transform_3d(pos)
    return res


def _wrench_to_level(state: RoboarmState) -> float | None:
    if state.ee_wrench is None:
        return None
    return np.linalg.norm(state.ee_wrench)


def _wire(
    world: pimm.World,
    ds_agent: DsWriterAgent | None,
    data_collection: DataCollectionController,
    webxr: WebXR | None,
    robot_arm: pimm.ControlSystem | None,
    sound: pimm.ControlSystem | None,
    leader: pimm.ControlSystem | None = None,
    keyboard: KeyboardControl | None = None,
):
    if webxr is not None:
        world.connect(webxr.controller_positions, data_collection.controller_positions)
        world.connect(webxr.buttons, data_collection.buttons_receiver)

    if leader is not None:
        # A driver's ports are its own: ``pimm.ControlSystem`` declares none for the checker to find.
        world.connect(leader.joints, data_collection.leader_joints)  # pyright: ignore[reportAttributeAccessIssue]
        world.connect(leader.grip, data_collection.leader_grip)  # pyright: ignore[reportAttributeAccessIssue]
        world.connect(data_collection.leader_move, leader.sync_move)  # pyright: ignore[reportAttributeAccessIssue]

    if keyboard is not None:
        world.connect(
            keyboard.keyboard_inputs, data_collection.session_events, receiver_wrapper=pimm.map(_session_event)
        )

    if robot_arm is not None:
        # A driver's ports are its own: ``pimm.ControlSystem`` declares none for the checker to find.
        world.connect(data_collection.sync_move, robot_arm.sync_move)  # pyright: ignore[reportAttributeAccessIssue]

    if sound is not None:
        world.connect(data_collection.sound, sound.wav_path)
        if robot_arm is not None:
            world.connect(robot_arm.state, sound.level, receiver_wrapper=pimm.map(_wrench_to_level))

    if ds_agent is not None:
        # A leader's joints are not recorded beside these: the command the follower is given is those very
        # joints, and it is recorded already.
        if robot_arm is not None and webxr is not None:
            ds_agent.add_signal('controller_positions', controller_positions_serializer)
            world.connect(webxr.controller_positions, ds_agent.inputs['controller_positions'])
        world.connect(data_collection.ds_agent_commands, ds_agent.command)

    return ds_agent


def _frame_array(frame: pimm.shared_memory.NumpySMAdapter) -> np.ndarray:
    return frame.array


def _check_rig(
    robot_arm: pimm.ControlSystem | None,
    webxr: WebXR | None,
    leader: pimm.ControlSystem | None,
    nominal_joints: Sequence[float],
    joints_spread: Sequence[float],
    park_joints: Sequence[float],
) -> None:
    """Refuse a rig that cannot run, before any of it is started.

    Every one of these reaches the operator as a failed move or an arm that does nothing, long after the
    run began and with nothing to say which part of the configuration was wrong.
    """
    if (robot_arm is not None) != (len(nominal_joints) > 0):
        raise ValueError(
            '--robot_arm and --nominal_joints are named together or not at all: the right stick puts the arm '
            'a station has at the pose it measured, and either one alone leaves the other with nothing'
        )
    if len(park_joints) not in (0, len(nominal_joints)):
        raise ValueError(
            f'--park_joints names {len(park_joints)} joints and --nominal_joints names {len(nominal_joints)}: '
            'both are poses of the same arm, so they name the same joints'
        )
    if len(joints_spread) not in (0, len(nominal_joints)):
        raise ValueError(
            f'--joints_spread names {len(joints_spread)} joints and --nominal_joints names {len(nominal_joints)}: '
            'the spread is jitter measured per joint, so it carries one value for each, or none at all'
        )
    if not np.all(np.isfinite([*nominal_joints, *joints_spread, *park_joints])):
        raise ValueError(
            '--nominal_joints, --joints_spread and --park_joints name joint angles: every value has to be '
            'finite, or the draw between them raises instead of reaching the arm'
        )
    if leader is not None and robot_arm is None:
        raise ValueError('--leader is held to drive a follower, and one with nothing on the other end drives nothing')
    if leader is not None and webxr is not None:
        raise ValueError(
            'the arm is driven by a leader or by a hand tracked in space, not by both: --leader and --webxr '
            'would leave two things asking one arm to be in two places at once'
        )


def main(
    robot_arm: pimm.ControlSystem | None,
    gripper: pimm.ControlSystem | None,
    webxr: WebXR | None,
    sound: pimm.ControlSystem | None,
    cameras: dict[str, pimm.ControlSystem] | None,
    # The start pose the stick and the space key put the arm at: drawn around ``nominal_joints``, within
    # ``joints_spread``.
    nominal_joints: Sequence[float] = (),
    joints_spread: Sequence[float] = (),
    # The pose the `h` key takes every arm to, where it rests between sessions.
    park_joints: Sequence[float] = (),
    # The arm the operator holds. Naming it is what makes this a rig driven by a leader rather than by a
    # hand tracked in space.
    leader: pimm.ControlSystem | None = None,
    output_dir: str | None = None,
    stream_video_to_webxr: str | None = None,
    operator_position: OperatorPosition = OperatorPosition.FRONT,
    task: str | None = None,
    video_options: dict[str, str] | None = None,
):
    """Runs data collection in real hardware."""
    _check_rig(robot_arm, webxr, leader, nominal_joints, joints_spread, park_joints)
    camera_instances = cameras or {}
    camera_emitters = {name: cam.frame for name, cam in camera_instances.items()}
    static_meta = {}
    if task is not None:
        static_meta[keys.TASK] = task
    if robot_arm is not None:
        static_meta.update(wire.ROBOT_STATIC_META)
    output_path = None
    if output_dir is not None:
        output_path = pos3.sync(output_dir, sync_on_error=True)
        utils.save_run_metadata(output_path, patterns=['*.py', '*.toml'])
    # An operator with a hand on the leader has none free for the session, so the keys carry it.
    keyboard = KeyboardControl(quit_key='q') if leader is not None else None
    data_collection = DataCollectionController(
        operator_position.value,
        nominal_joints,
        joints_spread,
        park_joints,
        teleop=Teleop.LEADER if leader is not None else Teleop.HAND,
        output_path=output_path,
        static_meta=static_meta,
    )

    dataset_factory = partial(LocalDatasetWriter, video_options=video_options) if output_path is not None else None
    with pimm.World() as world:
        ds_agent = wire.wire(world, data_collection, dataset_factory, camera_emitters, robot_arm, gripper, None)
        _wire(world, ds_agent, data_collection, webxr, robot_arm, sound, leader, keyboard)

        # SO-101 fills both the arm and gripper slots with one object; a control system runs in exactly one process.
        gripper_cs = [] if gripper is robot_arm else [gripper]
        bg_cs = [webxr, *camera_instances.values(), ds_agent, robot_arm, *gripper_cs, leader, sound]

        if stream_video_to_webxr is not None and webxr is not None:
            world.connect(camera_emitters[stream_video_to_webxr], webxr.frame, receiver_wrapper=pimm.map(_frame_array))

        # The keyboard reads the terminal this was started from, which no spawned process holds.
        try:
            world.run([data_collection, keyboard], bg_cs)
        # rules-allow: swallowed-error — an interrupt is how an operator ends a run, and the world has
        # already stopped every process by the time it arrives here
        except KeyboardInterrupt:
            logging.info('The run ended on an interrupt')


@cfn.config(
    mujoco_model_path=package_assets_path('assets/mujoco/franka_table.xml'),
    webxr=positronic.cfg.webxr.oculus,
    cameras={
        keys.WRIST_IMAGE: 'handcam_left_ph',
        keys.EXTERIOR_IMAGE: 'back_view_ph',
        'image.handcam_right': 'handcam_right_ph',
        'image.wrist_2': 'wrist_cam_ph',
    },
    sound=positronic.cfg.sound.sound,
    operator_position=OperatorPosition.BACK,
    loaders=positronic.cfg.simulator.stack_cubes_loaders,
)
def main_sim(
    mujoco_model_path: str,
    webxr: WebXR,
    cameras: dict[str, str],
    sound: pimm.ControlSystem | None = None,
    loaders: Sequence[MujocoSceneTransform] = (),
    output_dir: str | None = None,
    fps: int = 30,
    operator_position: OperatorPosition = OperatorPosition.FRONT,
    task: str | None = None,
):
    """Runs data collection in simulator."""

    sim = MujocoSim(mujoco_model_path, loaders, camera_fps=fps)
    cameras = {name: sim.cameras[orig_name] for name, orig_name in cameras.items()}
    gui = dpg_ui()

    static_meta: dict[str, Any] = dict(wire.ROBOT_STATIC_META)
    if task is not None:
        static_meta[keys.TASK] = task

    output_path = None
    if output_dir is not None:
        output_path = pos3.sync(output_dir, sync_on_error=True)
        utils.save_run_metadata(output_path, patterns=['*.py', '*.toml'])
    data_collection = DataCollectionController(
        operator_position.value,
        sim.initial_joints,
        output_path=output_path,
        static_meta=static_meta,
        metadata_getter=lambda: {k: v.tolist() for k, v in sim.save_state().items()},
    )

    dataset_factory = LocalDatasetWriter if output_path is not None else None
    with pimm.World(virtual_time=True) as world:
        # The sim carries both the arm and the gripper ports, so it fills both slots.
        ds_agent = wire.wire(world, data_collection, dataset_factory, cameras, sim, sim, gui, TimeMode.MESSAGE)
        _wire(world, ds_agent, data_collection, webxr, sim, sound)
        world.connect(data_collection.redraw_scene, sim.env_reset)

        sim_iter = world.start([sim, data_collection], [webxr, gui, ds_agent, sound])
        sim_iter = iter(sim_iter)

        # VR teleop is live, so pace virtual time to wall time: only step the sim when it has fallen behind.
        start_time = pimm.world.SystemClock().now_ns()
        sim_start_time = world.clock.now_ns()

        while not world.should_stop:
            try:
                time_since_start = pimm.world.SystemClock().now_ns() - start_time
                if world.clock.now_ns() < sim_start_time + time_since_start:
                    next(sim_iter)
                else:
                    time.sleep(0.001)
            except StopIteration:
                break


main_cfg = cfn.Config(
    main,
    robot_arm=None,
    gripper=positronic.cfg.hardware.gripper.dh_gripper,
    webxr=positronic.cfg.webxr.oculus,
    sound=positronic.cfg.sound.sound,
    cameras={
        'image.left': positronic.cfg.hardware.camera.arducam_left,
        'image.right': positronic.cfg.hardware.camera.arducam_right,
    },
    operator_position=OperatorPosition.FRONT,
)


@cfn.config(
    robot_arm=positronic.cfg.hardware.roboarm.so101,
    webxr=positronic.cfg.webxr.oculus,
    sound=positronic.cfg.sound.sound,
    operator_position=OperatorPosition.BACK,
    cameras={'image.right': positronic.cfg.hardware.camera.arducam_right},
    nominal_joints=positronic.cfg.hardware.roboarm.SO101_NOMINAL_JOINTS,
)
def so101cfg(robot_arm, **kwargs):
    """Runs data collection on SO101 robot"""
    main(robot_arm=robot_arm, gripper=robot_arm, **kwargs)


@cfn.config(
    robot_arm=positronic.cfg.hardware.roboarm.yam,
    webxr=positronic.cfg.webxr.oculus,
    sound=positronic.cfg.sound.sound,
    operator_position=OperatorPosition.BACK,
    cameras={},
    nominal_joints=positronic.cfg.hardware.roboarm.YAM_NOMINAL_JOINTS,
    # The YAM station records several cameras on a weak CPU; x264's default preset can't keep up with the
    # camera rate, so trade ~2x bitrate for ~2.5x faster encoding.
    video_options={'preset': 'ultrafast', 'tune': 'zerolatency'},
)
def yamcfg(robot_arm, **kwargs):
    """Runs data collection on a real i2rt YAM arm (the arm driver carries the gripper)."""
    main(robot_arm=robot_arm, gripper=robot_arm, **kwargs)


@cfn.config(
    robot_arm=positronic.cfg.hardware.roboarm.trossen,
    webxr=positronic.cfg.webxr.oculus,
    # The operator stands behind the arm, and it reaches away from them over the table. Solved from the
    # rig: the controller moved away from the base runs along -z of the headset's frame, and only `BACK`
    # takes that to +x of the arm's.
    operator_position=OperatorPosition.BACK,
    # The station has no audio device, so the operator reads the recording state off the terminal.
    sound=None,
    # The wrist camera of the arm this configuration drives, and the one that looks down on the table. The
    # station carries two more, `d405_wrist_left` and `d405_scene_bottom`, which belong to the other arm.
    cameras={
        keys.WRIST_IMAGE: positronic.cfg.hardware.camera.d405_wrist_right,
        keys.EXTERIOR_IMAGE: positronic.cfg.hardware.camera.d405_scene_top,
    },
    stream_video_to_webxr=keys.WRIST_IMAGE,
    nominal_joints=positronic.cfg.hardware.roboarm.TROSSEN_NOMINAL_JOINTS,
    park_joints=positronic.cfg.hardware.roboarm.TROSSEN_PARK_JOINTS,
)
def trossencfg(robot_arm, **kwargs):
    """Runs data collection on a real Trossen WidowX AI arm (the arm driver carries the gripper)."""
    main(robot_arm=robot_arm, gripper=robot_arm, **kwargs)


# The same arm, driven by the leader arm beside it instead of from the headset. There is nothing to wear:
# the operator holds the leader, and the keys carry the session — `r` records, space puts both arms at the
# start pose, `h` takes them to rest, and `q` ends the run.
trossen_leader = trossencfg.override(
    leader=positronic.cfg.hardware.roboarm.trossen_leader, webxr=None, stream_video_to_webxr=None
)


droid = cfn.Config(
    main,
    robot_arm=positronic.cfg.hardware.roboarm.franka_droid,
    gripper=positronic.cfg.hardware.gripper.robotiq,
    nominal_joints=positronic.cfg.hardware.roboarm.FRANKA_NOMINAL_JOINTS,
    joints_spread=positronic.cfg.hardware.roboarm.FRANKA_JOINTS_SPREAD,
    webxr=positronic.cfg.webxr.oculus,
    sound=positronic.cfg.sound.sound,
    cameras={
        keys.WRIST_IMAGE: positronic.cfg.hardware.camera.zed_m.override(view='left', resolution='hd720', fps=30),
        keys.EXTERIOR_IMAGE: positronic.cfg.hardware.camera.zed_2i.override(view='left', resolution='hd720', fps=30),
    },
    operator_position=OperatorPosition.BACK,
)


human = cfn.Config(
    main,
    robot_arm=None,
    gripper=None,
    webxr=positronic.cfg.webxr.oculus,
    sound=positronic.cfg.sound.sound,
    cameras={
        keys.EXTERIOR_IMAGE: positronic.cfg.hardware.camera.zed_2i.override(view='left', resolution='hd720', fps=30)
    },
    operator_position=OperatorPosition.BACK,
)


@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli({
        'real': main_cfg,
        'so101': so101cfg,
        'yam': yamcfg,
        'trossen': trossencfg,
        'trossen_leader': trossen_leader,
        'sim': main_sim,
        'sim_pnp': main_sim.override(loaders=positronic.cfg.simulator.multi_tote_loaders),
        'droid': droid,
        'human': human,
    })


if __name__ == '__main__':
    _internal_main()
