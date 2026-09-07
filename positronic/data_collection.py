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
from positronic.drivers.roboarm import State as RoboarmState
from positronic.drivers.webxr import WebXR
from positronic.gui import dpg_ui
from positronic.simulator.mujoco.sim import MujocoSim
from positronic.simulator.mujoco.transforms import MujocoSceneTransform
from positronic.utils import package_assets_path
from positronic.utils.buttons import ButtonHandler


def _parse_buttons(buttons: dict, button_handler: ButtonHandler):
    for side in ['left', 'right']:
        if buttons[side] is None:
            continue

        mapping = {
            f'{side}_A': buttons[side][4],
            f'{side}_B': buttons[side][5],
            f'{side}_trigger': buttons[side][0],
            f'{side}_thumb': buttons[side][1],
            f'{side}_stick': buttons[side][3],
        }
        button_handler.update_buttons(mapping)


def _check_error(is_error, was_error):
    return is_error, is_error and not was_error


class _Tracker:
    on = False
    _offset = geom.Transform3D()
    _teleop_t = geom.Transform3D()

    def __init__(self, operator_position: geom.Transform3D | None):
        self._operator_position = operator_position
        self.on = self.umi_mode

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

    def update(self, tracker_pos: geom.Transform3D):
        if self.umi_mode:
            return tracker_pos

        self._teleop_t = self._operator_position * tracker_pos * self._operator_position.inv
        return geom.Transform3D(
            self._teleop_t.translation + self._offset.translation, self._teleop_t.rotation * self._offset.rotation
        )


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
        *,
        output_path: Path | None = None,
        static_meta: dict | None = None,
        metadata_getter: Callable[[], dict] | None = None,
    ):
        self.operator_position = operator_position
        self._output_path = output_path
        self._nominal_joints = np.asarray(nominal_joints, dtype=np.float64)
        # A station that measured no jitter sends the arm exactly to its nominal.
        spread = joints_spread if len(joints_spread) else np.zeros_like(self._nominal_joints)
        self._joints_spread = np.asarray(spread, dtype=np.float64)
        self._static_meta = static_meta or {}
        self.metadata_getter = metadata_getter or (lambda: {})
        self.controller_positions = pimm.DefaultingReceiver(self, default={})
        self.buttons_receiver = pimm.ControlSystemReceiver(self)
        self.robot_state = pimm.ControlSystemReceiver(self)
        self.gripper_state = pimm.FakeReceiver(self)  # To make compatible with other "policy" control systems
        self.frames = pimm.ReceiverDict(self, fake=True)
        self.robot_meta_in = pimm.DefaultingReceiver(self, default={})

        self.robot_commands = pimm.ControlSystemEmitter(self)
        self.sync_move = pimm.calls.ControlSystemCaller[roboarm.command.CommandType, None](self)
        self.redraw_scene = pimm.calls.ControlSystemCaller[Any, None](self)
        self.target_grip = pimm.ControlSystemEmitter(self)

        self.ds_agent_commands = pimm.ControlSystemEmitter(self)
        self.sound = pimm.ControlSystemEmitter(self)

    def _ready(self, should_stop: pimm.SignalReceiver) -> Iterator[pimm.Sleep]:
        """Redraw the scene and put the arm at a start pose drawn around the nominal joints, yielding until
        both are done.

        Raises whatever a device failed on: a call hands its handler's exception back, so the vocabulary is
        the driver's — a move abandoned, a target it cannot hold, a fault from the vendor.
        """
        logging.info('Readying the rig for the next episode')
        asks = []
        if self.redraw_scene.connected:  # a real scene is a person's to set up, and nothing here is asked
            asks.append(self.redraw_scene(None))
        if len(self._nominal_joints):  # a station with no arm has none to put anywhere
            asks.append(self.sync_move(roboarm.command.sampled_joints(self._nominal_joints, self._joints_spread)))
        ready = pimm.calls.all_of(asks)
        while not ready.done():
            if should_stop.value:
                return
            yield pimm.Sleep(0.001)
        ready.result()

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:  # noqa: C901
        sounds = Path(package_assets_path('assets/sounds'))
        start_wav_path = sounds / 'recording-has-started.wav'
        end_wav_path = sounds / 'recording-has-stopped.wav'
        abort_wav_path = sounds / 'recording-has-been-aborted.wav'
        error_wav_path = sounds / 'error-occurred.wav'

        tracker = _Tracker(self.operator_position)
        button_handler = ButtonHandler()

        recording = False
        in_error = False

        while not should_stop.value:
            try:
                _parse_buttons(self.buttons_receiver.value, button_handler)
                if button_handler.just_pressed('right_B'):
                    if not recording:
                        meta = dict(self._static_meta)
                        meta.update(self.robot_meta_in.value)
                        meta.update(self.metadata_getter())
                        self.ds_agent_commands.emit(DsWriterCommand.START(self._output_path, meta))
                        self.sound.emit(start_wav_path)
                    else:
                        self.ds_agent_commands.emit(DsWriterCommand.STOP())
                        self.sound.emit(end_wav_path)
                    recording = not recording
                elif button_handler.just_pressed('right_A'):
                    if tracker.on:
                        tracker.turn_off()
                    else:
                        tracker.turn_on(self.robot_state.value.ee_pose)
                elif button_handler.just_pressed('right_stick') and not tracker.umi_mode:
                    if recording:
                        self.ds_agent_commands.emit(DsWriterCommand.ABORT())
                        self.sound.emit(abort_wav_path)
                    tracker.turn_off()
                    recording = False
                    try:
                        yield from self._ready(should_stop)
                    # rules-allow: swallowed-error — the operator hears it and asks again, session goes on
                    except Exception as e:
                        logging.error(f'The rig was not readied: {e}')
                        self.sound.emit(error_wav_path)

                self.target_grip.emit(button_handler.get_value('right_trigger'))
                cp_msg = self.controller_positions.read()
                if cp_msg.updated:
                    target_robot_pos = tracker.update(cp_msg.data['right'])

                if tracker.on:
                    in_error, entered_error = _check_error(
                        self.robot_state.value.status == roboarm.RobotStatus.ERROR, in_error
                    )
                    if entered_error:
                        self.sound.emit(error_wav_path)
                    if not in_error and cp_msg.updated:
                        cmd = roboarm.command.CartesianPosition(target_robot_pos)
                        self.robot_commands.emit(cmd)

                yield pimm.Sleep(0.001)

            except pimm.NoValueException:
                yield pimm.Sleep(0.001)
                continue


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
    webxr: WebXR,
    robot_arm: pimm.ControlSystem | None,
    sound: pimm.ControlSystem | None,
):
    world.connect(webxr.controller_positions, data_collection.controller_positions)
    world.connect(webxr.buttons, data_collection.buttons_receiver)

    if robot_arm is not None:
        # A driver's ports are its own: ``pimm.ControlSystem`` declares none for the checker to find.
        world.connect(data_collection.sync_move, robot_arm.sync_move)  # pyright: ignore[reportAttributeAccessIssue]

    if sound is not None:
        world.connect(data_collection.sound, sound.wav_path)
        if robot_arm is not None:
            world.connect(robot_arm.state, sound.level, receiver_wrapper=pimm.map(_wrench_to_level))

    if ds_agent is not None:
        if robot_arm is not None:
            ds_agent.add_signal('controller_positions', controller_positions_serializer)
            world.connect(webxr.controller_positions, ds_agent.inputs['controller_positions'])
        world.connect(data_collection.ds_agent_commands, ds_agent.command)

    return ds_agent


def main(
    robot_arm: pimm.ControlSystem | None,
    gripper: pimm.ControlSystem | None,
    webxr: WebXR,
    sound: pimm.ControlSystem | None,
    cameras: dict[str, pimm.ControlSystem] | None,
    # The start pose the right stick puts the arm at: drawn around ``nominal_joints``, within ``joints_spread``.
    nominal_joints: Sequence[float] = (),
    joints_spread: Sequence[float] = (),
    output_dir: str | None = None,
    stream_video_to_webxr: str | None = None,
    operator_position: OperatorPosition = OperatorPosition.FRONT,
    task: str | None = None,
    video_options: dict[str, str] | None = None,
):
    """Runs data collection in real hardware."""
    if (robot_arm is not None) != (len(nominal_joints) > 0):
        raise ValueError(
            '--robot_arm and --nominal_joints are named together or not at all: the right stick puts the arm '
            'a station has at the pose it measured, and either one alone leaves the other with nothing'
        )
    if len(joints_spread) not in (0, len(nominal_joints)):
        raise ValueError(
            f'--joints_spread names {len(joints_spread)} joints and --nominal_joints names {len(nominal_joints)}: '
            'the spread is jitter measured per joint, so it carries one value for each, or none at all'
        )
    if not np.all(np.isfinite([*nominal_joints, *joints_spread])):
        raise ValueError(
            '--nominal_joints and --joints_spread name joint angles: every value has to be finite, or the '
            'draw between them raises instead of reaching the arm'
        )
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
    data_collection = DataCollectionController(
        operator_position.value, nominal_joints, joints_spread, output_path=output_path, static_meta=static_meta
    )

    dataset_factory = partial(LocalDatasetWriter, video_options=video_options) if output_path is not None else None
    with pimm.World() as world:
        ds_agent = wire.wire(world, data_collection, dataset_factory, camera_emitters, robot_arm, gripper, None)
        _wire(world, ds_agent, data_collection, webxr, robot_arm, sound)

        # SO-101 fills both the arm and gripper slots with one object; a control system runs in exactly one process.
        gripper_cs = [] if gripper is robot_arm else [gripper]
        bg_cs = [webxr, *camera_instances.values(), ds_agent, robot_arm, *gripper_cs, sound]

        if stream_video_to_webxr is not None:
            world.connect(
                camera_emitters[stream_video_to_webxr],
                webxr.frame,
                receiver_wrapper=pimm.map(lambda adapter: adapter.array),
            )

        world.run(data_collection, bg_cs)


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


droid = cfn.Config(
    main,
    robot_arm=positronic.cfg.hardware.roboarm.franka_droid,
    gripper=positronic.cfg.hardware.gripper.robotiq,
    nominal_joints=positronic.cfg.hardware.roboarm.FRANKA_NOMINAL_JOINTS,
    joints_spread=positronic.cfg.hardware.roboarm.FRANKA_JOINTS_SPREAD,
    webxr=positronic.cfg.webxr.oculus,
    sound=positronic.cfg.sound.sound,
    cameras=positronic.cfg.hardware.camera.droid,
    operator_position=OperatorPosition.BACK,
)


droid_3cam = droid.override(cameras=positronic.cfg.hardware.camera.droid_3cam)


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
        'sim': main_sim,
        'sim_pnp': main_sim.override(loaders=positronic.cfg.simulator.multi_tote_loaders),
        'droid': droid,
        'droid_3cam': droid_3cam,
        'human': human,
    })


if __name__ == '__main__':
    _internal_main()
