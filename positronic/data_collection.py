import time
from contextlib import nullcontext
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Iterator, Sequence

import configuronic as cfn
import numpy as np

import pimm
import positronic.cfg.hardware.camera
import positronic.cfg.hardware.gripper
import positronic.cfg.hardware.roboarm
import positronic.cfg.simulator
import positronic.cfg.sound
import positronic.cfg.webxr
from positronic import geom
from positronic.dataset import ds_writer_agent
from positronic.dataset.ds_writer_agent import DsWriterAgent, DsWriterCommand, DsWriterCommandType, Serializers
from positronic.dataset.local_dataset import LocalDatasetWriter
from positronic.drivers import roboarm
from positronic.drivers.webxr import WebXR
from positronic.gui.dpg import DearpyguiUi
from positronic.simulator.mujoco.sim import MujocoCamera, MujocoFranka, MujocoGripper, MujocoSim
from positronic.simulator.mujoco.transforms import MujocoSceneTransform
from positronic.utils.buttons import ButtonHandler


def _parse_buttons(buttons: Dict, button_handler: ButtonHandler):
    for side in ['left', 'right']:
        if buttons[side] is None:
            continue

        mapping = {
            f'{side}_A': buttons[side][4],
            f'{side}_B': buttons[side][5],
            f'{side}_trigger': buttons[side][0],
            f'{side}_thumb': buttons[side][1],
            f'{side}_stick': buttons[side][3]
        }
        button_handler.update_buttons(mapping)


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
            print("Ignoring tracking on/off in UMI mode")
            return

        self.on = True
        print("Starting tracking")
        self._offset = geom.Transform3D(
            -self._teleop_t.translation + robot_pos.translation,
            self._teleop_t.rotation.inv * robot_pos.rotation,
        )

    def turn_off(self):
        if self.umi_mode:
            print("Ignoring tracking on/off in UMI mode")
            return
        self.on = False
        print("Stopped tracking")

    def update(self, tracker_pos: geom.Transform3D):
        if self.umi_mode:
            return tracker_pos

        self._teleop_t = self._operator_position * tracker_pos * self._operator_position.inv
        return geom.Transform3D(self._teleop_t.translation + self._offset.translation,
                                self._teleop_t.rotation * self._offset.rotation)


class OperatorPosition(Enum):
    # map xyz -> zxy
    FRONT = geom.Transform3D(rotation=geom.Rotation.from_quat([0.5, 0.5, 0.5, 0.5]))
    # map xyz -> zxy + flip x and y
    BACK = geom.Transform3D(rotation=geom.Rotation.from_quat([-0.5, -0.5, 0.5, 0.5]))


class DataCollectionController(pimm.ControlSystem):

    def __init__(self, operator_position: geom.Transform3D | None, metadata_getter: Callable[[], Dict] | None = None):
        self.operator_position = operator_position
        self.metadata_getter = metadata_getter or (lambda: {})
        self.controller_positions_receiver = pimm.ControlSystemReceiver(self)
        self.buttons_receiver = pimm.ControlSystemReceiver(self)
        self.robot_state = pimm.ControlSystemReceiver(self)

        self.robot_commands = pimm.ControlSystemEmitter(self)
        self.target_grip_emitter = pimm.ControlSystemEmitter(self)

        self.ds_agent_commands = pimm.ControlSystemEmitter(self)
        self.sound_emitter = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:  # noqa: C901
        start_wav_path = "positronic/assets/sounds/recording-has-started.wav"
        end_wav_path = "positronic/assets/sounds/recording-has-stopped.wav"

        controller_positions_receiver = pimm.ValueUpdated(self.controller_positions_receiver)

        tracker = _Tracker(self.operator_position)
        button_handler = ButtonHandler()

        recording = False

        while not should_stop.value:
            try:
                _parse_buttons(self.buttons_receiver.value, button_handler)
                if button_handler.just_pressed('right_B'):
                    op = DsWriterCommandType.START_EPISODE if not recording else DsWriterCommandType.STOP_EPISODE
                    meta = self.metadata_getter() if op == DsWriterCommandType.STOP_EPISODE else {}
                    self.ds_agent_commands.emit(DsWriterCommand(op, meta))
                    self.sound_emitter.emit(start_wav_path if not recording else end_wav_path)
                    recording = not recording
                elif button_handler.just_pressed('right_A'):
                    if tracker.on:
                        tracker.turn_off()
                    else:
                        tracker.turn_on(self.robot_state.value.ee_pose)
                elif button_handler.just_pressed('right_stick') and not tracker.umi_mode:
                    print("Resetting robot")
                    self.ds_agent_commands.emit(DsWriterCommand(DsWriterCommandType.ABORT_EPISODE))
                    # TODO: add sound for aborting
                    tracker.turn_off()
                    self.robot_commands.emit(roboarm.command.Reset())

                self.target_grip_emitter.emit(button_handler.get_value('right_trigger'))
                controller_pos, controller_pos_updated = controller_positions_receiver.value
                if controller_pos_updated:
                    target_robot_pos = tracker.update(controller_pos['right'])
                    if tracker.on:  # Don't spam the robot with commands.
                        self.robot_commands.emit(roboarm.command.CartesianMove(target_robot_pos))

                yield pimm.Sleep(0.001)

            except pimm.NoValueException:
                yield pimm.Sleep(0.001)
                continue


@ds_writer_agent.names({'.left': Serializers.transform_3d.names, '.right': Serializers.transform_3d.names})
def controller_positions_serializer(controller_positions: Dict[str, geom.Transform3D]) -> Dict[str, np.ndarray]:
    res = {}
    for side, pos in controller_positions.items():
        if pos is not None:
            res[f'.{side}'] = Serializers.transform_3d(pos)
    return res


def _wire(world: pimm.World, cameras: Dict[str, pimm.ControlSystem] | None, dataset_writer: LocalDatasetWriter | None,
          data_collection: DataCollectionController, webxr: WebXR, robot_arm: pimm.ControlSystem | None,
          gripper: pimm.ControlSystem | None, sound: pimm.ControlSystem | None):
    cameras = cameras or {}
    camera_mappings = {name: (f'image.{name}' if name != 'image' else 'image') for name in cameras.keys()}

    world.connect(webxr.controller_positions, data_collection.controller_positions_receiver)
    world.connect(webxr.buttons, data_collection.buttons_receiver)

    if robot_arm is not None:
        world.connect(data_collection.robot_commands, robot_arm.commands)
        world.connect(robot_arm.state, data_collection.robot_state)

    if gripper is not None:
        world.connect(data_collection.target_grip_emitter, gripper.target_grip)

    if sound is not None:
        world.connect(data_collection.sound_emitter, sound.wav_path)

    ds_agent = None
    if dataset_writer is not None:
        signals_spec = {v: None for v in camera_mappings.values()}
        signals_spec['target_grip'] = None
        signals_spec['robot_commands'] = Serializers.robot_command
        signals_spec['controller_positions'] = controller_positions_serializer
        signals_spec['robot_state'] = Serializers.robot_state
        signals_spec['grip'] = None
        ds_agent = DsWriterAgent(dataset_writer, signals_spec)
        # Controls for the data set agent
        world.connect(data_collection.ds_agent_commands, ds_agent.command)

        # The data itself
        for camera_name, camera in cameras.items():
            world.connect(camera.frame, ds_agent.inputs[camera_mappings[camera_name]])
        world.connect(webxr.controller_positions, ds_agent.inputs['controller_positions'])
        world.connect(data_collection.robot_commands, ds_agent.inputs['robot_commands'])
        world.connect(data_collection.target_grip_emitter, ds_agent.inputs['target_grip'])
        if robot_arm is not None:
            world.connect(robot_arm.state, ds_agent.inputs['robot_state'])
        if gripper is not None:
            world.connect(gripper.grip, ds_agent.inputs['grip'])

    return ds_agent


def main(robot_arm: pimm.ControlSystem | None,
         gripper: pimm.ControlSystem | None,
         webxr: WebXR,
         sound: pimm.ControlSystem | None,
         cameras: Dict[str, pimm.ControlSystem] | None,
         output_dir: str | None = None,
         stream_video_to_webxr: str | None = None,
         operator_position: OperatorPosition = OperatorPosition.FRONT):
    """Runs data collection in real hardware."""
    cameras = cameras or {}
    data_collection = DataCollectionController(operator_position.value)

    writer_cm = LocalDatasetWriter(Path(output_dir)) if output_dir is not None else nullcontext(None)
    with writer_cm as dataset_writer, pimm.World() as world:
        ds_agent = _wire(world, cameras, dataset_writer, data_collection, webxr, robot_arm, gripper, sound)

        bg_cs = [webxr]
        bg_cs.extend(cameras.values())
        if ds_agent is not None:
            bg_cs.append(ds_agent)
        if robot_arm is not None:
            bg_cs.append(robot_arm)
        if gripper is not None:
            bg_cs.append(gripper)
        if sound is not None:
            bg_cs.append(sound)

        if stream_video_to_webxr is not None:
            world.connect(cameras[stream_video_to_webxr].frame,
                          webxr.frame,
                          receiver_wrapper=lambda r: pimm.map(r, lambda x: x['image']))

        dc_steps = iter(world.start(data_collection, bg_cs))
        while not world.should_stop:
            try:
                time.sleep(next(dc_steps).seconds)
            except StopIteration:
                break


def main_sim(mujoco_model_path: str,
             webxr: WebXR,
             sound: pimm.ControlSystem | None = None,
             loaders: Sequence[MujocoSceneTransform] = (),
             output_dir: str | None = None,
             fps: int = 30,
             operator_position: OperatorPosition = OperatorPosition.FRONT):
    """Runs data collection in simulator."""

    sim = MujocoSim(mujoco_model_path, loaders)
    robot_arm = MujocoFranka(sim, suffix='_ph')
    cameras = {
        'handcam_left': MujocoCamera(sim.model, sim.data, 'handcam_left_ph', (320, 240), fps=fps),
        'handcam_right': MujocoCamera(sim.model, sim.data, 'handcam_right_ph', (320, 240), fps=fps),
    }
    side_camera = MujocoCamera(sim.model, sim.data, 'agentview', (320, 240), fps=fps)
    gripper = MujocoGripper(sim, actuator_name='actuator8_ph', joint_name='finger_joint1_ph')
    gui = DearpyguiUi()

    def metadata_getter():
        return {k: v.tolist() for k, v in sim.save_state().items()}

    data_collection = DataCollectionController(operator_position.value, metadata_getter=metadata_getter)

    writer_cm = LocalDatasetWriter(Path(output_dir)) if output_dir is not None else nullcontext(None)
    with writer_cm as dataset_writer, pimm.World(clock=sim) as world:
        ds_agent = _wire(world, cameras, dataset_writer, data_collection, webxr, robot_arm, gripper, sound)

        bg_cs = [webxr, gui]
        if ds_agent is not None:
            bg_cs.append(ds_agent)
        if sound is not None:
            bg_cs.append(sound)

        for camera_name, camera in cameras.items():
            world.connect(camera.frame, gui.cameras[camera_name])

        world.connect(side_camera.frame, gui.cameras['side'])

        sim_iter = world.start([sim, *cameras.values(), robot_arm, gripper, data_collection, side_camera], bg_cs)
        sim_iter = iter(sim_iter)

        start_time = pimm.world.SystemClock().now_ns()
        sim_start_time = sim.now_ns()

        while not world.should_stop:
            try:
                time_since_start = pimm.world.SystemClock().now_ns() - start_time
                if sim.now_ns() < sim_start_time + time_since_start:
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
        'left': positronic.cfg.hardware.camera.arducam_left,
        'right': positronic.cfg.hardware.camera.arducam_right,
    },
    operator_position=OperatorPosition.FRONT,
)

main_sim_cfg = cfn.Config(
    main_sim,
    mujoco_model_path="positronic/assets/mujoco/franka_table.xml",
    webxr=positronic.cfg.webxr.oculus,
    sound=positronic.cfg.sound.sound,
    operator_position=OperatorPosition.BACK,
    loaders=positronic.cfg.simulator.stack_cubes_loaders,
)


@cfn.config(robot_arm=positronic.cfg.hardware.roboarm.so101,
            webxr=positronic.cfg.webxr.oculus,
            sound=positronic.cfg.sound.sound,
            operator_position=OperatorPosition.BACK,
            cameras={'right': positronic.cfg.hardware.camera.arducam_right})
def so101cfg(robot_arm, **kwargs):
    """Runs data collection on SO101 robot"""
    main(robot_arm=robot_arm, gripper=robot_arm, **kwargs)


droid = cfn.Config(
    main,
    robot_arm=positronic.cfg.hardware.roboarm.franka,
    gripper=positronic.cfg.hardware.gripper.robotiq,
    webxr=positronic.cfg.webxr.oculus,
    sound=positronic.cfg.sound.sound,
    cameras={
        'wrist': positronic.cfg.hardware.camera.zed_m.override(view='side_by_side', resolution='vga', fps=30),
        'side': positronic.cfg.hardware.camera.zed_2i.override(view='left', resolution='vga', fps=30)
    },
    operator_position=OperatorPosition.FRONT,
)

if __name__ == "__main__":
    cfn.cli({
        "real": main_cfg,
        "so101": so101cfg,
        "sim": main_sim_cfg,
        "droid": droid,
    })
