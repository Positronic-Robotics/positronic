from enum import Enum
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterator, Sequence

import configuronic as cfn

import pimm
from positronic import geom
import positronic.cfg.hardware.camera
import positronic.cfg.hardware.gripper
import positronic.cfg.hardware.roboarm
import positronic.cfg.simulator
import positronic.cfg.sound
import positronic.cfg.webxr
from positronic.dataset.ds_writer_agent import DsWriterAgent, DsWriterCommand, DsWriterCommandType
from positronic.drivers import roboarm
from positronic.drivers.webxr import WebXR
from positronic.gui.dpg import DearpyguiUi
from positronic.simulator.mujoco.sim import (
    MujocoCamera,
    MujocoFranka,
    MujocoGripper,
    MujocoSim,
)
from positronic.simulator.mujoco.transforms import MujocoSceneTransform
from positronic.utils.buttons import ButtonHandler

from positronic.dataset.local_dataset import LocalDatasetWriter


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


class DataCollectionController:
    controller_positions_reader: pimm.SignalReader[Dict[str, geom.Transform3D]] = pimm.NoOpReader()
    buttons_reader: pimm.SignalReader[Dict] = pimm.NoOpReader()
    robot_state: pimm.SignalReader[roboarm.State] = pimm.NoOpReader()

    robot_commands: pimm.SignalEmitter[roboarm.command.CommandType] = pimm.NoOpEmitter()
    target_grip_emitter: pimm.SignalEmitter[float] = pimm.NoOpEmitter()

    ds_agent_commands: pimm.SignalEmitter[DsWriterCommand] = pimm.NoOpEmitter()
    sound_emitter: pimm.SignalEmitter[str] = pimm.NoOpEmitter()

    def __init__(self, operator_position: geom.Transform3D | None, metadata_getter: Callable[[], Dict] | None = None):
        self.operator_position = operator_position
        self.metadata_getter = metadata_getter or (lambda: {})

    def run(self, should_stop: pimm.SignalReader, clock: pimm.Clock) -> Iterator[pimm.Sleep]:  # noqa: C901
        start_wav_path = "positronic/assets/sounds/recording-has-started.wav"
        end_wav_path = "positronic/assets/sounds/recording-has-stopped.wav"

        controller_positions_reader = pimm.ValueUpdated(self.controller_positions_reader)

        tracker = _Tracker(self.operator_position)
        button_handler = ButtonHandler()

        recording = False

        while not should_stop.value:
            try:
                _parse_buttons(self.buttons_reader.value, button_handler)
                if button_handler.just_pressed('right_B'):
                    op = DsWriterCommandType.START_EPISODE if not recording else DsWriterCommandType.STOP_EPISODE
                    self.ds_agent_commands.emit(DsWriterCommand(op, self.metadata_getter()))
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
                controller_pos, controller_pos_updated = controller_positions_reader.value
                if controller_pos_updated:
                    target_robot_pos = tracker.update(controller_pos['right'])
                    if tracker.on:  # Don't spam the robot with commands.
                        self.robot_commands.emit(roboarm.command.CartesianMove(target_robot_pos))

                yield pimm.Sleep(0.001)

            except pimm.NoValueException:
                yield pimm.Sleep(0.001)
                continue


def main(robot_arm: Any | None,
         gripper: Any | None,
         webxr: WebXR,
         sound: Any | None,
         cameras: Dict[str, Any] | None,
         output_dir: str | None = None,
         stream_video_to_webxr: str | None = None,
         operator_position: OperatorPosition = OperatorPosition.FRONT):
    """Runs data collection in real hardware."""
    with pimm.World() as world:
        data_collection = DataCollectionController(operator_position.value)
        keys = ['target_grip', 'robot_commands', 'controller_positions', 'robot_state', 'grip']
        cameras = cameras or {}
        camera_mappings = {
            camera_name: f'image.{camera_name}' if camera_name != 'image' else 'image'
            for camera_name in cameras.keys()
        }
        keys.extend(camera_mappings.values())

        ds_agent = DsWriterAgent(LocalDatasetWriter(Path(output_dir)), keys) if output_dir is not None else None
        if ds_agent is not None:
            data_collection.ds_agent_commands, ds_agent.command = world.mp_pipe()
            for camera_name, output_name in camera_mappings.items():
                cameras[camera_name].frame, ds_agent.inputs[output_name] = world.mp_pipe()

        if gripper is not None:
            ems = []
            if ds_agent is not None:
                gripper.grip, ds_agent.inputs['grip'] = world.mp_pipe()
                tgem, ds_agent.inputs['target_grip'] = world.mp_pipe()
                ems.append(tgem)

            tgem, gripper.target_grip = world.mp_pipe()
            ems.append(tgem)
            data_collection.target_grip_emitter = pimm.BroadcastEmitter(ems)
            world.start_in_subprocess(gripper.run)

        if robot_arm is not None:
            cmd_ems = []
            x, robot_arm.commands = world.mp_pipe()
            cmd_ems.append(x)
            if ds_agent is not None:
                x, ds_agent.inputs['robot_commands'] = world.mp_pipe()
                cmd_ems.append(x)
            data_collection.robot_commands = pimm.BroadcastEmitter(cmd_ems)

            state_ems = []
            if ds_agent is not None:
                x, ds_agent.inputs['robot_state'] = world.shared_memory()
                state_ems.append(x)
            x, data_collection.robot_state = world.shared_memory()
            state_ems.append(x)
            robot_arm.state = pimm.BroadcastEmitter(state_ems)
            world.start_in_subprocess(robot_arm.run)

        ctrl_ems = []
        if ds_agent is not None:
            emt, ds_agent.inputs['controller_positions'] = world.mp_pipe()
            ctrl_ems.append(emt)
            world.start_in_subprocess(ds_agent.run)

        emt, data_collection.controller_positions_reader = world.mp_pipe()
        ctrl_ems.append(emt)
        webxr.controller_positions = pimm.BroadcastEmitter(ctrl_ems)

        webxr.buttons, data_collection.buttons_reader = world.mp_pipe()

        if stream_video_to_webxr is not None:
            emitter, reader = world.mp_pipe()
            cameras[stream_video_to_webxr].frame = pimm.BroadcastEmitter(
                [emitter, cameras[stream_video_to_webxr].frame])

            webxr.frame = pimm.map(reader, lambda x: x['image'])

        world.start_in_subprocess(webxr.run, *[camera.run for camera in cameras.values()])

        if sound is not None:
            data_collection.sound_emitter, sound.wav_path = world.mp_pipe()
            world.start_in_subprocess(sound.run)

        dc_steps = iter(world.interleave(data_collection.run))
        while not world.should_stop:
            try:
                time.sleep(next(dc_steps).seconds)
            except StopIteration:
                break


def main_sim(mujoco_model_path: str,
             webxr: WebXR,
             sound: Any | None = None,
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
    gripper = MujocoGripper(sim, actuator_name='actuator8_ph', joint_name='finger_joint1_ph')
    gui = DearpyguiUi()

    with pimm.World(clock=sim) as world:

        def metadata_getter():
            return {k: v.tolist() for k, v in sim.save_state().items()}

        data_collection = DataCollectionController(operator_position.value, metadata_getter=metadata_getter)

        keys = ['target_grip', 'robot_commands', 'controller_positions', 'robot_state', 'grip']
        cameras = cameras or {}
        camera_mappings = {
            camera_name: f'image.{camera_name}' if camera_name != 'image' else 'image'
            for camera_name in cameras.keys()
        }
        keys.extend(camera_mappings.values())

        ds_agent = DsWriterAgent(LocalDatasetWriter(Path(output_dir)), keys) if output_dir is not None else None

        for camera_name, camera in cameras.items():
            if ds_agent is not None:
                camera.frame, (ds_reader, gui_reader) = world.mp_one_to_many_pipe(2)
                ds_agent.inputs[camera_mappings[camera_name]] = ds_reader
                gui.cameras[camera_name] = gui_reader
            else:
                camera.frame, gui.cameras[camera_name] = world.mp_pipe()

        # WebXR I/O
        ctrl_emitters = []
        if ds_agent is not None:
            em, ds_agent.inputs['controller_positions'] = world.mp_pipe()
            ctrl_emitters.append(em)

        em, data_collection.controller_positions_reader = world.mp_pipe()
        ctrl_emitters.append(em)
        webxr.controller_positions = pimm.BroadcastEmitter(ctrl_emitters)
        webxr.buttons, data_collection.buttons_reader = world.mp_pipe()

        world.start_in_subprocess(webxr.run, gui.run)

        state_emitters = []
        if ds_agent is not None:
            x, ds_agent.inputs['robot_state'] = world.shared_memory()
            state_emitters.append(x)
        x, data_collection.robot_state = world.shared_memory()
        state_emitters.append(x)
        robot_arm.state = pimm.BroadcastEmitter(state_emitters)

        cmd_emitters = []
        x, robot_arm.commands = world.mp_pipe()
        cmd_emitters.append(x)
        if ds_agent is not None:
            x, ds_agent.inputs['robot_commands'] = world.mp_pipe()
            cmd_emitters.append(x)
        data_collection.robot_commands = pimm.BroadcastEmitter(cmd_emitters)

        tg_emitters = []
        x, gripper.target_grip = world.mp_pipe()
        tg_emitters.append(x)
        if ds_agent is not None:
            x, ds_agent.inputs['target_grip'] = world.mp_pipe()
            tg_emitters.append(x)
        data_collection.target_grip_emitter = pimm.BroadcastEmitter(tg_emitters)

        if ds_agent is not None:
            gripper.grip, ds_agent.inputs['grip'] = world.mp_pipe()

        if ds_agent is not None:
            data_collection.ds_agent_commands, ds_agent.command = world.mp_pipe()
            world.start_in_subprocess(ds_agent.run)

        if sound is not None:
            data_collection.sound_emitter, sound.wav_path = world.mp_pipe()
            world.start_in_subprocess(sound.run)

        sim_iter = world.interleave(
            sim.run,
            *[camera.run for camera in cameras.values()],
            robot_arm.run,
            gripper.run,
            data_collection.run,
        )

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


if __name__ == "__main__":
    cfn.cli({
        "real": main_cfg,
        "so101": so101cfg,
        "sim": main_sim_cfg,
    })
