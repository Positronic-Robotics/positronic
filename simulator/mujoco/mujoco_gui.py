import asyncio
import glob
import os
import multiprocessing
from pathlib import Path
import threading

from dataclasses import dataclass
from typing import Sequence
import hydra
import numpy as np
from omegaconf import DictConfig
import dearpygui.dearpygui as dpg

from drivers.roboarm import RobotStatus
import ironic as ir
from geom import Transform3D
from ironic.utils import FPSCounter
from simulator.mujoco.environment import MujocoSimulatorCS, InverseKinematics
from simulator.mujoco import metric_calculators
from simulator.mujoco.sim import MujocoRenderer, MujocoSimulator
from tools.dataset_dumper import DatasetDumper


def _set_image_uint8_to_float32(target, source):
    target[:] = source
    target[:] /= 255


def gen_fn(model_path):
    # TODO: robosuite's OpenGL is conflicting with dearpygui's OpenGL, so we need to run this in a separate process
    from simulator.mujoco.scene.scene_builder import construct_scene
    construct_scene(model_path)


def generate_scene(output_dir: str):
    assert output_dir is not None, "data_output_dir must be specified to build the scene"

    scene_dir = os.path.join(output_dir, "scenes")
    os.makedirs(scene_dir, exist_ok=True)

    next_scene_idx = len(glob.glob(f"{scene_dir}/*.xml"))
    new_model_path = os.path.join(scene_dir, f"scene_{next_scene_idx:04d}.xml")

    print(f"Building scene to {new_model_path}")

    process = multiprocessing.Process(target=gen_fn, args=(new_model_path, ))
    process.start()
    process.join()

    return new_model_path


@dataclass
class DesiredAction:
    position: np.ndarray
    orientation: np.ndarray
    grip: float


@ir.ironic_system(input_ports=["images", "robot_status"],
                  input_props=["robot_position", "actuator_values", "robot_grip", "metrics"],
                  output_ports=[
                      "gripper_target_grasp",
                      "robot_target_position",
                      "reset",
                      "start_recording",
                      "stop_recording",
                  ],
                  output_props=["metadata"])
class DearpyguiUi(ir.ControlSystem):
    speed_meters_per_second = 0.1
    movement_vectors = {
        'forward': np.array([speed_meters_per_second, 0, 0]),
        'backward': np.array([-speed_meters_per_second, 0, 0]),
        'left': np.array([0, speed_meters_per_second, 0]),
        'right': np.array([0, -speed_meters_per_second, 0]),
        'up': np.array([0, 0, speed_meters_per_second]),
        'down': np.array([0, 0, -speed_meters_per_second]),
    }

    key_map = {
        dpg.mvKey_W: 'forward',
        dpg.mvKey_S: 'backward',
        dpg.mvKey_A: 'left',
        dpg.mvKey_D: 'right',
        dpg.mvKey_LControl: 'down',
        dpg.mvKey_LShift: 'up',
    }

    def __init__(self, camera_names: Sequence[str]):
        super().__init__()
        self.width = None
        self.height = None
        self.camera_names = camera_names

        self.ui_thread_started = False  # this flag is needed to prevent thread re-initialization
        self.gui_ready = False  # this flag is needed to prevent update before GUI is ready
        self.ui_thread = threading.Thread(target=self.ui_thread_main, daemon=True)
        self.ui_stop_event = threading.Event()
        self.swap_buffer_lock = threading.Lock()
        self.loop = asyncio.get_running_loop()

        self.desired_action = None
        self.last_robot_status = None

        self.recording = False
        self.last_move_ts = None

        self.move_key_states = {
            'forward': False,
            'backward': False,
            'left': False,
            'right': False,
            'up': False,
            'down': False,
        }

        self.raw_textures = None

        self.second_buffer = None

    async def update(self):
        if not self.gui_ready:
            return

        self.move()

        if self.desired_action is None:
            await self._reset_desired_action()

        target_pos = Transform3D(self.desired_action.position, self.desired_action.orientation)

        _, _, robot_position = await asyncio.gather(
            self.outs.gripper_target_grasp.write(ir.Message(self.desired_action.grip)),
            self.outs.robot_target_position.write(ir.Message(target_pos.copy())), self.ins.robot_position())

        dpg.set_value(
            "robot_position", f"Robot Translation: {robot_position.data.translation}\n"
            f"Robot Quaternion: {robot_position.data.quaternion}")
        dpg.set_value(
            "target", f"Target Position: {self.desired_action.position}\n"
            f"Target Quat: {self.desired_action.orientation}\n"
            f"Target Grip: {self.desired_action.grip}")

        if self.is_bound('actuator_values'):
            actuator_values = await self.ins.actuator_values()
            values_str = "[" + ", ".join(map(lambda x: f"{x:.4f}", actuator_values.data)) + "]"
            dpg.set_value("actuator_values", values_str)

        if self.is_bound('metrics'):
            metrics = await self.ins.metrics()

            formatted_metrics = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.data.items())
            dpg.set_value("metrics", f"Metrics: {formatted_metrics}")

    def key_down(self, sender, app_data):
        key = app_data[0]
        key = self.key_map.get(key, None)
        if key is not None:
            self.move_key_states[key] = True

    def key_release(self, sender, app_data):
        key = app_data
        key = self.key_map.get(key, None)
        if key is not None:
            self.move_key_states[key] = False

    def grab(self):
        self.desired_action.grip = 1.0 - self.desired_action.grip

    def switch_recording(self):
        self.loop.create_task(self._switch_recording())

    async def _reset_desired_action(self):
        pos_msg, grip_msg = await asyncio.gather(self.ins.robot_position(), self.ins.robot_grip())
        self.desired_action = DesiredAction(position=pos_msg.data.translation.copy(),
                                            orientation=pos_msg.data.quaternion.copy(),
                                            grip=grip_msg.data)

    @ir.out_property
    async def metadata(self):
        return ir.Message({'ui': 'dearpygui'})

    async def _stop_recording(self):
        if self.recording:
            await self.outs.stop_recording.write(ir.Message({}))
            self.recording = False

    async def _switch_recording(self):
        if self.recording:
            await self._stop_recording()
        else:
            await self.outs.start_recording.write(ir.Message(True))
            self.recording = True

    def move(self):
        time_since_last_move = ir.system_clock() - self.last_move_ts if self.last_move_ts is not None else 0
        time_since_last_move /= 10**9

        for key, vector in self.movement_vectors.items():
            if self.move_key_states.get(key, False):
                self.desired_action.position += vector * time_since_last_move

        self.last_move_ts = ir.system_clock()

    def _configure_image_grid(self):
        n_images = len(self.camera_names)
        n_cols = int(np.ceil(np.sqrt(n_images)))
        n_rows = int(np.ceil(n_images / n_cols))

        with dpg.table(header_row=False):
            for _ in range(n_cols):
                dpg.add_table_column()

            for i in range(n_rows):
                with dpg.table_row():
                    for j in range(n_cols):
                        idx = i * n_cols + j
                        if idx < n_images:
                            cam_name = self.camera_names[idx]
                            dpg.add_image(texture_tag=cam_name, tag=f"image_{cam_name}")

    def _configure_image_sizes(self):
        n_images = len(self.camera_names)
        n_cols = int(np.ceil(np.sqrt(n_images)))
        n_rows = int(np.ceil(n_images / n_cols))

        width = dpg.get_item_width("image_grid")
        height = dpg.get_item_height("image_grid")

        for key in self.camera_names:
            dpg.set_item_width(f"image_{key}", int(width / n_cols))
            dpg.set_item_height(f"image_{key}", int(height / n_rows))

    def _viewport_resize(self):
        width = dpg.get_viewport_width()
        height = dpg.get_viewport_height()

        dpg.set_item_width("image_grid", width)
        dpg.set_item_height("image_grid", height)

    @ir.on_message('images')
    async def on_images(self, message: ir.Message):
        images = message.data
        if self.width is None and self.height is None:
            # TODO: Every image may have different size
            self.width = images[self.camera_names[0]].shape[1]
            self.height = images[self.camera_names[0]].shape[0]

            self.raw_textures = {
                cam_name: np.ones((self.height, self.width, 4), dtype=np.float32)
                for cam_name in self.camera_names
            }

            self.second_buffer = {
                cam_name: np.zeros((self.height, self.width, 3), dtype=np.float32)
                for cam_name in self.camera_names
            }

        with self.swap_buffer_lock:
            for cam_name in self.camera_names:
                try:
                    _set_image_uint8_to_float32(self.second_buffer[cam_name], images[cam_name])
                except KeyError as e:
                    raise ValueError(f"Camera {cam_name} not found among {images.keys()}") from e

    @ir.on_message('robot_status')
    async def on_robot_status(self, message: ir.Message):
        if message.data == RobotStatus.RESETTING and self.last_robot_status != RobotStatus.RESETTING:
            await self._stop_recording()
        elif message.data == RobotStatus.AVAILABLE and self.last_robot_status != RobotStatus.AVAILABLE:
            await self._reset_desired_action()

        self.last_robot_status = message.data

    def ui_thread_main(self):
        dpg.create_context()
        with dpg.texture_registry():
            for cam_name in self.camera_names:
                dpg.add_raw_texture(width=self.width,
                                    height=self.height,
                                    tag=cam_name,
                                    format=dpg.mvFormat_Float_rgba,
                                    default_value=self.raw_textures[cam_name])

        with dpg.window(tag="image_grid", label="Cameras", no_scrollbar=True, no_scroll_with_mouse=True):
            self._configure_image_grid()

        with dpg.window(label="Info"):
            dpg.add_text("", tag="target")
            dpg.add_text("", tag="robot_position")
            if self.is_bound('actuator_values'):
                dpg.add_input_text(label="actuator_values", tag="actuator_values", auto_select_all=True)

            if self.is_bound('metrics'):
                dpg.add_text("", tag="metrics")

        def reset_callback():
            self.loop.create_task(self.outs.reset.write(ir.Message(True)))

        with dpg.handler_registry():
            dpg.add_key_down_handler(callback=self.key_down)
            dpg.add_key_release_handler(callback=self.key_release)
            dpg.add_key_press_handler(key=dpg.mvKey_G, callback=self.grab)
            dpg.add_key_press_handler(key=dpg.mvKey_R, callback=self.switch_recording)
            dpg.add_key_press_handler(key=dpg.mvKey_Spacebar, callback=reset_callback)

        with dpg.item_handler_registry(tag="adjust_images"):
            dpg.add_item_resize_handler(callback=self._configure_image_sizes)

        dpg.bind_item_handler_registry("image_grid", "adjust_images")

        dpg.create_viewport(title='Custom Title', width=800, height=600)
        dpg.set_viewport_resize_callback(callback=self._viewport_resize)
        dpg.setup_dearpygui()
        dpg.show_viewport(maximized=True)
        self.gui_ready = True
        fps_counter = FPSCounter("UI")

        while not self.ui_stop_event.is_set() and dpg.is_dearpygui_running():
            with self.swap_buffer_lock:
                for key in self.raw_textures:
                    self.raw_textures[key][:, :, :3] = self.second_buffer[key]

            dpg.render_dearpygui_frame()
            fps_counter.tick()

    async def cleanup(self):
        self.ui_stop_event.set()
        self.ui_thread.join()
        dpg.destroy_context()

    async def step(self):
        if self.width is not None and self.height is not None and not self.ui_thread_started:
            self.ui_thread_started = True
            self.ui_thread.start()

        if self.width is None or self.height is None:
            return ir.State.ALIVE

        await self.update()
        return ir.State.ALIVE if self.ui_thread.is_alive() else ir.State.FINISHED


async def _main(cfg: DictConfig):
    width = cfg.mujoco.camera_width
    height = cfg.mujoco.camera_height

    if cfg.mujoco.model_path is None:
        cfg.mujoco.model_path = generate_scene(cfg.data_output_dir)

    loaders = hydra.utils.instantiate(cfg.mujoco.loaders)
    simulator = MujocoSimulator.load_from_xml_path(cfg.mujoco.model_path,
                                                   simulation_rate=1 / cfg.mujoco.simulation_hz,
                                                   loaders=loaders)
    renderer = MujocoRenderer(
        simulator,
        render_resolution=(width, height),
        camera_names=cfg.mujoco.camera_names,
    )
    inverse_kinematics = InverseKinematics(simulator)

    metrics = [
        metric_calculators.ObjectMovedCalculator(
            model=simulator.model,
            data=simulator.data,
            object_name='box_1_main',
            threshold=0.01,
        ),
        metric_calculators.ObjectDistanceCalculator(
            model=simulator.model,
            data=simulator.data,
            object_1='box_0_main',
            object_2='box_1_main',
        ),
        metric_calculators.ObjectLiftedTimeCalculator(
            model=simulator.model,
            data=simulator.data,
            object_name='box_1_main',
            threshold=0.01,
        ),
        metric_calculators.ObjectsStackedCalculator(
            model=simulator.model,
            data=simulator.data,
            object_1='box_0_main',
            object_2='box_1_main',
            velocity_threshold=0.01,
        ),
    ]

    # systems
    simulator = MujocoSimulatorCS(
        simulator=simulator,
        simulation_rate=1 / cfg.mujoco.simulation_hz,
        render_rate=1 / cfg.mujoco.observation_hz,
        renderer=renderer,
        inverse_kinematics=inverse_kinematics,
        metric_calculators=metrics,
    )

    window = DearpyguiUi(cfg.mujoco.camera_names)

    systems = [
        simulator.bind(
            gripper_target_grasp=window.outs.gripper_target_grasp,
            robot_target_position=window.outs.robot_target_position,
            reset=window.outs.reset,
        ),
        window.bind(
            images=simulator.outs.images,
            robot_position=simulator.outs.robot_position,
            robot_grip=simulator.outs.grip,
            actuator_values=simulator.outs.actuator_values,
            metrics=simulator.outs.metrics,
        ),
    ]

    if cfg.data_output_dir is not None:

        def get_translation(position):
            return position.translation

        def get_quaternion(position):
            return position.quaternion

        def discard_images(images):
            # The idea is just to pass the pulse of images, not the data
            return {}

        properties_to_dump = ir.utils.properties_dict(
            robot_joints=simulator.outs.joints,
            robot_position_translation=ir.utils.map_property(get_translation, simulator.outs.robot_position),
            robot_position_quaternion=ir.utils.map_property(get_quaternion, simulator.outs.robot_position),
            ext_force_ee=simulator.outs.ext_force_ee,
            ext_force_base=simulator.outs.ext_force_base,
            grip=simulator.outs.grip,
            actuator_values=simulator.outs.actuator_values,
        )

        model_path = Path(cfg.mujoco.model_path)
        data_output_dir = Path(cfg.data_output_dir)
        assert data_output_dir in model_path.parents, (
            f"Mujoco model {model_path}"
            f" must be in the data output directory {data_output_dir} for transferability")

        relative_data_output_dir = model_path.relative_to(data_output_dir)

        episode_metadata = {
            'mujoco_model_path': cfg.mujoco.model_path,
            'relative_mujoco_model_path': str(relative_data_output_dir),
            'simulation_hz': cfg.mujoco.simulation_hz,
        }

        data_dumper = DatasetDumper(cfg.data_output_dir, additional_metadata=episode_metadata)
        systems.append(
            data_dumper.bind(
                image=ir.utils.map_port(discard_images, simulator.outs.images),
                target_grip=window.outs.gripper_target_grasp,
                target_robot_position=window.outs.robot_target_position,
                start_episode=window.outs.start_recording,
                end_episode=window.outs.stop_recording,
                robot_data=properties_to_dump,
            ))

    composed = ir.compose(*systems)

    await ir.utils.run_gracefully(composed)


@hydra.main(version_base=None, config_path="../../configs", config_name="mujoco_gui")
def main(cfg: DictConfig):
    asyncio.run(_main(cfg))


if __name__ == "__main__":
    main()
