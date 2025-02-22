import abc
from typing import Dict, Optional, Sequence, Tuple

import hydra
import mujoco
import numpy as np
from dm_control import mujoco as dm_mujoco
from dm_control.utils import inverse_kinematics as ik
from omegaconf import DictConfig

from ironic.utils import FPSCounter
from geom import Transform3D
from simulator.mujoco.scene.transforms import MujocoSceneTransform, load_model_from_spec_file

STATE_SPECS = [
    mujoco.mjtState.mjSTATE_FULLPHYSICS,
    mujoco.mjtState.mjSTATE_USER,
    mujoco.mjtState.mjSTATE_INTEGRATION,
    mujoco.mjtState.mjSTATE_WARMSTART,
]


def xmat_to_quat(xmat):
    site_quat = np.empty(4)
    mujoco.mju_mat2Quat(site_quat, xmat)
    return site_quat


class MujocoMetricCalculator(abc.ABC):
    def __init__(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            grace_time: Optional[float] = 0.2,
    ):
        self.model = model
        self.data = data
        self.grace_time = grace_time

    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def _update(self) -> Dict[str, float]:
        pass

    def update(self):
        if self.grace_time is not None and self.data.time < self.grace_time:
            self.initialize()
            return
        self._update()

    @abc.abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class CompositeMujocoMetricCalculator(MujocoMetricCalculator):
    def __init__(self, metric_calculators: Sequence[MujocoMetricCalculator]):
        model = metric_calculators[0].model if len(metric_calculators) > 0 else None
        data = metric_calculators[0].data if len(metric_calculators) > 0 else None

        super().__init__(model=model, data=data, grace_time=None)

        self.metric_calculators = metric_calculators

    def initialize(self):
        for calculator in self.metric_calculators:
            calculator.initialize()

    def _update(self):
        for calculator in self.metric_calculators:
            calculator.update()

    def get_metrics(self) -> Dict[str, float]:
        metrics = {}
        for calculator in self.metric_calculators:
            metrics.update(calculator.get_metrics())
        return metrics

    def reset(self):
        for calculator in self.metric_calculators:
            calculator.reset()


class MujocoSimulator:
    def __init__(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            simulation_rate: float = 1 / 500,
            model_suffix: str = '',
            initial_ctrl: Optional[np.ndarray] = None,
            warmup_steps: int = 1000,
    ):
        super().__init__()
        self.model = model
        self.data = data
        self.model.opt.timestep = simulation_rate
        self.simulation_fps_counter = FPSCounter('Simulation')
        self._initial_position = None
        self.model_suffix = model_suffix
        self.initial_ctrl = initial_ctrl
        self.joint_names = [self.name(f'joint{i}') for i in range(1, 8)]
        self.joint_qpos_ids = [model.joint(joint).qposadr.item() for joint in self.joint_names]
        self.warmup_steps = warmup_steps

    def name(self, name: str):
        return f'{name}{self.model_suffix}'

    @property
    def robot_position(self):
        return Transform3D(
            translation=self.data.site(self.name('end_effector')).xpos.copy(),
            quaternion=xmat_to_quat(self.data.site(self.name('end_effector')).xmat.copy())
        )

    @property
    def grip(self):
        return float(self.data.actuator(self.name('actuator8')).ctrl)

    @property
    def joints(self):
        return np.array([self.data.qpos[i] for i in self.joint_qpos_ids])

    @property
    def actuator_values(self):
        return self.data.ctrl.copy()

    @property
    def ts_sec(self):
        return self.data.time

    @property
    def ts_ns(self):
        return int(self.ts_sec * 1e9)

    @property
    def ext_force_ee(self):
        return np.zeros(6)  # TODO: implement

    @property
    def ext_force_base(self):
        return np.zeros(6)  # TODO: implement

    @property
    def initial_position(self):
        return self._initial_position

    def step(self):
        mujoco.mj_step(self.model, self.data)
        self.simulation_fps_counter.tick()

    def save_state(self) -> Dict[str, np.ndarray]:
        """
        Saves full state of the simulator.

        This state could be used to restore the exact state of the simulator.

        Returns:
            data: A dictionary containing the full state of the simulator.
        """
        data = {}

        for spec in STATE_SPECS:
            size = mujoco.mj_stateSize(self.model, spec)
            data[spec.name] = np.empty(size, np.float64)
            mujoco.mj_getState(self.model, self.data, data[spec.name], spec)

        return data

    def load_state(self, state: Dict[str, np.ndarray], reset_time: bool = True):
        """
        Restores the state of the simulator.

        If reset_time is True, the time will be reset to 0. It will not affect simulation since
        physics is time-invariant. (https://mujoco.readthedocs.io/en/stable/computation/index.html#full-physics-state)

        Args:
            state: (Dict[str, np.ndarray]) The state to load.
            reset_time: (bool) If True, the time will be reset to 0.
        """
        mujoco.mj_resetData(self.model, self.data)
        for spec in STATE_SPECS:
            mujoco.mj_setState(self.model, self.data, np.array(state[spec.name]), spec)

        if reset_time:
            self.data.time = 0

    def reset(self):
        """
        Reset the simulator to the initial position.
        """
        mujoco.mj_resetData(self.model, self.data)
        if self.initial_ctrl is not None:
            self.data.ctrl = self.initial_ctrl
        mujoco.mj_step(self.model, self.data, self.warmup_steps)
        self.data.time = 0
        self._initial_position = self.robot_position

    def set_actuator_values(self, actuator_values: np.ndarray):
        for i in range(7):
            self.data.actuator(self.name(f'actuator{i + 1}')).ctrl = actuator_values[i]

    def set_grip(self, grip: float):
        self.data.actuator(self.name('actuator8')).ctrl = grip

    @staticmethod
    def load_from_xml_path(model_path: str, loaders: Sequence[MujocoSceneTransform] = (), **kwargs) -> 'MujocoSimulator':
        model, metadata = load_model_from_spec_file(model_path, loaders)
        data = mujoco.MjData(model)

        if 'model_suffix' in metadata:
            kwargs['model_suffix'] = metadata['model_suffix']

        if 'initial_ctrl' in metadata:
            kwargs['initial_ctrl'] = [float(x) for x in metadata['initial_ctrl'].split(',')]

        return MujocoSimulator(model, data, **kwargs)


class InverseKinematics:
    def __init__(self, simulator: MujocoSimulator):
        super().__init__()
        self.simulator = simulator
        self.physics = dm_mujoco.Physics.from_model(simulator.data)

    def recalculate_ik(self, target_robot_position: Transform3D) -> Optional[np.ndarray]:
        """
        Returns None if the IK calculation failed
        """
        result = ik.qpos_from_site_pose(
            physics=self.physics,
            site_name=self.simulator.name('end_effector'),
            target_pos=target_robot_position.translation,
            target_quat=target_robot_position.quaternion,
            joint_names=self.simulator.joint_names,
            rot_weight=0.5,
        )

        if result.success:
            return result.qpos[self.simulator.joint_qpos_ids]
        print(f"Failed to calculate IK for {target_robot_position}")
        return None


class MujocoRenderer:
    def __init__(
            self,
            simulator: MujocoSimulator,
            camera_names: Sequence[str],
            render_resolution: Tuple[int, int] = (320, 240),
    ):
        super().__init__()
        self.simulator = simulator
        self.renderer = None
        self.render_resolution = render_resolution
        self.observation_fps_counter = FPSCounter('Renderer')
        self.camera_names = camera_names

    def render_frames(self):
        views = {}

        for cam_name in self.camera_names:
            camera_name = self.simulator.name(cam_name)
            self.renderer.update_scene(self.simulator.data, camera=camera_name)
            views[cam_name] = self.renderer.render()

        self.observation_fps_counter.tick()
        return views

    def initialize(self):
        """
        Initialize the renderer. This must be called before calling render().
        """
        assert self.renderer is None, "Renderer already initialized."
        # in case we have other code which works with OpenGL, we need to initialize the renderer in a separate thread to avoid conflicts
        self.renderer = mujoco.Renderer(self.simulator.model, height=self.render_resolution[1], width=self.render_resolution[0])

    def render(self) -> Dict[str, np.ndarray]:
        assert self.renderer is not None, "You must call initialize() before calling render()"
        images = self.render_frames()

        return images

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def create_from_config(cfg: DictConfig) -> Tuple[MujocoSimulator, MujocoRenderer, InverseKinematics]:
    loaders = hydra.utils.instantiate(cfg.mujoco_loaders)

    simulator = MujocoSimulator.load_from_xml_path(
        model_path=cfg.mujoco.model_path,
        loaders=loaders,
        simulation_rate=1 / cfg.mujoco.simulation_hz
    )
    renderer = MujocoRenderer(
        simulator,
        camera_names=cfg.mujoco.camera_names,
        render_resolution=(cfg.mujoco.camera_width, cfg.mujoco.camera_height)
    )
    inverse_kinematics = InverseKinematics(simulator)

    simulator.reset()

    return simulator, renderer, inverse_kinematics