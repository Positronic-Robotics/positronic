"""Serve MolmoSpaces benchmarks through the env-server protocol.

Runs in MolmoSpaces' isolated interpreter, with ``mapping``, ``server`` and ``protocol`` on ``PYTHONPATH``.
End-effector poses describe ``gripper/grasp_site`` in the arm base frame, ``fr3_link0``, as the real FR3 reports them.
"""

import argparse
import copy
import os
import sys
import types

# MuJoCo's CGL package and its ctypes bindings to Apple's OpenGL framework.
_CGL_PACKAGE = 'mujoco.cgl'
_CGL_MODULE = 'mujoco.cgl.cgl'


def _install_cgl_noop_stub() -> None:
    # HACK: MolmoSpaces treats every device_id=None context as CGL and imports ``from mujoco.cgl import cgl``
    # to unlock it. On Linux that import tries to load Apple's OpenGL framework, even with EGL or OSMesa.
    # Stub the CGL lock calls there; macOS needs the real bindings.
    if sys.platform == 'darwin' or _CGL_PACKAGE in sys.modules:
        return
    # ModuleType supports runtime attributes absent from its type definition.
    cgl = types.ModuleType(_CGL_MODULE)
    cgl.CGLLockContext = cgl.CGLUnlockContext = lambda *args, **kwargs: None  # pyright: ignore[reportAttributeAccessIssue]
    package = types.ModuleType(_CGL_PACKAGE)
    package.cgl = cgl  # pyright: ignore[reportAttributeAccessIssue]
    sys.modules[_CGL_PACKAGE] = package
    sys.modules[_CGL_MODULE] = cgl


_install_cgl_noop_stub()

from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import arm_action  # noqa: E402 -- positronic-free wire mappings, on PYTHONPATH
import env_hook  # noqa: E402 -- positronic-free, on PYTHONPATH
import mapping  # noqa: E402 -- positronic-free wire mappings, on PYTHONPATH
import mujoco  # noqa: E402
import mujoco_state  # noqa: E402 -- positronic-free, on PYTHONPATH
import numpy as np  # noqa: E402
import protocol  # noqa: E402
import success_profile  # noqa: E402 -- positronic-free, on PYTHONPATH
from server import EnvProtocol, EnvServer  # noqa: E402

_DESCRIPTION = 'Serve MolmoSpaces over the env-server protocol.'
# validate.py and parity_native.py import this module with flags and a -h of their own, so main checks the flags.
_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_parser.add_argument('--host', default='localhost')
_parser.add_argument('--port', type=int)
_parser.add_argument('--right-view-blocked', type=Path)
_parser.add_argument('--hook-root', type=Path, help='the directory of the hook files that a request can name')

# Loading the runner first avoids a circular import in json_eval_task_sampler.
import molmo_spaces.evaluation.json_eval_runner  # noqa: E402, F401  # pyright: ignore[reportMissingImports]
from molmo_spaces.configs.camera_configs import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    FrankaDroidCameraSystem,
    FrankaOmniPurposeCameraSystem,
)
from molmo_spaces.configs.policy_configs import DummyPolicyConfig  # noqa: E402  # pyright: ignore[reportMissingImports]
from molmo_spaces.configs.robot_configs import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    ActionNoiseConfig,
    FrankaRobotConfig,
)
from molmo_spaces.env.object_manager import ObjectManager  # noqa: E402  # pyright: ignore[reportMissingImports]
from molmo_spaces.evaluation.benchmark_schema import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    RobotMountedCameraSpec,
    load_all_episodes,
)
from molmo_spaces.evaluation.configs.evaluation_configs import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    JsonBenchmarkEvalConfig,
)
from molmo_spaces.evaluation.eval_main import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    determine_task_horizon,
)
from molmo_spaces.tasks.json_eval_task_sampler import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    JsonEvalTaskSampler,
)
from molmo_spaces.tasks.opening_tasks import OpeningTask  # noqa: E402  # pyright: ignore[reportMissingImports]
from molmo_spaces.tasks.pick_and_place_next_to_task import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    PickAndPlaceNextToTask,
)
from molmo_spaces.tasks.pick_and_place_task import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    PickAndPlaceTask,
)
from molmo_spaces.tasks.pick_task import PickTask  # noqa: E402  # pyright: ignore[reportMissingImports]
from molmo_spaces.utils import mujoco_scene_utils  # noqa: E402  # pyright: ignore[reportMissingImports]


class _DroidEvalConfig(JsonBenchmarkEvalConfig):
    """Franka DROID configuration for benchmark episodes, with policy inference supplied by the client."""

    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: DummyPolicyConfig = DummyPolicyConfig()

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config = ActionNoiseConfig(enabled=False)
        # The benchmark episodes are judged under the physics they were recorded in. Island sleep freezes bodies at
        # rest, and the grasp probe adds a free body to the state; the eval needs neither.
        self.task_sampler_config.enable_scene_sleep = False
        self.task_sampler_config.add_grasp_probe = False


def _names_on_top_by_contact(data: mujoco.MjData, receptacle: set[int], names: set[str], min_cos: float) -> set[str]:
    """The ``names`` of the root bodies that touch a ``receptacle`` geom from above, as MolmoSpaces tests a contact."""
    model = data.model
    found: set[str] = set()
    for contact in data.contact:
        first, second = (int(geom) in receptacle for geom in contact.geom)
        if first == second:
            continue
        other = int(model.body_rootid[model.geom_bodyid[contact.geom[1 if first else 0]]])
        name = model.body(other).name
        if name not in names:
            continue
        normal = contact.frame[:3] / np.linalg.norm(contact.frame[:3])
        if second:
            normal = -normal
        center, _ = mujoco_scene_utils.body_aabb(model, data, other)
        if contact.pos[2] < center[2] and normal[2] >= min_cos:
            found.add(name)
    return found


_upstream_objects_on_receptacle = ObjectManager.objects_on_receptacle


def _objects_on_receptacle(
    self: Any,
    objs_to_check: list[Any],
    bench_geom_ids: list[int],
    angle_threshold: float = np.radians(30),
    fallback_thres: float = 0.01,
    attempt_contact: bool = True,
) -> list[Any]:
    """``ObjectManager.objects_on_receptacle`` that computes a box only for a body it can find and the receptacle."""
    data = self.data
    model = data.model
    on_top: set[str] = set()
    if attempt_contact:
        names = {obj.name for obj in objs_to_check}
        on_top = _names_on_top_by_contact(data, {int(g) for g in bench_geom_ids}, names, np.cos(angle_threshold))
    # The fallback stage reads only the root body of each receptacle geom.
    geom_of_root = {int(model.body_rootid[model.geom_bodyid[g]]): g for g in bench_geom_ids}
    rest = [obj for obj in objs_to_check if obj.name not in on_top]
    fallback = _upstream_objects_on_receptacle(
        self, rest, list(geom_of_root.values()), angle_threshold, fallback_thres, attempt_contact=False
    )
    found = on_top | {obj.name for obj in fallback}
    return [obj for name in sorted(found) if (obj := self.get_object_by_name(name))]


def use_fast_objects_on_receptacle() -> None:
    """Make every ``ObjectManager`` of this process use ``_objects_on_receptacle``."""
    # HACK: upstream computes the box of the root body across each receptacle contact before it checks the body name.
    # For a static geom that root is ``world``, whose box reads every mesh vertex, so ``get_info`` takes seconds.
    # rules-allow: hidden-dependency — MolmoSpaces builds its ObjectManager internally and takes no class to use.
    ObjectManager.objects_on_receptacle = _objects_on_receptacle


# The arm joints of the robot model that the client records.
_FR3_ARM_JOINTS = 7


class MolmoSpacesEnv(EnvProtocol):
    """A MolmoSpaces benchmark environment with one active episode, under the hook that its reset token names."""

    def __init__(
        self,
        assets_dir: Path,
        right_view_blocked: dict[mapping.BenchmarkPath, frozenset[int]],
        hook_root: Path | None = None,
    ) -> None:
        """``hook_root`` is the directory of the hook files that a request can name; None refuses every hook file."""
        use_fast_objects_on_receptacle()
        self._assets_dir = assets_dir
        self._hooks = env_hook.Hooks(hook_root)
        self._hook = self._hooks.of({}).hook  # The hook of the active episode.
        self._found = mapping.discover_benchmarks(assets_dir)
        self._right_view_blocked = right_view_blocked
        self._episodes: dict[mapping.BenchmarkPath, list[Any]] = {}
        self._sampler: Any = None
        self._task: Any = None
        self._robot_view: Any = None
        self._control_dt: float | None = None
        self._judge: success_profile.Judge | None = None
        self._meta: dict[str, Any] = {}
        self._camera_names: list[str] = []

    def _episodes_of(self, bench: mapping.BenchmarkPath) -> list[Any]:
        if bench not in self._episodes:
            # Report available benchmarks when the requested directory is missing.
            mapping.select_benchmarks(self._found, bench._asdict())
            episodes = load_all_episodes(bench.under(self._assets_dir))
            if not episodes:
                raise ValueError(f'{bench.relative} holds no episodes')
            self._episodes[bench] = episodes
        return self._episodes[bench]

    def _assert_bundled_arm(self) -> None:
        """Require the FR3 arm joints of the recorded robot model, and arm poses at the gripper grasp site."""
        arm = self._robot_view.get_move_group(mapping.MOLMO_ARM_GROUP)
        if len(arm.joint_pos) != _FR3_ARM_JOINTS:
            raise ValueError(f'the arm has {len(arm.joint_pos)} joints, not the {_FR3_ARM_JOINTS} of the FR3')
        if arm.leaf_frame_type != 'site':
            raise ValueError(f'arm move group measures at a {arm.leaf_frame_type}, not the expected site')
        name = mujoco.mj_id2name(arm.mj_model, mujoco.mjtObj.mjOBJ_SITE, arm.leaf_frame_id)
        if name != mapping.MOLMO_GRASP_SITE and not name.endswith(f'/{mapping.MOLMO_GRASP_SITE}'):
            raise ValueError(f'arm move group measures at site {name!r}, expected {mapping.MOLMO_GRASP_SITE!r}')

    @staticmethod
    def _left_shoulder_config(episode: Any) -> Any:
        """The config of the episode's left shoulder camera, in the camera system that recorded the episode."""
        # The camera systems that record the Franka benchmarks, by the name an episode's source gives.
        systems = {system.__name__: system for system in (FrankaDroidCameraSystem, FrankaOmniPurposeCameraSystem)}
        system = episode.source.camera_system_class if episode.source is not None else None
        if system not in systems:
            raise ValueError(f'camera system {system!r} is not one of {sorted(systems)}')
        recorded = {camera.name for camera in episode.cameras}
        for config in systems[system]().cameras:
            if config.name in mapping.LEFT_SHOULDER_CAMERAS and config.name in recorded:
                return config
        raise ValueError(f'the episode records none of {mapping.LEFT_SHOULDER_CAMERAS}; it records {sorted(recorded)}')

    @staticmethod
    def _with_right_shoulder_camera(episode: Any, rng: np.random.Generator) -> Any:
        """A copy of ``episode`` with a right shoulder camera: the left camera's nominal pose mirrored and jittered."""
        left = MolmoSpacesEnv._left_shoulder_config(episode)
        offset, quat = mapping.mirror_camera_pose(left.camera_offset, left.camera_quaternion)
        offset, quat = mapping.jitter_camera_pose(
            offset, quat, left.pos_noise_range, left.orientation_noise_degrees, rng
        )
        right = RobotMountedCameraSpec(
            name=mapping.RIGHT_SHOULDER_CAMERA,
            reference_body_names=left.reference_body_names,
            camera_offset=offset.tolist(),
            lookat_offset=left.lookat_offset,
            camera_quaternion=quat.tolist(),
            fov=left.fov,
        )
        with_right = episode.model_copy()
        with_right.cameras = [*episode.cameras, right]
        return with_right

    def _episode(self, bench: mapping.BenchmarkPath, episode_index: int, with_right_shoulder: bool) -> Any:
        """The episode that the server builds: with the right shoulder camera when ``with_right_shoulder``."""
        episode = self._episodes_of(bench)[episode_index]
        if with_right_shoulder:
            episode = self._with_right_shoulder_camera(episode, mapping.episode_rng(bench, episode_index))
        return episode

    def _eval_config(self, bench: mapping.BenchmarkPath, episode: Any, hook: env_hook.EnvHook) -> _DroidEvalConfig:
        """The eval config of ``episode``, with the changes of ``hook``."""
        cfg = _DroidEvalConfig()
        if episode.scene_dataset == mapping.MOLMO_USER_SCENE_DATASET:
            sampling = cfg.task_sampler_config
            scenes = mapping.user_scene_xml_paths(self._assets_dir, bench.scene_dataset, episode.data_split)
            sampling.scene_xml_paths = [str(path) for path in scenes]
            sampling.house_variant = mapping.MOLMO_USER_SCENE_VARIANT
        hook.eval_config(cfg, copy.deepcopy(episode))
        return cfg

    def _build(
        self,
        hook: env_hook.EnvHook,
        bench: mapping.BenchmarkPath,
        episode_index: int,
        seed: int | None,
        horizon_sec: float | None,
        with_right_shoulder: bool,
        with_object_image_points: bool = True,
    ) -> float:
        """Sample the episode's task under ``hook``, and return its control period in seconds."""
        if self._sampler is not None:
            self._sampler.close()
        episodes = self._episodes_of(bench)
        episode = self._episode(bench, episode_index, with_right_shoulder)
        cfg = self._eval_config(bench, episode, hook)
        cfg.seed = mapping.resolve_episode_seed(episode, episode_index, seed)
        # MolmoSpaces resolves one horizon from the full benchmark, unless an override in steps replaces it.
        cfg.task_horizon = determine_task_horizon(
            episodes, mapping.horizon_steps(horizon_sec, cfg.policy_dt_ms), cfg.policy_dt_ms
        )
        self._sampler = JsonEvalTaskSampler(cfg, episode)
        # Task sampling places the objects; task.reset() alone does not restore the scene.
        self._task = self._sampler.sample_task(house_index=episode.house_index)
        hook.task_sampled(self._task)
        if not with_object_image_points:
            # The sensor draws its points from the global NumPy generator, so without it an episode draws other values.
            self._task.sensor_suite.sensors.pop(mapping.MOLMO_OBJECT_IMAGE_POINTS_SENSOR)
        self._robot_view = self._task.env.current_robot.robot_view
        self._assert_bundled_arm()
        self._meta = {
            # Generated task descriptions can mislabel close episodes as "Open ...".
            mapping.META_TASK: episode.language.task_description,
            mapping.META_HOUSE_INDEX: episode.house_index,
        }
        return cfg.policy_dt_ms / 1000.0

    def _strict_check(self, profile: success_profile.SuccessProfile) -> success_profile.Check:
        """The ``profile`` check of the sampled task, which must be of a family that the profile judges."""
        cfg = self._task.config.task_config
        data = self._robot_view.mj_data
        robot = self._robot_view.root_body_id

        def body(name: str) -> int:
            return data.model.body(name).id

        # A subclass of a family's task is of that family, so the more derived classes come first.
        if isinstance(self._task, OpeningTask):
            if cfg.joint_index is None:
                raise ValueError(f'{profile} needs the joint that the task names')
            target = self._task.env.object_managers[0].get_object_by_name(cfg.pickup_obj_name)
            joint = target.joint_ids[cfg.joint_index]
            if self._task.config.task_type == 'close':
                return success_profile.Close(data, joint, profile)
            return success_profile.Open(data, joint, profile)
        if isinstance(self._task, PickAndPlaceNextToTask):
            return success_profile.PlaceNextTo(data, body(cfg.pickup_obj_name), body(cfg.place_receptacle_name), robot)
        if isinstance(self._task, PickAndPlaceTask):
            return success_profile.Place(
                data,
                body(cfg.pickup_obj_name),
                body(cfg.place_receptacle_name),
                robot,
                cfg.max_place_receptacle_rot_displacement,
            )
        if isinstance(self._task, PickTask):
            return success_profile.Pick(data, body(cfg.pickup_obj_name), robot)
        raise ValueError(f'{profile} judges no {type(self._task).__name__}')

    def tasks(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        """Selected episode records, with names, reset parameters and horizons in seconds."""
        selection = spec.get(mapping.SELECT_EPISODES)
        override_sec = spec.get(mapping.SELECT_HORIZON_SEC)
        pinned = None if selection is None else [selection] if isinstance(selection, int) else list(selection)
        hook, hook_record = self._hooks.of(spec)
        records = []
        for bench in mapping.select_benchmarks(self._found, spec):
            episodes = self._episodes_of(bench)
            with_right_shoulder = spec[mapping.WITH_RIGHT_SHOULDER]
            blocked = self._right_view_blocked.get(bench, frozenset()) if with_right_shoulder else frozenset()
            # The horizon in seconds for each policy period of the benchmark's episodes.
            horizons: dict[float, float] = {}
            for i in mapping.select_episodes(bench, len(episodes), pinned, blocked):
                dt_ms = self._eval_config(bench, self._episode(bench, i, with_right_shoulder), hook).policy_dt_ms
                if dt_ms not in horizons:
                    steps = determine_task_horizon(episodes, mapping.horizon_steps(override_sec, dt_ms), dt_ms)
                    horizons[dt_ms] = steps * dt_ms / 1000.0
                records.append({
                    **bench._asdict(),
                    mapping.TOKEN_EPISODE_INDEX: i,
                    mapping.TASK_NAME: episodes[i].language.task_description,
                    protocol.TASK_INSTRUCTION: episodes[i].language.task_description,
                    mapping.TASK_HORIZON_SEC: horizons[dt_ms],
                    mapping.TASK_HORIZON_OVERRIDE_SEC: override_sec,
                    mapping.TASK_ENV_HOOK: hook_record,
                })
        return records

    def reset(self, token: dict[str, Any], slots: int | None = None) -> dict[str, Any]:
        bench = mapping.BenchmarkPath(**{d: token[d] for d in mapping.BenchmarkPath._fields})
        profile = success_profile.SuccessProfile(
            token.get(mapping.TOKEN_SUCCESS_PROFILE, success_profile.SuccessProfile.MOLMO_NATIVE)
        )
        hook, hook_record = self._hooks.of(token)
        self._control_dt = control_dt = self._build(
            hook,
            bench,
            token[mapping.TOKEN_EPISODE_INDEX],
            token.get(mapping.TOKEN_SEED),
            token.get(mapping.TOKEN_HORIZON_SEC),
            token[mapping.WITH_RIGHT_SHOULDER],
            token.get(mapping.WITH_OBJECT_IMAGE_POINTS, True),
        )
        self._hook = hook
        obs, infos = self._task.reset()  # One observation dict and one info dict per environment.
        # The checks of a strict judge take the start poses of the task's bodies, so they come after the task reset.
        match profile:
            case success_profile.SuccessProfile.STRICT_V1 | success_profile.SuccessProfile.STRICT_V2:
                self._judge = success_profile.StrictJudge(profile, self._task, self._strict_check(profile), control_dt)
            case success_profile.SuccessProfile.MOLMO_NATIVE_TAIL_V1:
                self._judge = success_profile.MolmoNativeTailJudge(self._task, control_dt)
            case success_profile.SuccessProfile.MOLMO_NATIVE:
                self._judge = success_profile.MolmoNativeJudge(self._task)
        env_obs = obs[0]
        self._camera_names = [k for k, v in env_obs.items() if mapping.is_rgb_frame(v)]
        metrics = self._task_metrics(self._task.get_reward(), infos, self._judge.observe(infos[0]))
        return {
            protocol.SLOTS: [{protocol.FRAME_OBS: self._observe(env_obs, metrics, arm_error=False)}],
            protocol.FRAME_META: {
                **self._meta,
                mapping.META_SUCCESS_PROFILE: profile.value,
                mapping.META_ENV_HOOK: hook_record,
            },
            protocol.FRAME_ROBOT_META: {},  # The client supplies the robot model through static_meta.
            protocol.FRAME_CONTROL_DT: self._control_dt,
        }

    def step(self, actions: dict[int, dict[str, Any]]) -> dict[str, Any]:
        (action,) = actions.values()  # MolmoSpaces runs one scene per server, so this server serves one slot
        wire = protocol.single_arm(action)
        measured = self._measured_arm_q()
        arm = arm_action.arm_targets(
            wire[protocol.ROBOT_COMMAND], measured, ik=self._ik, read_eef=self._measured_eef_pose
        )
        gripper = np.array([self._hook.grip_actuator(wire[protocol.TARGET_GRIP])], dtype=np.float32)
        obs, rewards, _term, _trunc, infos = self._task.step({
            mapping.MOLMO_ARM_GROUP: arm.q,
            mapping.MOLMO_GRIPPER_GROUP: gripper,
        })
        if self._judge is None:
            raise RuntimeError('a MolmoSpaces env steps only after a reset')
        judged = self._judge.step(infos[0])
        metrics = self._task_metrics(rewards, infos, judged.metrics)
        return {
            protocol.SLOTS: [
                {
                    protocol.FRAME_OBS: self._observe(obs[0], metrics, arm_error=arm.refusal is not None),
                    protocol.FRAME_DONE: judged.done,
                    protocol.FRAME_SUCCESS: judged.success,
                }
            ],
            protocol.FRAME_CONTROL_DT: self._control_dt,
        }

    def _measured_arm_q(self) -> np.ndarray:
        return np.asarray(self._robot_view.get_move_group(mapping.MOLMO_ARM_GROUP).joint_pos, dtype=np.float32)

    def _measured_eef_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """The grasp-site arm-base pose as (translation, 3x3 rotation)."""
        eef = np.asarray(self._robot_view.get_move_group(mapping.MOLMO_ARM_GROUP).leaf_frame_to_root, dtype=np.float64)
        return eef[:3, 3].copy(), eef[:3, :3].copy()

    def _ik(self, target_pos: np.ndarray, target_rot: np.ndarray) -> np.ndarray | None:
        """Solve an arm-base grasp-site target with MolmoSpaces' IK; None where it finds no solution."""
        pose = np.eye(4)
        pose[:3, 3] = target_pos
        pose[:3, :3] = target_rot
        arm_base = np.asarray(self._robot_view.get_move_group(mapping.MOLMO_ARM_GROUP).root_frame_to_robot)
        solution = self._task.env.current_robot.kinematics.ik(
            move_group_id=mapping.MOLMO_ARM_GROUP,
            pose=arm_base @ pose,
            unlocked_move_group_ids=[mapping.MOLMO_ARM_GROUP],
            q0=self._robot_view.get_qpos_dict(),
            base_pose=self._robot_view.base.pose,
            rel_to_base=True,
        )
        return None if solution is None else solution[mapping.MOLMO_ARM_GROUP]

    @staticmethod
    def _task_metrics(rewards: np.ndarray, infos: list[dict[str, Any]], judged: dict[str, Any]) -> dict[str, Any]:
        """The first environment's reward and info entries, and the metrics of the success profile, as signals."""
        return protocol.numeric_signals({**infos[0], mapping.TASK_METRIC_REWARD: rewards[0], **judged})

    def _observe(self, env_obs: dict[str, Any], metrics: dict[str, Any], *, arm_error: bool) -> dict[str, Any]:
        """Raw observations, including the grasp-site pose in the arm base frame."""
        arm = self._robot_view.get_move_group(mapping.MOLMO_ARM_GROUP)
        scene = self._robot_view.mj_data
        eef_pos, eef_rot = self._measured_eef_pose()
        eef_quat = np.zeros(4)
        mujoco.mju_mat2Quat(eef_quat, np.ascontiguousarray(eef_rot.reshape(9)))
        payload = {
            protocol.OBS_JOINT_POS: np.asarray(arm.joint_pos, dtype=np.float32),
            protocol.OBS_JOINT_VEL: np.asarray(arm.joint_vel, dtype=np.float32),
            protocol.OBS_EEF_POS: eef_pos.astype(np.float32),
            protocol.OBS_EEF_QUAT: eef_quat.astype(np.float32),
            protocol.OBS_GRIP: np.float32(
                self._hook.grip(env_obs[mapping.MOLMO_OBS_QPOS][mapping.MOLMO_GRIPPER_GROUP])
            ),
            mapping.OBS_SIM_STATE: mujoco_state.integration_state(scene.model, scene),
            mapping.OBS_TASK_METRICS: metrics,
            protocol.OBS_ARM_ERROR: arm_error,
        }
        for name in self._camera_names:
            payload[name] = np.ascontiguousarray(env_obs[name])
        return payload

    def close(self) -> None:
        if self._sampler is not None:
            self._sampler.close()
            self._sampler = None
            self._task = None


def main() -> None:
    parser = argparse.ArgumentParser(description=_DESCRIPTION, parents=[_parser], allow_abbrev=False)
    args = parser.parse_args()
    if args.port is None or args.right_view_blocked is None:
        parser.error('--port and --right-view-blocked are required')
    assets = os.environ.get(mapping.ASSETS_DIR_ENV)
    if not assets:
        _parser.error(f'{mapping.ASSETS_DIR_ENV} must point at the MolmoSpaces asset packs')
    env = MolmoSpacesEnv(Path(assets), mapping.load_right_view_blocked(args.right_view_blocked), args.hook_root)
    EnvServer(env, args.host, args.port).serve_forever()


if __name__ == '__main__':
    main()
