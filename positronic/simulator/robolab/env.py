"""RoboLab behind the env-server protocol, in RoboLab's own interpreter, as ``launcher.py`` runs it.

It imports only robolab, isaaclab and the modules beside it, never positronic. One server serves one robot
(``--robot``) and one camera set (``--cameras``), and steps at most ``--max-envs`` scene clones, one per slot.

The wire contract:

- DROID: the env steps RoboLab's 8 joint-position values at 15 Hz. ``joint_pos`` passes through bit-identically,
  ``joint_vel`` anchors on the measured joints, ``hold`` re-commands them, and a pose solves through ``pose_ik``
  once per step. Poses are in the eef control frame (``eef_frame``): position env-local, rotation world.
- YAM: the env steps the 14 values of ``yam_spec.ACTION_TERMS``, from one command and one grip per arm on the
  ``protocol.arm_channel`` channels. It solves no pose command. Every pose is in the rig world frame
  (``yam_spec.ROOT_IN_RIG_WORLD``), and ``robot_meta`` carries the arm mounts in that frame.
- The reset token names the RoboLab env and the instruction variant, and carries no seed: RoboLab's eval path has no
  seed hook. A reset rebuilds the scene only when the task key or the clone count changes.
- A clone that terminates freezes, with a zero action, on its verdict until the slowest clone of the batch ends.
- ``gt_state`` (``keys.GT_STATE``) holds per slot, for each contact object but the table, its pose (position in the
  clone frame, for the YAM in the rig world; rotation in the world frame) and whether the gripper contact sensor
  touches it (for the YAM, any finger of either hand); and RoboLab's progress through each condition chain of the
  current subtask.
- ``--task-lib`` adds a task library (``task_catalog``): its tasks run by name beside the benchmark, and the camera
  sets its ``cameras.py`` returns join RoboLab's presets for ``--cameras``.
"""

import argparse
import functools
import importlib.util
import math
import os
import sys
import tempfile
from abc import abstractmethod
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from typing import Any

import cv2  # noqa: F401 -- robolab requires cv2 imported before isaaclab
import keys
import numpy as np
import pose_ik
import protocol  # pyright: ignore[reportMissingImports]
import torch
import yam_spec
from isaaclab.app import AppLauncher
from server import DEFAULT_JOIN_DEADLINE, EnvProtocol, EnvServer
from task_catalog import TaskCatalog


class _NoopTelemetry:
    """Inert stand-in for the span writer when it is not bundled into this interpreter, so an uninstrumented
    run pays nothing and every span call site works without a telemetry-present check."""

    # Read at decorator-application time by ``@telemetry.traced(telemetry.SPAN_ENV_STEP)``; the value is unused
    # (``traced`` below ignores the name) but the names must resolve when the real writer is not bundled in.
    SPAN_ENV_STEP = ''
    SPAN_ENV_RESET = ''

    def active(self) -> bool:
        return False

    def span(self, name: str, **attrs: Any) -> AbstractContextManager[None]:
        return nullcontext()

    def traced(self, name: str, **attrs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return lambda method: method

    def bind_from_env(self) -> AbstractContextManager[None]:
        return nullcontext()


# The env server writes its spans through the positronic-free ``telemetry`` module copied flat alongside
# ``server``/``protocol``; the no-op stand-in keeps every span call site inert when it is not bundled in.
try:
    import telemetry
except ImportError:
    telemetry = _NoopTelemetry()

# Isaac's rigid bring-up order: parse CLI args, launch the app, and only then import anything that touches the
# omni/isaaclab runtime — which forces the second import block below and the module-level parse. ``validate.py``
# imports this module without ``--port``, so ``main`` checks the port instead of argparse.
parser = argparse.ArgumentParser(description='Serve RoboLab over the env-server protocol.')
parser.add_argument('--host', default='localhost')
parser.add_argument('--port', type=int)
parser.add_argument('--robot', default=keys.DROID, choices=keys.ROBOTS)
parser.add_argument(
    '--cameras', help='a camera set of the robot, in keys.ROBOT_CAMERA_SETS; keys.DEFAULT_CAMERA_SETS without it'
)
parser.add_argument('--max-envs', type=int, default=1, help='the most scene clones this server steps at once')
parser.add_argument(
    '--join-deadline',
    type=float,
    default=DEFAULT_JOIN_DEADLINE,
    help='seconds the first reset of a batch waits for the clones whose clients have not asked for it',
)
parser.add_argument('--task-lib', type=Path, action='append', default=[], help='a task library directory; repeatable')
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.cameras = args.cameras or keys.DEFAULT_CAMERA_SETS[args.robot]
keys.camera_set(args.robot, args.cameras)
args.enable_cameras = True  # not a CLI flag: every robolab runner forces it (the image obs need rendering)
simulation_app = AppLauncher(args).app

import omni.kit.app  # noqa: E402
import omni.timeline  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    matrix_from_quat,
    quat_from_matrix,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

import robolab.constants  # noqa: E402
from robolab.core.environments.config import parse_env_cfg  # noqa: E402  # pyright: ignore[reportMissingImports]
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402

# robolab resolves from its own uv project at run time, so the checker cannot see these modules.
from robolab.core.events.subtask_recorder import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    SubtaskCompletionRecorderTerm,
)
from robolab.core.task.instruction import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    select_instruction_fields,
)
from robolab.core.task.task import resolve_instruction  # noqa: E402  # pyright: ignore[reportMissingImports]
from robolab.core.utils.function_loader import get_callable_info  # noqa: E402  # pyright: ignore[reportMissingImports]
from robolab.core.world.world_state import get_world  # noqa: E402  # pyright: ignore[reportMissingImports]
from robolab.registrations.droid import camera_presets  # noqa: E402  # pyright: ignore[reportMissingImports]
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402
from robolab.robots.droid import EEF_OFFSET_ROT, DroidCfg  # noqa: E402

# Both flags gate recorder construction in the env cfg's ``__post_init__``, so they are set before any
# ``create_env``: subtask progress feeds the wire ``subtask`` observation; per-step image recording only bloats
# the HDF5 the recorder writes. ``create_env`` also writes ``env_cfg.json`` and recordings into the global
# output dir — point it at a temp dir instead of robolab's repo checkout.
robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = True
robolab.constants.RECORD_IMAGE_DATA = False
robolab.constants.set_output_dir(tempfile.mkdtemp(prefix='robolab-env-'))


def _load_robot_meta() -> dict[str, Any]:
    """The DROID rig's model (the viewer's URDF + Robotiq meshes) decoded from the wire codec and emitted as
    ``robot_meta``. This env runs in RoboLab's Isaac interpreter and cannot import positronic to build the model
    itself, so the launcher builds ``bundled_franka_model()`` on the positronic side, serializes it, and passes
    the path in ``ROBOLAB_ROBOT_META``. Empty when a caller (e.g. ``validate.py``) imports the env without it."""
    path = os.environ.get('ROBOLAB_ROBOT_META')
    if path is None:
        return {}
    with open(path, 'rb') as f:
        return protocol.decode(f.read())


def _camera_presets(task_libs: list[Path]) -> dict[str, list[Any]]:
    """Every camera set by name: RoboLab's presets, and the sets each task library's ``cameras.py`` returns."""
    presets: dict[str, list[Any]] = {
        keys.WRIST_LEFT_RIGHT: camera_presets.WRIST_LEFT_RIGHT,
        keys.WRIST_LEFT: camera_presets.WRIST_LEFT,
        keys.WRIST_RIGHT: camera_presets.WRIST_RIGHT,
    }
    for index, lib in enumerate(task_libs):
        path = lib / 'cameras.py'
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location(f'task_lib_cameras_{index}', path)
        assert spec is not None and spec.loader is not None  # a file that exists always yields a spec
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        presets.update(module.camera_sets())
    return presets


@contextmanager
def _no_table_fixture(library_task: bool) -> Iterator[None]:
    """Register a task library's task on no robot table fixture: its scene holds the robot base.

    RoboLab spawns the table fixture that the robot cfg names (``DroidCfg.table_fixture``) into every task scene
    when the task registers.
    """
    if not library_task:
        yield
        return
    fixture, DroidCfg.table_fixture = DroidCfg.table_fixture, None
    try:
        yield
    finally:
        DroidCfg.table_fixture = fixture


class RobolabEnv(EnvProtocol):
    """A RoboLab task behind the ``tasks``/``reset``/``step``/``close`` the env server serves.

    Built from the reset token's key (RoboLab env name, instruction variant, episode length) and the clone count,
    and cached; ``reset`` rebuilds when the key changes and re-randomizes the scene through RoboLab's own reset
    events, or restores an exact recorded state when the token carries one (demo replay). ``step`` maps each slot's
    forwarded commands into its row of the robot's action and reports, per slot, ``done`` (episode over)
    and ``success`` (the task's success termination fired). A slot whose arm state stops being finite is
    faulted: its episode ends as a failure with ``fault``, and the other slots continue.

    The scene clones, one per slot and at most ``max_envs``, share one task, one sim and one ``env.step``; the whole
    batch reruns the same task on every reset, since RoboLab clones a single registered env.
    """

    # The one entry of a task's contact list that the gripper's batch contact sensor leaves out. A contact query over
    # the other entries, in their order, reads that sensor in one pass
    # (``robolab.core.sensors.contact_sensor_utils.create_contact_sensors``).
    _TABLE = 'table'
    # The env origin in the world frame that the server reports every pose in.
    _ENV_IN_WORLD = np.zeros(3, dtype=np.float32)

    def __init__(self, max_envs: int = 1):
        if max_envs < 1:
            raise ValueError(f'a server steps at least one scene clone, not {max_envs}')
        self._max_envs = max_envs
        self._num_envs = 0  # the clones of the built env
        self._key = None
        # RoboLab's env, its cfg and the handles below resolve only inside RoboLab's own interpreter, so the
        # checker cannot see their types here.
        self._env: Any = None
        self._env_cfg: Any = None
        self._meta = None
        self._control_dt = None
        # Tasks registered with gymnasium in this process; re-registering rebuilds the cfg class and churns
        # the registry, so each registers once.
        self._registered: set[str] = set()
        self._catalog = TaskCatalog.load(
            Path(robolab.constants.TASK_DIR), args.task_lib, robolab.constants.BENCHMARK_TASK_CATEGORIES
        )
        # Rebuilt with the env: robot handles and model-structure indices.
        self._robot: Any = None
        self._joint_ids: Any = None  # every arm joint of the robot, in the order of the wire arms
        self._grip: Any = None  # each slot's latest grip command per arm, which an arm with no command holds
        self._faulted: Any = None  # per slot, whether its arm state stopped being finite in this episode
        self._world: Any = None
        self._gripper = ''  # the contact gripper label that the gripper contact of the ``gt_state`` reads
        self._objects: list[str] = []
        self._subtask_recorder: Any = None
        self._shown_chains: list[Any] = []  # each slot's chain machine at the last frame
        self._episode = 0  # the number of the current episode, counted by resets
        self._episode_task = ''  # the task of the current episode
        self._timeline = omni.timeline.get_timeline_interface()
        self._kit_app = omni.kit.app.get_app()

    @property
    def num_slots(self) -> int:
        return self._max_envs

    def _timed_phase(self, method: Callable[..., Any], name: str) -> Callable[..., Any]:
        """``method`` wrapped to record a ``name`` span (parented to the in-flight ``env.step``/``env.reset``)
        each call, so the sim's native physics/render cost decomposes the step under ``--timing``."""

        def timed(*args: Any, **kwargs: Any) -> Any:
            with telemetry.span(name):
                return method(*args, **kwargs)

        return timed

    def tasks(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        """The tasks ``spec`` selects: ``task`` is one of RoboLab's categories, a task name, or a list of
        names. Without a ``task`` the whole benchmark runs. A task library's task runs only by its name. Each record
        states the instruction in the phrasing that ``instruction_type`` names and names that phrasing for the reset.
        A task that has phrasings but not that one raises ``ValueError`` here, before any scene is built."""
        selected = self._catalog.select(spec.get('task'))
        instruction_type = spec[keys.SELECT_INSTRUCTION_TYPE]
        return [
            {
                keys.RECORD_NAME: t.name,
                keys.RECORD_EPISODE_LENGTH_S: t.episode_length_s,
                protocol.TASK_INSTRUCTION: resolve_instruction(
                    select_instruction_fields(t.instruction, instruction_type), instruction_type
                ),
                keys.RECORD_INSTRUCTION_TYPE: instruction_type,
            }
            for t in selected
        ]

    def _build(self, task: str, instruction_type: str, episode_length_s: float | None, num_envs: int) -> None:
        self._key = None  # a build that raises leaves no cached env for the next reset to reuse
        if self._env is not None:
            self._env.close()  # release the prior task's env before create_env opens a fresh USD stage
        self._num_envs = num_envs
        if task not in self._registered:
            self._register(task)
            self._registered.add(task)
        env_name = get_envs(task=task)[0]
        scene: Any = env_name
        if episode_length_s is not None:
            # ``create_env`` parses a named scene with these arguments and builds its registered entry point,
            # ``RobolabEnv``, from the cfg; a parsed cfg goes to the same class, with the episode length replaced.
            scene = parse_env_cfg(
                env_name,
                device=args.device,
                seed=0,
                num_envs=self._num_envs,
                env_spacing=None,
                eye=None,
                lookat=None,
                use_fabric=True,
            )
            scene.episode_length_s = episode_length_s
            # RoboLab takes a time-out within two control steps for a physics artifact and resets the clone in
            # place, so such an episode never ends.
            if math.ceil(episode_length_s / (scene.sim.dt * scene.decimation)) <= 2:
                raise ValueError(f'an episode of {episode_length_s} s is too short for RoboLab to end it')
        self._env, self._env_cfg = create_env(
            scene, device=args.device, num_envs=self._num_envs, instruction_type=instruction_type
        )
        self._control_dt = self._env_cfg.sim.dt * self._env_cfg.decimation
        # The sim context's physics ``step`` and ``render`` are the two native phases inside ``env.step``;
        # re-wrapped per build (a fresh env brings a fresh SimulationContext) so their spans nest under the
        # step. Only under --timing — the wrapper is skipped otherwise so an untimed run pays nothing.
        if telemetry.active():
            self._env.sim.step = self._timed_phase(self._env.sim.step, 'physics')
            self._env.sim.render = self._timed_phase(self._env.sim.render, 'render')
        # ``META_INSTRUCTION`` is the instruction text in the phrasing of the reset: ``create_env`` keeps the selected
        # phrasings as a ``{phrasing: text}`` dict. The rest names the task.
        self._meta = {
            keys.META_INSTRUCTION: resolve_instruction(self._env_cfg.instruction, instruction_type),
            'env_name': env_name,
            'task_name': self._env_cfg._task_name,
            'attributes': list(self._env_cfg._task_attributes),
        }
        self._robot = self._env.scene['robot']
        self._bind_robot()
        self._world = get_world(self._env)
        self._objects = [name for name in self._env_cfg.contact_object_list if name != self._TABLE]
        # None for a task without subtasks: RoboLab adds no term for it.
        self._subtask_recorder = self._env.recorder_manager.get_term(SubtaskCompletionRecorderTerm)

    @abstractmethod
    def _register(self, task: str) -> None:
        """Register the RoboLab env of ``task`` with this server's robot and cameras."""

    @abstractmethod
    def _bind_robot(self) -> None:
        """Read the handles of the robot from the built env: ``_joint_ids``, ``_grip``, ``_gripper`` and the robot's
        own."""

    @abstractmethod
    def _robot_meta(self) -> dict[str, Any]:
        """The ``robot_meta`` of a reset."""

    @abstractmethod
    def _action(self, actions: dict[int, dict[str, Any]]) -> torch.Tensor:
        """The ``(num_envs, action values)`` action that the env steps, from each slot's wire action. It also updates
        ``_grip``."""

    @abstractmethod
    def _arm_observations(self, proprio: dict[str, Any]) -> list[dict[str, Any]]:
        """Per slot, the arm fields of the observation payload."""

    def _end_episode(self) -> None:
        """Report what the episode that ends has to report."""

    def _nonfinite_arms(self) -> torch.Tensor:
        """Per slot, whether its arm joints or joint velocities hold a value that is not finite, ``(num_envs,)``."""
        joints = torch.cat([self._measured_q(), self._robot.data.joint_vel[:, self._joint_ids]], dim=1)
        return ~torch.isfinite(joints).all(dim=1)

    @telemetry.traced(telemetry.SPAN_ENV_RESET)
    def reset(self, token: dict[str, Any], slots: int | None = None) -> dict[str, Any]:
        num_envs = self._max_envs if slots is None else slots
        if not 1 <= num_envs <= self._max_envs:
            raise ValueError(f'{num_envs} scene clones asked for, but this server steps 1 to {self._max_envs}')
        self._end_episode()
        self._episode += 1
        self._episode_task = token[keys.TOKEN_TASK]
        key = (
            token[keys.TOKEN_TASK],
            token[keys.TOKEN_INSTRUCTION_TYPE],
            token.get(keys.TOKEN_EPISODE_LENGTH_S),
            num_envs,
        )
        if key != self._key:
            self._build(*key)
            self._key = key
        else:
            assert self._env is not None  # the else runs only when a prior reset built the cached env
            self._env.reset_eval_state()  # unfreeze terminated envs so the next trial runs on the cached env
        assert self._env is not None  # built above (key change) or already live (cached)
        self._grip.zero_()
        self._shown_chains = [None] * self._num_envs
        # RoboLab's own episode runner resets twice — the first randomizes, the second settles the freshly
        # placed scene into the sensors (robolab/eval/episode.py).
        self._env.reset()
        obs, _extras = self._env.reset()
        state = token.get(keys.TOKEN_STATE)
        if state is not None:
            # Exact-state replay (a demo's own scene). RoboLab records states env-origin relative
            # (``scene.get_state(is_relative=True)``), so restore them the same way. The recorded entries
            # overlay the scene's own post-reset state: ``reset_to`` wants every scene entity, and a
            # recording may predate entities that never move (e.g. the table joined the scene's rigid
            # objects for contact sensing after the shipped demos were recorded).
            merged = self._merged_state(self._env.scene.get_state(is_relative=True), state)
            obs, _extras = self._env.reset_to(merged, None, is_relative=True)
        self._faulted = self._nonfinite_arms()
        # Zero progress: the recorder's infos still hold the prior trial until a step refreshes them.
        observed = self._observe(obs, np.zeros((self._num_envs, 4), dtype=np.float32), reset=True)
        return {
            protocol.SLOTS: [{protocol.FRAME_OBS: o} for o in observed],
            protocol.FRAME_META: self._meta,
            protocol.FRAME_ROBOT_META: self._robot_meta(),
            protocol.FRAME_CONTROL_DT: self._control_dt,
        }

    def _frozen_slots(self) -> list[bool]:
        """Per slot, whether RoboLab froze it — its episode ended and its verdict is recorded."""
        active = set(self._env.active_env_ids)
        return [slot not in active for slot in range(self._num_envs)]

    @telemetry.traced(telemetry.SPAN_ENV_STEP)
    def step(self, actions: dict[int, dict[str, Any]]) -> dict[str, Any]:
        assert self._env is not None  # step is only served after a reset built the env
        # Isaac pauses its timeline while assets stream in; stepping a paused sim stalls, so pump the kit
        # update loop until it plays again (robolab's episode loop does the same before every step).
        while not self._timeline.is_playing():
            self._kit_app.update()
        obs, _reward, _term, _trunc, _info = self._env.step(self._action(actions))
        self._faulted |= self._nonfinite_arms()
        # ``done``/``success`` key off RoboLab's frozen-env accounting, not the raw term/trunc flags: a
        # termination within the first two steps is a physics artifact its env resets in place and keeps
        # running (never frozen, no verdict), while a real success/time-out freezes the env and records one. A
        # faulted slot is done too, with no success.
        frozen = self._frozen_slots()
        faulted = self._faulted.tolist()
        results = self._env.get_env_results()
        observed = self._observe(obs, self._subtask_progress(), reset=False)
        return {
            protocol.SLOTS: [
                {
                    protocol.FRAME_OBS: observed[slot],
                    protocol.FRAME_DONE: frozen[slot] or faulted[slot],
                    protocol.FRAME_SUCCESS: frozen[slot] and not faulted[slot] and bool(results[slot]['success']),
                    **({keys.FRAME_FAULT: keys.FAULT_NONFINITE_ARM} if faulted[slot] else {}),
                }
                for slot in range(self._num_envs)
            ],
            protocol.FRAME_CONTROL_DT: self._control_dt,
        }

    def _measured_q(self) -> torch.Tensor:
        """Every slot's measured arm joints, ``(num_envs, len(_joint_ids))``."""
        return self._robot.data.joint_pos[:, self._joint_ids]

    def _merged_state(self, base: Any, overlay: Any) -> Any:
        """The scene's live state tree with the recording's entries written over it, as device tensors.

        A recording holds one scene, so each recorded leaf is copied into the row of every clone.
        """
        if not isinstance(base, dict):
            return torch.as_tensor(overlay, dtype=base.dtype, device=self._env.device).expand_as(base).clone()
        return {
            key: self._merged_state(value, overlay[key]) if key in overlay else value for key, value in base.items()
        }

    def _subtask_progress(self) -> np.ndarray:
        """Every slot's live subtask progress as the wire ``[status, completed, total, score]`` vector,
        ``(num_envs, 4)``. A width RoboLab reports for some other number of slots raises here."""
        subtask = np.zeros((self._num_envs, 4), dtype=np.float32)
        if self._subtask_recorder is not None:
            infos = self._subtask_recorder.infos
            subtask[:] = [(info['status'], info['completed'], info['total'], info['score']) for info in infos]
        return subtask

    @telemetry.traced('gt_state')
    def _gt_state(self, *, reset: bool) -> list[dict[str, Any]]:
        """Every slot's ``keys.GT_STATE`` entry: a flat dict from a recorded suffix to its value."""
        states: list[dict[str, Any]] = [{} for _ in range(self._num_envs)]
        for name in self._objects:
            pos, quat = self._world.get_pose(name, is_relative=True)
            poses = torch.cat([pos, quat], dim=-1).cpu().numpy().astype(np.float32)
            poses[:, :3] += self._ENV_IN_WORLD
            for slot, pose in enumerate(poses):
                states[slot][f'.{keys.GT_STATE_OBJECTS}.{name}.{keys.GT_STATE_POSE}'] = pose
        for slot, state in enumerate(states):
            # On a reset frame the contact sensor still holds the prior trial's last physics step.
            touched = (
                set()
                if reset
                else set(self._world.get_objects_in_contact_with(self._gripper, self._objects, env_id=slot))
            )
            for name in self._objects:
                state[f'.{keys.GT_STATE_OBJECTS}.{name}.{keys.GT_STATE_GRIPPER_CONTACT}'] = name in touched
            state.update(self._progress(slot))
        return states

    def _progress(self, slot: int) -> dict[str, bool]:
        """Per condition of each chain in the slot's current subtask, whether RoboLab counts it done.

        RoboLab replaces a subtask's chain machine in the step that completes the subtask. So the frame of that step
        also reports the machine that RoboLab replaced. Empty for a task without subtasks.
        """
        if self._subtask_recorder is None:
            return {}
        current = self._subtask_recorder.subtask_state_machines[slot].conditionals_state_machine
        shown, self._shown_chains[slot] = self._shown_chains[slot], current
        progress = {}
        for chains in {id(c): c for c in (shown, current) if c is not None}.values():
            progress.update(self._chain_progress(chains))
        return progress

    @staticmethod
    def _chain_progress(chains: Any) -> dict[str, bool]:
        """Per condition of each chain that one RoboLab chain machine tracks, whether it is done."""
        prefix = f'.{keys.GT_STATE_PROGRESS}.{chains.subtask_group_id}'
        progress = {}
        for group, done in chains.object_completed_table.items():
            for index, ((condition, _score), condition_done) in enumerate(
                zip(chains.subtask.get_group(group), done, strict=True)
            ):
                name, _params = get_callable_info(condition)
                progress[f'{prefix}.{group}.{index}.{name}'] = bool(condition_done)
        return progress

    def _observe(self, obs: dict[str, Any], subtask: np.ndarray, *, reset: bool) -> list[dict[str, Any]]:
        """One observation payload per slot."""
        images = obs['image_obs']
        arms = self._arm_observations(obs['proprio_obs'])
        gt_states = self._gt_state(reset=reset)
        return [
            {
                **arms[slot],
                **{name: frame[slot].cpu().numpy() for name, frame in images.items()},
                keys.SUBTASK: subtask[slot],
                keys.GT_STATE: gt_states[slot],
            }
            for slot in range(self._num_envs)
        ]

    def close(self) -> None:
        self._end_episode()
        if self._env is not None:
            self._env.close()
            self._env = None


class DroidEnv(RobolabEnv):
    """The DROID rig in a RoboLab task: one Franka arm with a Robotiq 2F-85 gripper, and the 8-dim jointpos action."""

    def __init__(self, max_envs: int = 1):
        super().__init__(max_envs)
        self._franka_meta = _load_robot_meta()
        self._body_idx = None
        self._link0_idx = None
        self._eef_frame_idx = None
        self._eef_offset_rot = None
        self._far_past_limit_solves: list[int] = []  # per slot, this episode's IK solves with a joint far past a limit
        presets = _camera_presets(args.task_lib)
        if args.cameras not in presets:
            raise ValueError(f'camera set {args.cameras!r} needs a task library whose cameras.py places it')
        self._cameras = presets[args.cameras]

    def _register(self, task: str) -> None:
        # A library task registers by its file path; RoboLab names the env by the Task class in that file.
        with _no_table_fixture(task in self._catalog.library):
            auto_register_droid_envs(task=[self._catalog.register_target(task)], cameras=self._cameras)

    def _bind_robot(self) -> None:
        robot = self._robot
        self._joint_ids = [i for i, name in enumerate(robot.data.joint_names) if name.startswith('panda_joint')]
        self._body_idx = robot.data.body_names.index('base_link')  # the Robotiq gripper mount flange
        self._link0_idx = robot.data.body_names.index('panda_link0')
        self._eef_frame_idx = self._env.scene['frames'].data.target_frame_names.index('eef_frame')
        # One row per slot: isaaclab's quat ops require both operands to share a shape.
        self._eef_offset_rot = torch.tensor(
            [EEF_OFFSET_ROT] * self._num_envs, dtype=torch.float32, device=self._env.device
        )
        self._grip = torch.zeros(self._num_envs, dtype=torch.float32, device=self._env.device)
        (self._gripper,) = self._env_cfg.contact_gripper
        self._far_past_limit_solves = [0] * self._num_envs

    def _robot_meta(self) -> dict[str, Any]:
        # The DROID rig's model, which the launcher serialized, since this server cannot build it.
        return self._franka_meta

    def _end_episode(self) -> None:
        """Print each slot's count of this episode's IK solves with a joint far past a limit, then zero the counts."""
        for slot, solves in enumerate(self._far_past_limit_solves):
            if solves:
                print(
                    f'{self._episode_task} episode {self._episode} slot {slot}: {solves} IK solves with a joint '
                    f'more than {pose_ik.MAX_JOINT_STEP} rad past a limit',
                    file=sys.stderr,
                )
        self._far_past_limit_solves = [0] * self._num_envs

    def _action(self, actions: dict[int, dict[str, Any]]) -> torch.Tensor:
        wires = {slot: protocol.single_arm(action) for slot, action in actions.items()}
        for slot, wire in wires.items():
            grip = float(wire[protocol.TARGET_GRIP])
            if math.isfinite(grip):  # the rig refuses a grip that is not finite, so the slot keeps its last grip
                self._grip[slot] = grip
        act = torch.zeros(self._num_envs, 8, device=self._env.device)
        act[:, :7] = self._joint_targets({slot: wire[protocol.ROBOT_COMMAND] for slot, wire in wires.items()})
        # The binary gripper term closes above 0.5, so the wire grip ([0, 1], 1 = closed) feeds it as-is.
        act[:, 7] = self._grip
        return act

    def _arm_observations(self, proprio: dict[str, Any]) -> list[dict[str, Any]]:
        # The eef pose is reported in the control frame Cartesian commands arrive in (``eef_frame``: env-local
        # position, world quat wxyz), so the observed and commanded pose share a frame. The proprio group has
        # no joint velocities, so they read straight from the articulation.
        joint_vel = self._robot.data.joint_vel[:, self._joint_ids].cpu().numpy()
        return [
            {
                protocol.OBS_JOINT_POS: proprio['arm_joint_pos'][slot].cpu().numpy(),
                protocol.OBS_JOINT_VEL: joint_vel[slot],
                protocol.OBS_EEF_POS: proprio['eef_pos'][slot].cpu().numpy(),
                protocol.OBS_EEF_QUAT: proprio['eef_quat'][slot].cpu().numpy(),
                protocol.OBS_GRIP: float(proprio['gripper_pos'][slot, 0]),
            }
            for slot in range(self._num_envs)
        ]

    def _joint_targets(self, commands: dict[int, dict[str, Any]]) -> torch.Tensor:
        """The ``(num_envs, 7)`` absolute joint targets the substrate steps, from each slot's wire command.

        A slot with no command, or with a command whose targets are not finite, holds its measured joints.
        """
        measured = self._measured_q()
        targets = measured.clone()  # what ``hold`` commands, and the neutral row for every other case
        cur_pos, cur_quat = self._eef_pose()
        pose_pos, pose_quat = cur_pos.clone(), cur_quat.clone()
        pose_slots = []
        for slot, command in commands.items():
            match command[protocol.COMMAND_TYPE]:
                case protocol.JOINT_POS:  # pass through untouched — bit-identical to RoboLab's own leaderboard stack
                    targets[slot] = torch.as_tensor(
                        command[protocol.COMMAND_JOINT_POS], dtype=torch.float32, device=self._env.device
                    )
                case protocol.JOINT_DELTA:  # a per-step joint delta (positronic applies ``JointDelta`` as q + dq)
                    dq = torch.as_tensor(
                        command[protocol.COMMAND_JOINT_DELTA], dtype=torch.float32, device=self._env.device
                    )
                    targets[slot] = measured[slot] + dq
                case protocol.HOLD:  # re-command the measured joints; a zero action drives the arm to zero
                    pass
                case protocol.CARTESIAN:
                    pose_pos[slot], pose_quat[slot] = self._wire_pose(command[protocol.COMMAND_POSE])
                    pose_slots.append(slot)
                case protocol.CARTESIAN_DELTA:
                    delta_pos, delta_quat = self._wire_pose(command[protocol.COMMAND_DELTA])
                    # The world-frame compose of positronic's ``CartesianDelta.apply``: translation adds and
                    # rotation left-multiplies onto the measured eef pose.
                    pose_pos[slot] = cur_pos[slot] + delta_pos
                    pose_quat[slot] = quat_mul(delta_quat.unsqueeze(0), cur_quat[slot].unsqueeze(0)).squeeze(0)
                    pose_slots.append(slot)
                case other:
                    raise ValueError(f'unknown command type {other!r}')
        if pose_slots:
            # ``pose_ik`` cannot solve from joints or towards a pose that is not finite, so such a slot holds.
            finite = torch.isfinite(pose_pos).all(dim=1) & torch.isfinite(pose_quat).all(dim=1) & ~self._faulted
            solvable = [slot for slot in pose_slots if finite[slot]]
            if solvable:
                targets[solvable] = self._solve_ik(pose_pos, pose_quat, solvable)
        # The rig's driver refuses a command that is not finite. Holding it here keeps the clone's physics finite.
        # The row of a faulted slot stays not finite, because its measured joints are not finite.
        refused = ~torch.isfinite(targets).all(dim=1)
        targets[refused] = measured[refused]
        return targets

    def _solve_ik(self, target_pos: torch.Tensor, target_quat: torch.Tensor, slots: list[int]) -> torch.Tensor:
        """The ``(len(slots), 7)`` joint targets of ``pose_ik`` towards each slot's eef-control-frame pose.

        The solve tracks base_link in the panda_link0 frame. The commanded eef_frame orientation is un-offset first
        (``q_base = q_eef ⊗ R_offset⁻¹``, the ``run_abs_ik_demo.py`` conversion; positions are shared — the offset
        translation is zero). The flange-to-base_link transform comes from the measured joints and base_link pose.
        """
        robot = self._robot
        link0_pos_w = robot.data.body_pos_w[:, self._link0_idx]
        link0_quat_w = robot.data.body_quat_w[:, self._link0_idx]
        target = self._link0_poses(
            link0_pos_w,
            link0_quat_w,
            target_pos + self._env.scene.env_origins,
            quat_mul(target_quat, quat_inv(self._eef_offset_rot)),
        )
        measured = self._link0_poses(
            link0_pos_w,
            link0_quat_w,
            robot.data.body_pos_w[:, self._body_idx],
            robot.data.body_quat_w[:, self._body_idx],
        )
        joint_pos = self._measured_q().double().cpu().numpy()
        limits = robot.data.joint_pos_limits[:, self._joint_ids].double().cpu().numpy()
        targets = []
        for slot in slots:
            flange_to_tool = np.linalg.inv(pose_ik.panda_flange(joint_pos[slot])) @ measured[slot]
            lower, upper = limits[slot, :, 0], limits[slot, :, 1]
            past = np.maximum(np.maximum(lower - joint_pos[slot], joint_pos[slot] - upper), 0.0)
            if past.max() > pose_ik.MAX_JOINT_STEP:
                # The solve clamps such a joint to its limit. Print it, so that a physics fault stays visible.
                self._far_past_limit_solves[slot] += 1
                if self._far_past_limit_solves[slot] == 1:
                    print(
                        f'{self._episode_task} episode {self._episode} slot {slot}: joints '
                        f'{np.round(past, 3).tolist()} rad past their limits; the count of such solves follows at the '
                        'episode end',
                        file=sys.stderr,
                    )
            targets.append(pose_ik.joint_targets(target[slot], joint_pos[slot], flange_to_tool, lower, upper))
        return torch.as_tensor(np.stack(targets), dtype=torch.float32, device=self._env.device)

    @staticmethod
    def _link0_poses(link0_pos_w: torch.Tensor, link0_quat_w: torch.Tensor, pos_w: Any, quat_w: Any) -> np.ndarray:
        """World poses as ``(num_envs, 4, 4)`` poses in the panda_link0 frame."""
        pos, quat = subtract_frame_transforms(link0_pos_w, link0_quat_w, pos_w, quat_w)
        poses = np.tile(np.eye(4), (len(pos), 1, 1))
        poses[:, :3, :3] = matrix_from_quat(quat).double().cpu().numpy()
        poses[:, :3, 3] = pos.double().cpu().numpy()
        return poses

    def _eef_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Every slot's measured eef control-frame pose: env-local position ``(num_envs, 3)``, world-frame
        quat wxyz ``(num_envs, 4)``."""
        frames = self._env.scene['frames'].data
        pos = frames.target_pos_w[:, self._eef_frame_idx] - self._env.scene.env_origins
        return pos, frames.target_quat_w[:, self._eef_frame_idx]

    def _wire_pose(self, vec: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """One slot's flat ``[translation(3), rotation_matrix(9)]`` wire pose as ``(pos (3,), quat (4,))``."""
        vec = torch.as_tensor(vec, dtype=torch.float32, device=self._env.device)
        return vec[:3], quat_from_matrix(vec[3:].reshape(1, 3, 3)).squeeze(0)


class YamEnv(RobolabEnv):
    """The two-arm YAM robot of ``yam_spec`` in a RoboLab task, with the 14-value joint position action.

    The robot root is at the env origin, so the server adds ``yam_spec.ROOT_IN_RIG_WORLD`` to every position.
    """

    _ENV_IN_WORLD = np.asarray(yam_spec.ROOT_IN_RIG_WORLD, dtype=np.float32)

    def __init__(self, max_envs: int = 1):
        super().__init__(max_envs)
        import yam_robolab  # noqa: PLC0415 -- the YAM configs, which a DROID server does not use

        # The robot USD is converted once per server, into a directory of its own.
        robot_usd, stand_usd = yam_robolab.convert(Path(tempfile.mkdtemp(prefix='robolab-yam-')))
        self._register_envs = functools.partial(yam_robolab.register_envs, robot_usd=robot_usd, stand_usd=stand_usd)
        self._pose_refusals = yam_spec.PoseRefusals(0)

    def _register(self, task: str) -> None:
        self._register_envs([task])

    def _bind_robot(self) -> None:
        names = self._robot.data.joint_names
        self._joint_ids = [names.index(f'{side}_{joint}') for side in yam_spec.SIDES for joint in yam_spec.ARM_JOINTS]
        self._grip = torch.zeros(self._num_envs, len(yam_spec.SIDES), dtype=torch.float32, device=self._env.device)
        self._gripper = yam_spec.CONTACT_GRIPPERS
        self._pose_refusals = yam_spec.PoseRefusals(self._num_envs)

    def _robot_meta(self) -> dict[str, Any]:
        # The client holds the arm model, so the meta holds only where each arm stands.
        return {
            protocol.MOUNTS: {
                protocol.arm_channel(protocol.ROBOT_STATE, side) + protocol.JOINTS_SUFFIX: [
                    float(v) for v in np.add(yam_spec.mount_xyz(side), self._ENV_IN_WORLD)
                ]
                for side in yam_spec.SIDES
            }
        }

    def _action(self, actions: dict[int, dict[str, Any]]) -> torch.Tensor:
        # Every slot's measured joints per arm, (num_envs, arms, 6).
        measured = self._measured_q().cpu().numpy().reshape(self._num_envs, len(yam_spec.SIDES), -1)
        # A slot with no wire action holds its measured joints and its grips.
        rows = np.concatenate([measured, self._grip.cpu().numpy()[:, :, None]], axis=2).reshape(self._num_envs, -1)
        for slot, wire in actions.items():
            act = yam_spec.action(wire, measured[slot], self._grip[slot].cpu().numpy())
            rows[slot] = act.values
            self._grip[slot] = torch.as_tensor(act.grips, device=self._env.device)
            for warning in self._pose_refusals.add(slot, act.pose_refused):
                print(f'{self._episode_task} episode {self._episode} {warning}', file=sys.stderr)
        return torch.as_tensor(rows, dtype=torch.float32, device=self._env.device)

    def _end_episode(self) -> None:
        """Print each arm's count of this episode's steps that held on a pose command, then zero the counts."""
        for line in self._pose_refusals.report():
            print(f'{self._episode_task} episode {self._episode} {line}', file=sys.stderr)
        self._pose_refusals = yam_spec.PoseRefusals(self._num_envs)

    def _arm_observations(self, proprio: dict[str, Any]) -> list[dict[str, Any]]:
        # The proprio group holds no joint velocities, so they come from the articulation.
        joint_vel = self._robot.data.joint_vel[:, self._joint_ids].cpu().numpy()
        joint_vel = joint_vel.reshape(self._num_envs, len(yam_spec.SIDES), -1)
        observations: list[dict[str, Any]] = [{} for _ in range(self._num_envs)]
        for arm, side in enumerate(yam_spec.SIDES):
            terms = (
                yam_spec.PROPRIO_JOINT_POS,
                yam_spec.PROPRIO_GRIP,
                yam_spec.PROPRIO_EE_POS,
                yam_spec.PROPRIO_EE_QUAT,
            )
            joint_pos, grip, ee_pos, ee_quat = (proprio[yam_spec.proprio_term(side, t)].cpu().numpy() for t in terms)
            ee_pos = ee_pos + self._ENV_IN_WORLD
            for slot, observation in enumerate(observations):
                observation.update({
                    protocol.arm_channel(protocol.OBS_JOINT_POS, side): joint_pos[slot],
                    protocol.arm_channel(protocol.OBS_JOINT_VEL, side): joint_vel[slot, arm],
                    protocol.arm_channel(protocol.OBS_EEF_POS, side): ee_pos[slot],
                    protocol.arm_channel(protocol.OBS_EEF_QUAT, side): ee_quat[slot],
                    protocol.arm_channel(protocol.OBS_GRIP, side): float(grip[slot, 0]),
                })
        return observations


def main() -> None:
    if args.port is None:
        parser.error('--port is required')
    # Under --timing the parent forwards the telemetry dir + run id via the environment; the server then
    # writes its own ``env.spans.jsonl`` sidecar. Inert otherwise.
    with telemetry.bind_from_env():
        env = {keys.DROID: DroidEnv, keys.YAM: YamEnv}[args.robot](args.max_envs)
        EnvServer(env, args.host, args.port, join_deadline=args.join_deadline).serve_forever()
    simulation_app.close()


if __name__ == '__main__':
    main()
