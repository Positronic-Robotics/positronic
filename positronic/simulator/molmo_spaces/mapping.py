"""Shared benchmark selection and wire conversions for the client and MolmoSpaces server."""

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation

if __package__:
    from positronic.simulator.env_server import protocol
else:
    import protocol

MOLMO_ARM_GROUP = 'arm'
MOLMO_GRIPPER_GROUP = 'gripper'

# The end-effector site; scene models prefix it with the robot namespace (e.g. robot_0/).
MOLMO_GRASP_SITE = 'gripper/grasp_site'

ASSETS_DIR_ENV = 'MLSPACES_ASSETS_DIR'
ASSETS_BENCHMARKS_DIR = 'benchmarks'

MOLMO_BENCHMARK_MANIFEST = 'benchmark.json'  # JSON list of episode specs.


class BenchmarkPath(NamedTuple):
    """Benchmark directory components below ``MLSPACES_ASSETS_DIR/benchmarks``, in path order."""

    suite: str
    scene_dataset: str
    task_config: str
    benchmark: str

    @classmethod
    def parse(cls, relative: str) -> 'BenchmarkPath':
        parts = Path(relative).parts
        if len(parts) != len(cls._fields):
            raise ValueError(f'a benchmark path is {"/".join(cls._fields)}, not {relative!r}')
        return cls(*parts)

    @property
    def relative(self) -> Path:
        return Path(*self)

    def under(self, assets_dir: Path) -> Path:
        return assets_dir / ASSETS_BENCHMARKS_DIR / self.relative


def discover_benchmarks(assets_dir: Path) -> list[BenchmarkPath]:
    """Benchmarks with an episode manifest under the asset directory.

    The walk follows symlinked directories, and skips a link back into one of the directory's own ancestors.
    """
    root = assets_dir / ASSETS_BENCHMARKS_DIR
    found = []
    ancestors = {root: {root.resolve()}}

    def unreadable(error: OSError) -> None:
        raise error

    # `os.walk` rather than `Path.walk`, which needs Python 3.12.
    for dirpath, dirnames, filenames in os.walk(root, onerror=unreadable, followlinks=True):
        here = Path(dirpath)
        above = ancestors.pop(here)
        if MOLMO_BENCHMARK_MANIFEST in filenames:
            found.append(BenchmarkPath.parse(str(here.relative_to(root))))
        descend = []
        for name in dirnames:
            real = (here / name).resolve()
            if real not in above:
                descend.append(name)
                ancestors[here / name] = above | {real}
        dirnames[:] = descend
    return sorted(found)


def select_benchmarks(found: list[BenchmarkPath], spec: dict[str, Any]) -> list[BenchmarkPath]:
    """The benchmarks ``spec`` selects: each dimension is one name, a list of them, or absent (any)."""

    def admits(dimension: str, value: str) -> bool:
        pinned = spec.get(dimension)
        return pinned is None or value == pinned or (not isinstance(pinned, str) and value in pinned)

    selected = [b for b in found if all(admits(d, v) for d, v in zip(BenchmarkPath._fields, b, strict=True))]
    if not selected:
        pinned = {d: spec[d] for d in BenchmarkPath._fields if d in spec}
        available = ', '.join(str(b.relative) for b in found) or 'none'
        raise ValueError(f'no benchmark matches {pinned}; available under {ASSETS_BENCHMARKS_DIR}/: {available}')
    return selected


TOKEN_EPISODE_INDEX = 'episode_index'
TOKEN_SEED = 'seed'

SELECT_EPISODES = 'episodes'
TASK_NAME = 'name'
TASK_HORIZON_SEC = 'task_horizon_sec'

META_TASK = 'task'
META_HOUSE_INDEX = 'house_index'

MOLMO_OBS_QPOS = 'qpos'  # MolmoSpaces joint positions, grouped by robot move group.

OBS_JOINT_POS = 'joint_pos'
OBS_JOINT_VEL = 'joint_vel'
OBS_EEF_POS = 'eef_pos'  # Position in the robot base frame, metres.
OBS_EEF_QUAT = 'eef_quat'  # Orientation in the robot base frame, wxyz.
OBS_GRIP = 'grip'  # Closure in [0, 1].
OBS_SIM_STATE = 'sim_state'  # MuJoCo mjSTATE_INTEGRATION vector.

# The left shoulder camera under its benchmark names, in Pi-policy preference order.
LEFT_SHOULDER_CAMERAS = ('droid_shoulder_light_randomization', 'exo_camera_1')
# The robot's right shoulder camera. No benchmark records it, so the name is positronic's own.
RIGHT_SHOULDER_CAMERA = 'right_shoulder_camera'

# Closed finger joint position used by MolmoSpaces' pi_policy.py to normalize gripper observations.
GRIPPER_QPOS_CLOSED = 0.824033

# Closed actuator command from MolmoSpaces' franka_droid_view.py; 0 is fully open.
ROBOTIQ_CLOSED = 255.0


def is_rgb_frame(value: Any) -> bool:
    """Whether a value is an HWC uint8 RGB image."""
    return isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[2] == 3 and value.dtype == np.uint8


def normalize_grip_qpos(gripper_qpos: Any) -> float:
    """Gripper closure in [0, 1], from the first Robotiq finger's joint position."""
    return float(np.clip(np.asarray(gripper_qpos).reshape(-1)[0] / GRIPPER_QPOS_CLOSED, 0.0, 1.0))


def grip_command_to_actuator(grip: float) -> float:
    """Robotiq actuator command in [0, 255] for a closure in [0, 1]; larger values close the gripper."""
    return float(np.clip(grip, 0.0, 1.0)) * ROBOTIQ_CLOSED


def unpack_wire_pose(vector: Any) -> tuple[np.ndarray, np.ndarray]:
    """Translation and rotation matrix from a wire pose ``[t(3), R(9) row-major]``."""
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    if vec.shape[0] != 12:
        raise ValueError(f'wire pose must be [t(3), R(9)], got {vec.shape[0]} values')
    return vec[:3].copy(), vec[3:].reshape(3, 3).copy()


def compose_reference_delta(
    cur_pos: Any, cur_rot: Any, delta_pos: Any, delta_rot: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a reference-frame translation and rotation delta to a measured pose."""
    return (
        np.asarray(cur_pos, dtype=np.float64).reshape(3) + np.asarray(delta_pos, dtype=np.float64).reshape(3),
        np.asarray(delta_rot, dtype=np.float64).reshape(3, 3) @ np.asarray(cur_rot, dtype=np.float64).reshape(3, 3),
    )


def wire_command_to_arm_action(
    command: dict[str, Any],
    current_q: Any,
    *,
    ik: Callable[[np.ndarray, np.ndarray], Any],
    current_eef: tuple[Any, Any],
) -> np.ndarray | None:
    """Absolute arm joint targets, or None where ``ik`` finds no solution.

    ``ik`` and ``current_eef`` use robot-frame grasp-site poses.
    """
    current = np.asarray(current_q, dtype=np.float32).reshape(-1)
    match command[protocol.COMMAND_TYPE]:
        case protocol.JOINT_POS:
            target = np.asarray(command[protocol.COMMAND_JOINT_POS], dtype=np.float32).reshape(-1)
        case protocol.JOINT_DELTA:
            dq = np.asarray(command[protocol.COMMAND_JOINT_DELTA], dtype=np.float32).reshape(-1)
            if dq.shape[0] != current.shape[0]:
                raise ValueError(f'joint delta {dq.shape[0]} vs measured joints {current.shape[0]}')
            target = current + dq
        case protocol.HOLD:
            target = current
        case protocol.CARTESIAN:
            solution = ik(*unpack_wire_pose(command[protocol.COMMAND_POSE]))
            if solution is None:
                return None
            target = np.asarray(solution, dtype=np.float32).reshape(-1)
        case protocol.CARTESIAN_DELTA:
            delta_pos, delta_rot = unpack_wire_pose(command[protocol.COMMAND_DELTA])
            solution = ik(*compose_reference_delta(*current_eef, delta_pos, delta_rot))
            if solution is None:
                return None
            target = np.asarray(solution, dtype=np.float32).reshape(-1)
        case other:
            raise ValueError(
                f'{other!r} is not a canonical command type; the contract is {list(protocol.CANONICAL_COMMAND_TYPES)}'
            )
    return target.astype(np.float32)


class ArmStep(NamedTuple):
    """The arm's joint target for one step, and whether the step reports an arm error."""

    target: np.ndarray
    error: bool


def _same_command(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a.keys() == b.keys() and all(np.array_equal(a[name], b[name]) for name in a)


class ArmTargets:
    """The arm's joint targets for successive wire commands of one episode.

    A command out of IK reach holds the last target, or, before any target, the joints measured at that refusal. The
    error reads true only on the step a refused command first arrives: a client sends its held command again on each
    step, and an error that never clears pauses the policy.
    """

    def __init__(self) -> None:
        self._target: np.ndarray | None = None
        self._previous: dict[str, Any] | None = None

    def step(
        self,
        command: dict[str, Any],
        current_q: Any,
        *,
        ik: Callable[[np.ndarray, np.ndarray], Any],
        current_eef: tuple[Any, Any],
    ) -> ArmStep:
        repeated = self._previous is not None and _same_command(command, self._previous)
        self._previous = command
        target = wire_command_to_arm_action(command, current_q, ik=ik, current_eef=current_eef)
        if target is None:
            if self._target is None:
                self._target = np.asarray(current_q, dtype=np.float32).reshape(-1)
            return ArmStep(self._target, error=not repeated)
        self._target = target
        return ArmStep(target, error=False)


def resolve_episode_seed(episode: Any, episode_index: int, override_seed: int | None = None) -> int:
    """The episode seed, following MolmoSpaces' ``JsonEvalRunner.get_episode_seed`` convention."""
    if override_seed is not None:
        return int(override_seed)
    spec_seed = getattr(episode, 'seed', None)
    return int(spec_seed) if spec_seed is not None else int(episode_index)


# A reflection across the robot base's XZ plane, and a flip of the camera's own x axis. A camera looks along -Z with +Y
# up, so its view and up directions reflect, and its x axis turns to keep the frame right-handed.
_REFLECT_Y = np.diag([1.0, -1.0, 1.0])
_FLIP_CAMERA_X = np.diag([-1.0, 1.0, 1.0])


def mirror_camera_pose(offset: Any, quat: Any) -> tuple[np.ndarray, np.ndarray]:
    """A robot-mounted camera pose reflected to the other side of the robot base; ``quat`` is wxyz."""
    rotation = Rotation.from_quat(np.asarray(quat, dtype=np.float64), scalar_first=True).as_matrix()
    mirrored = Rotation.from_matrix(_REFLECT_Y @ rotation @ _FLIP_CAMERA_X)
    return _REFLECT_Y @ np.asarray(offset, dtype=np.float64), mirrored.as_quat(scalar_first=True)


def jitter_camera_pose(
    offset: Any,
    quat: Any,
    pos_noise_range: tuple[float, float] | None,
    orientation_noise_degrees: float | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """A robot-mounted camera pose randomized as MolmoSpaces' ``CameraManager`` does it, with draws from ``rng``."""
    offset = np.asarray(offset, dtype=np.float64)
    rotation = Rotation.from_quat(np.asarray(quat, dtype=np.float64), scalar_first=True)
    if pos_noise_range is not None:
        offset = offset + rng.uniform(pos_noise_range[0], pos_noise_range[1], size=3)
    if orientation_noise_degrees is not None:
        euler = rng.uniform(-orientation_noise_degrees, orientation_noise_degrees, size=3)
        rotation = Rotation.from_euler('xyz', euler, degrees=True) * rotation
    return offset, rotation.as_quat(scalar_first=True)


def episode_rng(bench: BenchmarkPath, episode_index: int) -> np.random.Generator:
    """A generator keyed by the episode's place in the asset packs alone; no run or trial seed changes its draws."""
    digest = hashlib.sha256(f'{bench.relative.as_posix()}#{episode_index}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'little'))


# `positronic_evals.molmo_spaces` holds the list, and the sim server's interpreter cannot import that package.
def load_right_view_blocked(path: Path) -> dict[BenchmarkPath, frozenset[int]]:
    """The blocked episodes of every swept benchmark, from the right-view blocked list at ``path``."""
    return {BenchmarkPath.parse(rel): frozenset(indices) for rel, indices in json.loads(path.read_text()).items()}


def select_episodes(bench: BenchmarkPath, count: int, pinned: list[int] | None, blocked: frozenset[int]) -> list[int]:
    """Every episode outside ``blocked`` when ``pinned`` is None, else ``pinned``, which must avoid ``blocked``."""
    if pinned is None:
        return [i for i in range(count) if i not in blocked]
    out_of_range = [i for i in pinned if not 0 <= i < count]
    if out_of_range:
        raise ValueError(f'episodes {out_of_range} out of range for the {count} episodes of {bench.relative}')
    hidden = sorted(set(pinned) & blocked)
    if hidden:
        raise ValueError(
            f'episodes {hidden} of {bench.relative} hide the start grasp site from the right shoulder camera; '
            'select others (the right-view blocked list names them)'
        )
    return list(pinned)
