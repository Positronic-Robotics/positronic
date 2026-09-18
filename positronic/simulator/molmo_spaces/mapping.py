"""Shared benchmark selection and wire conversions for the client and MolmoSpaces server."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

if __package__:
    from positronic.simulator.env_server import protocol
else:
    import protocol  # pyright: ignore[reportMissingImports] -- supplied on the isolated server's PYTHONPATH.

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
    """Benchmarks with an episode manifest under the asset directory."""
    root = assets_dir / ASSETS_BENCHMARKS_DIR
    return [BenchmarkPath.parse(str(p.parent.relative_to(root))) for p in sorted(root.rglob(MOLMO_BENCHMARK_MANIFEST))]


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
OBS_EEF_POS = 'eef_pos'  # World coordinates, metres.
OBS_EEF_QUAT = 'eef_quat'  # World orientation, wxyz.
OBS_GRIP = 'grip'  # Closure in [0, 1].
OBS_SIM_STATE = 'sim_state'  # MuJoCo mjSTATE_INTEGRATION vector.

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


def compose_world_delta(cur_pos: Any, cur_rot: Any, delta_pos: Any, delta_rot: Any) -> tuple[np.ndarray, np.ndarray]:
    """Apply a world-frame translation and rotation delta to a measured pose."""
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
) -> np.ndarray:
    """Absolute arm joint targets; ``ik`` and ``current_eef`` use world-frame grasp-site poses."""
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
            target = np.asarray(ik(*unpack_wire_pose(command[protocol.COMMAND_POSE])), dtype=np.float32).reshape(-1)
        case protocol.CARTESIAN_DELTA:
            delta_pos, delta_rot = unpack_wire_pose(command[protocol.COMMAND_DELTA])
            target_pos, target_rot = compose_world_delta(*current_eef, delta_pos, delta_rot)
            target = np.asarray(ik(target_pos, target_rot), dtype=np.float32).reshape(-1)
        case other:
            raise ValueError(
                f'{other!r} is not a canonical command type; the contract is {list(protocol.CANONICAL_COMMAND_TYPES)}'
            )
    return target.astype(np.float32)


def resolve_episode_seed(episode: Any, episode_index: int, override_seed: int | None = None) -> int:
    """The episode seed, following MolmoSpaces' ``JsonEvalRunner.get_episode_seed`` convention."""
    if override_seed is not None:
        return int(override_seed)
    spec_seed = getattr(episode, 'seed', None)
    return int(spec_seed) if spec_seed is not None else int(episode_index)
