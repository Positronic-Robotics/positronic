"""Pure MolmoSpaces <-> positronic-wire mappings, free of both molmo_spaces and positronic.

Imported from two interpreters: the client-side ``MolmoAdapter`` (positronic) reads the raw payload and builds
the reset token with it, and the molmo-venv ``env.py`` builds that payload and decodes wire commands with it. It
imports numpy plus the positronic-free ``protocol`` (which owns the wire command tags), so it loads under a
bare pytest and inside the molmo venv alike — the fixture tests exercise it without either framework. The
MuJoCo reads that need the live model (joint velocities, the end-effector world pose) stay in ``env.py``;
only the framework-independent arithmetic lives here.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

# ``protocol`` lands as a package on the positronic side and flat on ``PYTHONPATH`` inside the molmo venv,
# where ``positronic`` is not installed — the same two-shape import ``server`` uses.
try:
    from positronic.simulator.env_server import protocol
except ImportError:
    import protocol  # pyright: ignore[reportMissingImports]

# The move groups a reset token's action names on the DROID rig.
MOLMO_ARM_GROUP = 'arm'
MOLMO_GRIPPER_GROUP = 'gripper'

# The MolmoSpaces site the arm move group's leaf frame resolves to, and so the frame this adoption reports
# poses in and resolves Cartesian targets against. The eval declares its recorded model's control frame at the
# same physical point, and ``env.py`` checks the live scene against this name so the two cannot drift apart.
# A scene prefixes every model name with the robot's namespace (``robot_0/``), so the live name ends with this.
MOLMO_GRASP_SITE = 'gripper/grasp_site'

# Where the MolmoSpaces asset packs live, and the subdirectory of that root holding the benchmarks.
ASSETS_DIR_ENV = 'MLSPACES_ASSETS_DIR'
ASSETS_BENCHMARKS_DIR = 'benchmarks'

# A benchmark dir's manifest: the JSON list of episode specs ``load_all_episodes`` reads, and what marks a
# directory as a benchmark for discovery.
MOLMO_BENCHMARK_MANIFEST = 'benchmark.json'


class BenchmarkPath(NamedTuple):
    """A benchmark's place under the asset packs' ``benchmarks/``: the four directory levels MolmoSpaces lays
    its benchmarks out in. The field names are the dimensions an eval spec pins, the keys a task record and a
    reset token carry, and the order of the path segments."""

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
    """Every benchmark under the asset packs, by the manifest that marks it."""
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


# The reset token: the benchmark under ``BenchmarkPath._fields``, the episode within it, and the seed overriding
# the episode spec's own.
TOKEN_EPISODE_INDEX = 'episode_index'
TOKEN_SEED = 'seed'

# The reset frame's scene meta: the episode's resolved language goal and the ProcTHOR house it runs in.
META_TASK = 'task'
META_HOUSE_INDEX = 'house_index'

# The MolmoSpaces observation field holding the per-move-group joint positions, which is where the
# gripper closure is read from.
MOLMO_OBS_QPOS = 'qpos'

# The raw observation payload ``env.py`` reports and ``MolmoAdapter`` reads back.
OBS_JOINT_POS = 'joint_pos'
OBS_JOINT_VEL = 'joint_vel'
OBS_EEF_POS = 'eef_pos'
OBS_EEF_QUAT = 'eef_quat'
OBS_GRIP = 'grip'
OBS_SIM_STATE = 'sim_state'

# The Robotiq 2F-85 finger qpos saturates at this closure; the DROID observation's grip is normalized against
# it into the [0, 1] closure the policy was trained on (molmospaces pi_policy.py:126).
GRIPPER_QPOS_CLOSED = 0.824033

# The Robotiq gripper actuator is a single command, 0 fully open .. 255 fully closed (franka_droid_view.py:43).
ROBOTIQ_CLOSED = 255.0


def is_rgb_frame(value: Any) -> bool:
    """Whether an observation entry is a rendered camera frame — which is how the camera keys are discovered,
    since MolmoSpaces names them per benchmark and the obs dict carries no other HWC uint8 array."""
    return isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[2] == 3 and value.dtype == np.uint8


def normalize_grip_qpos(gripper_qpos: Any) -> float:
    """A Robotiq finger qpos -> the [0, 1] closure the observation reports (0 open, 1 closed)."""
    return float(np.clip(np.asarray(gripper_qpos).reshape(-1)[0] / GRIPPER_QPOS_CLOSED, 0.0, 1.0))


def grip_command_to_actuator(grip: float) -> float:
    """A wire grip closure ([0, 1], 1 = closed) -> the Robotiq actuator command ([0, 255], 255 = closed).

    Continuous: the pi05 codec already binarizes the grip channel (``binarize_grip``), so the rig maps the
    closure straight through rather than re-thresholding it here.
    """
    return float(np.clip(grip, 0.0, 1.0)) * ROBOTIQ_CLOSED


def unpack_wire_pose(vector: Any) -> tuple[np.ndarray, np.ndarray]:
    """A wire pose ``[t(3), R(9)]`` -> ``(translation, 3x3 rotation)``.

    The client encodes every pose with ``Transform3D.as_vector(ROTATION_MATRIX)``: translation first, then the
    rotation matrix row-major.
    """
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    if vec.shape[0] != 12:
        raise ValueError(f'wire pose must be [t(3), R(9)], got {vec.shape[0]} values')
    return vec[:3].copy(), vec[3:].reshape(3, 3).copy()


def compose_world_delta(cur_pos: Any, cur_rot: Any, delta_pos: Any, delta_rot: Any) -> tuple[np.ndarray, np.ndarray]:
    """The absolute pose a world-frame ``cartesian_delta`` targets from a measured pose.

    Translation adds in the world frame and rotation left-multiplies (``goal_ori = R(delta) @ ee_ori``) — the
    convention positronic's ``apply_cartesian_delta`` and LIBERO's own delta bridging both use.
    """
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
    """A tagged wire command + the live measured arm joints -> the 7 absolute joint targets molmo steps.

    This is where the adoption covers the canonical command contract: MolmoSpaces' Franka natively takes only
    joint-position targets, so every canonical type is converted into one. ``joint_pos`` passes through,
    ``joint_vel`` integrates the per-step delta onto the measured joints (positronic applies ``JointDelta`` as
    ``q + dq``), and ``hold`` re-commands the measured joints.

    The Cartesian pair needs the live model, which this module deliberately does not hold: ``env.py`` supplies
    ``ik`` (an absolute world target ``(pos, rot)`` -> joint targets) and the measured ``current_eef`` pose a
    delta composes onto.
    """
    current = np.asarray(current_q, dtype=np.float32).reshape(-1)
    match command[protocol.COMMAND_TYPE]:
        case protocol.JOINT_POS:
            target = np.asarray(command[protocol.COMMAND_JOINT_POS], dtype=np.float32).reshape(-1)
        case protocol.JOINT_VEL:
            dq = np.asarray(command[protocol.COMMAND_JOINT_VEL], dtype=np.float32).reshape(-1)
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
    """The seed an episode runs under, mirroring MolmoSpaces' own precedence.

    An explicit ``override_seed`` wins, then the episode spec's own seed. A spec carrying none falls back to
    the episode index, which is what ``JsonEvalRunner.get_episode_seed`` does — a constant instead would put
    every unseeded episode of a benchmark on one random stream, and none of them on the native one.
    """
    if override_seed is not None:
        return int(override_seed)
    spec_seed = getattr(episode, 'seed', None)
    return int(spec_seed) if spec_seed is not None else int(episode_index)
