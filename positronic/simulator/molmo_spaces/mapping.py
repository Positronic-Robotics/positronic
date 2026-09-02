"""Pure MolmoSpaces <-> positronic-wire mappings, free of both molmo_spaces and positronic.

Imported from two interpreters: the client-side ``MolmoAdapter`` (positronic) resolves camera keys with
it, and the molmo-venv ``env.py`` builds its raw observation payload and decodes wire commands with it. It
imports numpy plus the positronic-free ``protocol`` (which owns the wire command tags), so it loads under a
bare pytest and inside the molmo venv alike — the fixture tests exercise it without either framework. The
MuJoCo reads that need the live model (joint velocities, the end-effector world pose) stay in ``env.py``;
only the framework-independent arithmetic lives here.
"""

import sys
from collections.abc import Callable, Iterable
from typing import Any, TypeAlias

import numpy as np

# ``protocol`` lands as a package on the positronic side and flat on ``PYTHONPATH`` inside the molmo venv,
# where ``positronic`` is not installed — the same two-shape import ``server`` uses.
try:
    from positronic.simulator.env_server import protocol
except ImportError:
    import protocol  # pyright: ignore[reportMissingImports]

# The DROID rig runs 7 Franka arm joints; the reset token's per-move-group action names them 'arm'/'gripper'.
NUM_ARM_JOINTS = 7

# An absolute world target ``(translation, 3x3 rotation)`` -> the arm joint targets that reach it. Supplied
# by ``env.py``, which holds the model this module deliberately does not.
IkSolver: TypeAlias = Callable[[np.ndarray, np.ndarray], Any]
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

# The env-server subprocess CLI, spelled by the launcher building the command and by ``env.py``'s parser
# declaring it — two interpreters, so a rename that misses one fails at spawn rather than at import.
OPT_BENCHMARK_DIR = '--benchmark_dir'
OPT_TASK_HORIZON_STEPS = '--task_horizon_steps'

# MuJoCo's backend selector, and the backend this adoption asks for. MuJoCo validates the value against the
# host platform and raises on one it does not offer there, so the default follows the platform: EGL is the
# headless-GPU path on Linux, CGL the only context macOS has. A GPU-less Linux box overrides with osmesa.
GL_BACKEND_ENV = 'MUJOCO_GL'
GL_BACKEND_DEFAULT = 'cgl' if sys.platform == 'darwin' else 'egl'

# The reset token: which benchmark episode to build, and the seed overriding the episode spec's own.
TOKEN_EPISODE_INDEX = 'episode_index'
TOKEN_SEED = 'seed'

# The reset frame's scene meta: the episode's resolved language goal and the ProcTHOR house it runs in.
META_TASK = 'task'
META_HOUSE_INDEX = 'house_index'

# The MolmoSpaces observation field holding the per-move-group joint positions, which is where the
# gripper closure is read from.
MOLMO_OBS_QPOS = 'qpos'

# The benchmark episode spec's task definition, and the horizon it declares in sim-seconds.
MOLMO_EPISODE_TASK = 'task'
MOLMO_TASK_HORIZON_SEC = 'task_horizon_sec'

# The raw observation payload ``env.py`` reports and ``MolmoAdapter`` reads back.
OBS_JOINT_POS = 'joint_pos'
OBS_JOINT_VEL = 'joint_vel'
OBS_EEF_POS = 'eef_pos'
OBS_EEF_QUAT = 'eef_quat'
OBS_GRIP = 'grip'
OBS_SIM_STATE = 'sim_state'

# MolmoSpaces DROID rig camera names (FrankaDroidCameraSystem); a benchmark's own variants replace the defaults
# and the adapter resolves them so the default camera_dict works across the benchmarks: the light-randomization
# suite records the exterior as ``droid_shoulder_light_randomization`` (MolmoSpaces' Pi policy prefers it), and
# the RandCam suite records it as ``randomized_zed2_analogue_1`` (its ``--camera_names`` exterior); the Zed wrist
# variant is ``wrist_camera_zed_mini``.
MOLMO_WRIST_CAMERA = 'wrist_camera'
MOLMO_EXTERIOR_CAMERA = 'exo_camera_1'
MOLMO_WRIST_CAMERA_VARIANTS = ('wrist_camera_zed_mini',)
MOLMO_EXTERIOR_CAMERA_VARIANTS = ('droid_shoulder_light_randomization', 'randomized_zed2_analogue_1')

# The Robotiq 2F-85 finger qpos saturates at this closure; the DROID observation's grip is normalized against
# it into the [0, 1] closure the policy was trained on (molmospaces pi_policy.py:126).
GRIPPER_QPOS_CLOSED = 0.824033

# The Robotiq gripper actuator is a single command, 0 fully open .. 255 fully closed (franka_droid_view.py:43).
ROBOTIQ_OPEN = 0.0
ROBOTIQ_CLOSED = 255.0


def is_rgb_frame(value: Any) -> bool:
    """Whether an observation entry is a rendered camera frame — which is how the camera keys are discovered,
    since MolmoSpaces names them per benchmark and the obs dict carries no other HWC uint8 array."""
    return isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[2] == 3 and value.dtype == np.uint8


def normalize_grip_qpos(gripper_qpos: Any, gripper_qpos_closed: float = GRIPPER_QPOS_CLOSED) -> float:
    """A Robotiq finger qpos -> the [0, 1] closure the observation reports (0 open, 1 closed)."""
    value = float(np.asarray(gripper_qpos).reshape(-1)[0])
    return float(np.clip(value / gripper_qpos_closed, 0.0, 1.0))


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


def _require_ik(ik: IkSolver | None, kind: str) -> IkSolver:
    """The caller's IK solver, or a loud failure — a Cartesian target is unresolvable without the live model."""
    if ik is None:
        raise ValueError(f'command {kind!r} needs an ik solver; none was supplied')
    return ik


def wire_command_to_arm_action(
    command: dict[str, Any], current_q: Any, *, ik: IkSolver | None = None, current_eef: tuple[Any, Any] | None = None
) -> np.ndarray:
    """A tagged wire command + the live measured arm joints -> the 7 absolute joint targets molmo steps.

    This is where the adoption covers the canonical command contract: MolmoSpaces' Franka natively takes only
    joint-position targets, so every canonical type is converted into one. ``joint_pos`` passes through,
    ``joint_vel`` integrates the per-step delta onto the measured joints (positronic applies ``JointDelta`` as
    ``q + dq``), and ``hold`` re-commands the measured joints.

    The Cartesian pair needs the live model, which this module deliberately does not hold: the caller passes
    ``ik`` (an absolute world target ``(pos, rot)`` -> joint targets) and, for ``cartesian_delta``, the measured
    ``current_eef`` pose the delta composes onto. Both are supplied by ``env.py``, which owns the sim.
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
            solver = _require_ik(ik, protocol.CARTESIAN)
            target = np.asarray(solver(*unpack_wire_pose(command[protocol.COMMAND_POSE])), dtype=np.float32).reshape(-1)
        case protocol.CARTESIAN_DELTA:
            solver = _require_ik(ik, protocol.CARTESIAN_DELTA)
            if current_eef is None:
                raise ValueError(f'command {protocol.CARTESIAN_DELTA!r} needs the measured eef pose; none supplied')
            delta_pos, delta_rot = unpack_wire_pose(command[protocol.COMMAND_DELTA])
            target_pos, target_rot = compose_world_delta(*current_eef, delta_pos, delta_rot)
            target = np.asarray(solver(target_pos, target_rot), dtype=np.float32).reshape(-1)
        case other:
            raise ValueError(
                f'{other!r} is not a canonical command type; the contract is {list(protocol.CANONICAL_COMMAND_TYPES)}'
            )
    return target.astype(np.float32)


def resolve_camera_key(available: Any, key: str, variants: tuple[str, ...] = ()) -> str:
    """The MolmoSpaces observation key to read for a camera role, mirroring the upstream policy's precedence.

    A present benchmark variant wins over ``key`` (matching molmo_spaces pi_policy); with no variants ``key``
    is read as-is. Raises with the candidate list on a miss.
    """
    keys = set(available)
    for candidate in (*variants, key):
        if candidate in keys:
            return candidate
    raise KeyError(f'observation has none of {(*variants, key)}; available: {sorted(keys)}')


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


def declared_task_horizon_sec(declared: Iterable[float | None]) -> float:
    """The one horizon a benchmark declares, in sim-seconds, over every episode's ``task_horizon_sec``.

    The horizon belongs to the benchmark, not to an episode within it, so a benchmark that declares none, one
    that disagrees with itself, and one that declares a non-positive span all have no horizon to run at.
    Callers pass the values already read from their own representation of the specs: parsed episode objects in
    the molmo venv, raw JSON on the positronic side.
    """
    horizons = set()
    for value in declared:
        if value is None:
            raise ValueError(
                f'benchmark episodes carry no {MOLMO_TASK_HORIZON_SEC} in their task dict — the horizon is part '
                'of the task definition; add it to the benchmark'
            )
        if value <= 0:
            raise ValueError(f'benchmark declares a non-positive {MOLMO_TASK_HORIZON_SEC} of {value}s')
        horizons.add(value)
    if len(horizons) != 1:
        raise ValueError(f'benchmark declares inconsistent {MOLMO_TASK_HORIZON_SEC} values {sorted(horizons)}')
    return float(horizons.pop())


def resolve_task_horizon_steps(episodes: Any, policy_dt_ms: float, override_steps: int | None = None) -> int:
    """A benchmark's enforced horizon in policy steps, mirroring MolmoSpaces' own resolution.

    Upstream's ``determine_task_horizon`` (``evaluation/eval_main.py``, the entrypoint its README documents)
    resolves in this order and nothing else: an explicit ``--task_horizon_steps`` override, then the benchmark's
    own ``task_horizon_sec`` from the episodes' task dicts, converted with ``round(sec * 1000 / policy_dt_ms)``.
    It raises when any episode declares none, and again when the episodes disagree. This reproduces all three,
    raises included: ``JsonBenchmarkEvalConfig.task_horizon``'s 500-step default is a config default upstream
    overwrites before the runner ever sees it. A horizon that resolves below one step is refused on either
    path, so no route reaches the task with a budget it expires inside.
    """
    if override_steps is not None:
        if override_steps < 1:
            raise ValueError(f'task_horizon_steps override must be at least 1 step, got {override_steps}')
        return override_steps
    sec = declared_task_horizon_sec(episode.task.get(MOLMO_TASK_HORIZON_SEC) for episode in episodes)
    steps = round(sec * 1000.0 / policy_dt_ms)
    if steps < 1:
        raise ValueError(
            f'benchmark {MOLMO_TASK_HORIZON_SEC} of {sec}s rounds to {steps} steps at a {policy_dt_ms}ms policy '
            'period — the episode would expire before its first action'
        )
    return steps
