"""Pure MolmoSpaces <-> positronic-wire mappings, free of both molmo_spaces and positronic.

Imported from two interpreters: the client-side ``MolmoAdapter`` (positronic) resolves camera keys with
it, and the molmo-venv ``env.py`` builds its raw observation payload and decodes wire commands with it. It
imports numpy plus the positronic-free ``protocol`` (which owns the wire command tags), so it loads under a
bare pytest and inside the molmo venv alike — the fixture tests exercise it without either framework. The
MuJoCo reads that need the live model (joint velocities, the end-effector world pose) stay in ``env.py``;
only the framework-independent arithmetic lives here.
"""

from collections.abc import Callable
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

# Where the MolmoSpaces asset packs live.
ASSETS_DIR_ENV = 'MLSPACES_ASSETS_DIR'

# The reset token: which benchmark episode to build, and the seed overriding the episode spec's own.
TOKEN_EPISODE_INDEX = 'episode_index'
TOKEN_SEED = 'seed'

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
    match command['type']:
        case protocol.JOINT_POS:
            target = np.asarray(command['q'], dtype=np.float32).reshape(-1)
        case protocol.JOINT_VEL:
            dq = np.asarray(command['dq'], dtype=np.float32).reshape(-1)
            if dq.shape[0] != current.shape[0]:
                raise ValueError(f'joint delta {dq.shape[0]} vs measured joints {current.shape[0]}')
            target = current + dq
        case protocol.HOLD:
            target = current
        case protocol.CARTESIAN:
            solver = _require_ik(ik, protocol.CARTESIAN)
            target = np.asarray(solver(*unpack_wire_pose(command['pose'])), dtype=np.float32).reshape(-1)
        case protocol.CARTESIAN_DELTA:
            solver = _require_ik(ik, protocol.CARTESIAN_DELTA)
            if current_eef is None:
                raise ValueError(f'command {protocol.CARTESIAN_DELTA!r} needs the measured eef pose; none supplied')
            delta_pos, delta_rot = unpack_wire_pose(command['delta'])
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


def resolve_task_horizon_steps(episode: Any, policy_dt_ms: float, override_steps: int | None = None) -> int:
    """A benchmark episode's enforced horizon in policy steps, mirroring MolmoSpaces' own precedence.

    An explicit ``override_steps`` wins — the way MolmoSpaces' ``--task_horizon_steps`` overrides the benchmark —
    so an operator can pin the exact horizon a reference run used. Otherwise read the benchmark's own
    ``task_horizon_sec`` (sim-seconds): shipped DROID benchmarks carry it as an **episode-level** field (a
    pydantic extra on the loaded spec — DROID Pick = 20 s), so read it there first, falling back to the task dict
    for layouts that nest it, and convert with ``round(sec * 1000 / policy_dt_ms)`` (MolmoSpaces' own conversion).
    The horizon is part of the task definition, so a benchmark carrying none and no override fails loud.

    Discrepancy worth knowing: MolmoSpaces' own ``determine_task_horizon`` reads only ``episode.task``, which is
    absent on the shipped benchmarks, so it raises there and a native run needs ``--task_horizon_steps`` (or a
    ``patch_benchmarks`` pass that moves the field into ``task``, defaulting PickTask to 20 s). Reading the
    episode-level field reproduces the benchmark's declared horizon without that override.
    """
    if override_steps is not None:
        return override_steps
    horizon_sec = getattr(episode, 'task_horizon_sec', None)
    if horizon_sec is None:
        horizon_sec = episode.task.get('task_horizon_sec')
    if horizon_sec is None:
        raise ValueError(
            'benchmark episode carries no task_horizon_sec (neither episode-level nor in its task) and no '
            'task_horizon_steps override — the horizon is part of the task definition; add it or pass an override'
        )
    return round(horizon_sec * 1000.0 / policy_dt_ms)
