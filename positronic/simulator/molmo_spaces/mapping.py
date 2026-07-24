"""Pure MolmoSpaces <-> positronic-wire mappings, free of both molmo_spaces and positronic.

Imported from two interpreters: the client-side ``MolmoAdapter`` (positronic) resolves camera keys with
it, and the molmo-venv ``env.py`` builds its raw observation payload and decodes wire commands with it. It
imports only numpy, so it loads under a bare pytest and inside the molmo venv alike — the fixture tests
exercise it without either framework. The MuJoCo reads that need the live model (joint velocities, the
end-effector world pose) stay in ``env.py``; only the framework-independent arithmetic lives here.
"""

from typing import Any

import numpy as np

# The DROID rig runs 7 Franka arm joints; the reset token's per-move-group action names them 'arm'/'gripper'.
NUM_ARM_JOINTS = 7
MOLMO_ARM_GROUP = 'arm'
MOLMO_GRIPPER_GROUP = 'gripper'

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


def wire_command_to_arm_action(command: dict[str, Any], current_q: Any) -> np.ndarray:
    """A tagged wire command + the live measured arm joints -> the 7 absolute joint targets molmo steps.

    MolmoSpaces' Franka runs the joint-position controller, so every command resolves to absolute joint
    targets: ``joint_pos`` passes through, ``joint_vel`` integrates the per-step delta onto the measured
    joints (positronic applies ``JointDelta`` as ``q + dq``), and ``hold`` re-commands the measured joints.
    Cartesian commands would need IK against the live model, which this jointpos substrate does not run.
    """
    current = np.asarray(current_q, dtype=np.float32).reshape(-1)
    match command['type']:
        case 'joint_pos':
            target = np.asarray(command['q'], dtype=np.float32).reshape(-1)
        case 'joint_vel':
            dq = np.asarray(command['dq'], dtype=np.float32).reshape(-1)
            if dq.shape[0] != current.shape[0]:
                raise ValueError(f'joint delta {dq.shape[0]} vs measured joints {current.shape[0]}')
            target = current + dq
        case 'hold':
            target = current
        case other:
            raise ValueError(f'MolmoSpaces jointpos substrate cannot map command {other!r}')
    return target.astype(np.float32)


def resolve_camera_key(available: Any, key: str, default: str, variants: tuple[str, ...]) -> str:
    """The MolmoSpaces observation key to read for a camera role, mirroring the upstream policy's precedence.

    An explicitly configured non-default key is read as-is; for the default role a present benchmark-variant
    key wins over the default (matching molmo_spaces pi_policy). Raises with the candidate list on a miss.
    """
    keys = set(available)
    if key != default:
        if key not in keys:
            raise KeyError(f'observation has no camera {key!r}; available: {sorted(keys)}')
        return key
    for candidate in (*variants, key):
        if candidate in keys:
            return candidate
    raise KeyError(f'observation has none of {(*variants, key)}; available: {sorted(keys)}')
