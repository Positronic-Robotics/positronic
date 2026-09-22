"""The wire command of one arm as absolute joint targets, for an environment server to apply.

This module must work in an isolated interpreter without Positronic installed.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

if __package__:
    from positronic.simulator.env_server import protocol
else:
    import protocol


def unpack_wire_pose(vector: Any) -> tuple[np.ndarray, np.ndarray]:
    """Translation and rotation matrix from a wire pose ``[t(3), R(9) row-major]``."""
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    if vec.shape[0] != 12:
        raise ValueError(f'wire pose must be [t(3), R(9)], got {vec.shape[0]} values')
    return vec[:3].copy(), vec[3:].reshape(3, 3).copy()


def compose_world_delta(cur_pos: Any, cur_rot: Any, delta_pos: Any, delta_rot: Any) -> tuple[np.ndarray, np.ndarray]:
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
    """Absolute arm joint targets; ``ik`` and ``current_eef`` use poses in the frame the env measures."""
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
