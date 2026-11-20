"""The wire command of one arm as absolute joint targets, for an environment server to apply.

This module must work in an isolated interpreter without Positronic installed.
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

if __package__:
    from positronic.simulator.env_server import protocol
else:
    import protocol

logger = logging.getLogger(__name__)


def unpack_wire_pose(vector: Any) -> tuple[np.ndarray, np.ndarray]:
    """Translation and rotation matrix from a wire pose ``[t(3), R(9) row-major]``."""
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    if vec.shape[0] != 12:
        raise ValueError(f'wire pose must be [t(3), R(9)], got {vec.shape[0]} values')
    return vec[:3].copy(), vec[3:].reshape(3, 3).copy()


def compose_reference_delta(
    cur_pos: Any, cur_rot: Any, delta_pos: Any, delta_rot: Any
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(cur_pos, dtype=np.float64).reshape(3) + np.asarray(delta_pos, dtype=np.float64).reshape(3),
        np.asarray(delta_rot, dtype=np.float64).reshape(3, 3) @ np.asarray(cur_rot, dtype=np.float64).reshape(3, 3),
    )


def wire_command_to_arm_action(
    command: dict[str, Any],
    current_q: Any,
    *,
    ik: Callable[[np.ndarray, np.ndarray], Any],
    current_eef: Callable[[], tuple[Any, Any]],
) -> np.ndarray | None:
    """Absolute arm joint targets, or None where ``ik`` finds no solution.

    ``ik`` and ``current_eef`` use poses in the frame the env measures. Only a Cartesian delta reads ``current_eef``.
    """
    current = np.asarray(current_q, dtype=np.float32).reshape(-1)
    match command[protocol.COMMAND_TYPE]:
        case protocol.JOINT_POS:
            target = command[protocol.COMMAND_JOINT_POS]
        case protocol.JOINT_DELTA:
            dq = np.asarray(command[protocol.COMMAND_JOINT_DELTA], dtype=np.float32).reshape(-1)
            if dq.shape[0] != current.shape[0]:
                raise ValueError(f'joint delta {dq.shape[0]} vs measured joints {current.shape[0]}')
            target = current + dq
        case protocol.HOLD:
            target = current
        case protocol.CARTESIAN:
            target = ik(*unpack_wire_pose(command[protocol.COMMAND_POSE]))
        case protocol.CARTESIAN_DELTA:
            delta_pos, delta_rot = unpack_wire_pose(command[protocol.COMMAND_DELTA])
            target = ik(*compose_reference_delta(*current_eef(), delta_pos, delta_rot))
        case other:
            raise ValueError(
                f'{other!r} is not a canonical command type; the contract is {list(protocol.CANONICAL_COMMAND_TYPES)}'
            )
    return None if target is None else np.asarray(target, dtype=np.float32).reshape(-1)


def arm_targets_or_hold(
    command: dict[str, Any],
    measured_q: Any,
    *,
    ik: Callable[[np.ndarray, np.ndarray], Any],
    current_eef: Callable[[], tuple[Any, Any]],
) -> np.ndarray:
    """Absolute arm joint targets, or the measured joints and an error log for a command that the env refuses.

    The env refuses a joint delta or a target of the wrong width, a target that is not finite, and a pose that ``ik``
    cannot solve. The arm holds, so a bad command fails the task and not the run.
    """
    measured = np.asarray(measured_q, dtype=np.float32).reshape(-1)
    kind = command[protocol.COMMAND_TYPE]
    deltas = np.asarray(command[protocol.COMMAND_JOINT_DELTA]).size if kind == protocol.JOINT_DELTA else measured.size
    target = (
        None
        if deltas != measured.size
        else wire_command_to_arm_action(command, measured, ik=ik, current_eef=current_eef)
    )
    if target is None:
        refusal = 'no IK solution' if deltas == measured.size else f'{deltas} joint deltas for {measured.size} joints'
    elif target.shape != measured.shape:
        refusal = f'{target.size} joint targets for {measured.size} joints'
    elif not np.isfinite(target).all():
        refusal = 'a joint target is not finite'
    else:
        return target
    logger.error('%s command not applied, the arm holds: %s', kind, refusal)
    return measured
