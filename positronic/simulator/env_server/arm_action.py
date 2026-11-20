"""The wire command of one arm as absolute joint targets, for an environment server to apply.

This module must work in an isolated interpreter without Positronic installed.
"""

import logging
from collections.abc import Callable
from typing import Any, NamedTuple

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


class ArmTargets(NamedTuple):
    """The absolute joint targets of one arm, and why the env refused its command; ``refusal`` is None for an applied
    command."""

    q: np.ndarray
    refusal: str | None


class _Refused(Exception):
    """A command that the env does not apply; the message is the reason."""


def _joint_values(values: Any, joints: int) -> np.ndarray:
    q = np.asarray(values, dtype=np.float32).reshape(-1)
    if q.size != joints:
        raise _Refused(f'width {q.size} for {joints} joints')
    if not np.isfinite(q).all():
        raise _Refused('a value is not finite')
    return q


def _solved(q: Any, joints: int) -> np.ndarray:
    if q is None:
        raise _Refused('no IK solution')
    return _joint_values(q, joints)


def arm_targets(
    command: dict[str, Any],
    measured_q: Any,
    *,
    ik: Callable[[np.ndarray, np.ndarray], Any],
    read_eef: Callable[[], tuple[Any, Any]],
    arm: str | None = None,
) -> ArmTargets:
    """The absolute joint targets of one arm's wire command, or the measured joints for a command that the env refuses.

    The env refuses a target or a joint delta of the wrong width, a value that is not finite, and a pose that ``ik``
    cannot solve, with one warning that names ``arm``. A command type that is not canonical and a pose that is not 12
    values raise ``ValueError``. ``ik`` and ``read_eef`` use poses in the frame the env measures; only a Cartesian delta
    calls ``read_eef``.
    """
    measured = np.asarray(measured_q, dtype=np.float32).reshape(-1)
    kind = command[protocol.COMMAND_TYPE]
    try:
        match kind:
            case protocol.JOINT_POS:
                q = _joint_values(command[protocol.COMMAND_JOINT_POS], measured.size)
            case protocol.JOINT_DELTA:
                q = measured + _joint_values(command[protocol.COMMAND_JOINT_DELTA], measured.size)
            case protocol.HOLD:
                q = measured
            case protocol.CARTESIAN:
                q = _solved(ik(*unpack_wire_pose(command[protocol.COMMAND_POSE])), measured.size)
            case protocol.CARTESIAN_DELTA:
                delta_pos, delta_rot = unpack_wire_pose(command[protocol.COMMAND_DELTA])
                q = _solved(ik(*compose_reference_delta(*read_eef(), delta_pos, delta_rot)), measured.size)
            case other:
                raise ValueError(
                    f'{other!r} is not a canonical command type; the contract is '
                    f'{list(protocol.CANONICAL_COMMAND_TYPES)}'
                )
    except _Refused as refusal:
        logger.warning('%s command not applied, the %s holds: %s', kind, f'{arm} arm' if arm else 'arm', refusal)
        return ArmTargets(measured, str(refusal))
    return ArmTargets(q, None)
