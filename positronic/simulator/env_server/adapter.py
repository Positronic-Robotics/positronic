"""The ``EnvAdapter`` interface: the per-benchmark canonical<->raw mappings, on the client side.

``RemoteEnvControlSystem`` is a dumb translator — it moves data between pimm signals and this adapter.
The adapter is the smart half: it maps the env's own task records into trial params, turns the Harness's
commands into the env's raw action (owning what to hold between them), maps raw observations back to
canonical signals — policy-facing and privileged ground-truth kept separate — and reads the terminal. Each
benchmark ships one adapter (``vendors/``-style); the native ``MujocoSim`` fixture is the reference.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, final

import numpy as np

import pimm
from positronic import geom, keys
from positronic.drivers.roboarm import command as roboarm_command
from positronic.simulator.env_server import protocol


class EnvAdapter(ABC):
    """The mappings between the canonical embodiment contract and an env's raw wire payloads."""

    @abstractmethod
    def task_params(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The env's own task records -> one trial params dict per task; the mirror of ``reset_token``."""

    @abstractmethod
    def reset_token(self, params: dict[str, Any]) -> Any:
        """The environment's reset token for the trial parameters.

        Called at each trial start; clear any per-trial command state here.
        """

    @abstractmethod
    def action(self, commands: dict[str, pimm.Message]) -> dict[str, Any]:
        """The latest per-channel command messages -> the raw action the env steps.

        A channel delivers a command only when one comes due, so the adapter owns what happens in between —
        e.g. hold the last commanded value, the absolute-mode invariant.
        """

    @abstractmethod
    def observations(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        """An env's raw observation payload -> the canonical, policy-facing observation signals."""

    @abstractmethod
    def privileged(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        """An env's raw payload -> the privileged ground-truth signals: recorded, never fed to the policy.

        The split mirrors ``Embodiment.observations`` against ``Eval.privileged``. The env exposes one raw
        payload; the adapter routes ground-truth (full sim state, a real scale) here so it can never reach
        the policy.
        """

    @abstractmethod
    def terminal(self, result: dict[str, Any]) -> dict[str, Any] | None:
        """A ``step`` result -> a non-empty ``done`` payload when the trial has ended, else ``None``.

        ``done`` is truthy-valued: a non-empty payload ends the trial and is recorded into the episode's
        static data; ``None`` or an empty ``{}`` keeps it running. ``{}`` is reserved as the non-terminal
        value ``reset`` republishes to clear the prior trial's terminal.
        """


def _in_env_control_frame(cmd: Any, env_control_frame: geom.Transform3D) -> Any:
    """A command against the embodiment's ``default``, re-expressed in the frame the env measures and drives."""
    match cmd:
        case roboarm_command.CartesianPosition(pose, mode):
            return roboarm_command.CartesianPosition(pose=pose * env_control_frame, mode=mode)
        case roboarm_command.CartesianDelta(delta, frame, mode):
            return roboarm_command.CartesianDelta(delta, env_control_frame.inv * frame, mode)
        case _:
            return cmd


def _wire_command(cmd: Any) -> dict[str, Any]:
    """The held command as a positronic-free payload the server decodes (no ``geom``/``roboarm`` on its side).

    A pinned control mode rides along under ``mode``; whether the env honors it is the env's.
    """
    wire: dict[str, Any]
    rep = geom.Rotation.Representation.ROTATION_MATRIX
    match cmd:
        case roboarm_command.CartesianPosition(pose):
            wire = {protocol.COMMAND_TYPE: protocol.CARTESIAN, protocol.COMMAND_POSE: pose.as_vector(rep)}
        case roboarm_command.JointPosition(positions):
            wire = {protocol.COMMAND_TYPE: protocol.JOINT_POS, protocol.COMMAND_JOINT_POS: positions}
        case roboarm_command.JointDelta(velocities):
            wire = {protocol.COMMAND_TYPE: protocol.JOINT_DELTA, protocol.COMMAND_JOINT_DELTA: velocities}
        case roboarm_command.CartesianDelta(delta, frame):
            # The env anchors a delta on the pose it measures, which is its control frame and nowhere else, so
            # a delta still expressed somewhere else has no faithful wire form.
            if not np.allclose(frame.as_matrix, np.eye(4)):
                raise ValueError('CartesianDelta outside the env control frame cannot be sent to a remote env')
            wire = {protocol.COMMAND_TYPE: protocol.CARTESIAN_DELTA, protocol.COMMAND_DELTA: delta.as_vector(rep)}
        case None:
            return {protocol.COMMAND_TYPE: protocol.HOLD}
        case other:
            raise ValueError(f'no wire encoding for robot_command {type(other).__name__}')
    if cmd.mode is not None:
        wire[protocol.COMMAND_MODE] = roboarm_command.to_wire(cmd.mode)
    return wire


class WireCommandAdapter(EnvAdapter):
    """An adapter whose action is the shared wire payload: one ``{'name', 'command', 'grip'}`` entry per arm.

    The command side of every remote benchmark adapter: it holds an absolute setpoint until the next command
    arrives and fires a relative delta once, and flattens each arm's held command (a pose as ``[t(3), R(9)]``,
    joint positions, or per-step joint deltas) plus that arm's gripper closure into one entry.
    All action *encoding* — how the tagged command becomes the env's native action — stays server-side with
    the env's own model. Subclasses implement ``_reset_token`` (the base clears the per-trial command state
    around it) and keep the task, observation and terminal mappings to themselves.
    """

    def __init__(self, env_control_frame: geom.Transform3D | None = None, arms: Sequence[str | None] = (None,)):
        """``env_control_frame`` places the frame the env measures and drives relative to the embodiment's
        ``default``; the adapter re-expresses outgoing commands into it, and observations back out of it.
        ``arms`` names the arms the env drives, in the order the action lists them; the default is the one
        unnamed arm of a single-arm embodiment, whose channels are the bare ``robot_command``/``target_grip``.
        """
        self.env_control_frame = env_control_frame if env_control_frame is not None else geom.Transform3D.identity
        self.arms = tuple(arms)
        self._reset_command_state()

    def _reset_command_state(self) -> None:
        self._held: dict[str, Any] = {}  # last command per channel — re-sent until the next one arrives

    @final
    def reset_token(self, params: dict[str, Any]) -> Any:
        self._reset_command_state()
        return self._reset_token(params)

    @abstractmethod
    def _reset_token(self, params: dict[str, Any]) -> Any:
        """The trial's params -> the env's opaque reset token; the command state is already cleared."""

    def action(self, commands: dict[str, pimm.Message]) -> dict[str, Any]:
        for name, msg in commands.items():
            if msg.updated:
                self._held[name] = msg.data
        return {protocol.ACTION_ARMS: [self._arm_action(arm) for arm in self.arms]}

    def _arm_action(self, arm: str | None) -> dict[str, Any]:
        """One arm's wire entry, from whatever it holds; an arm nobody has commanded yet holds nothing."""
        # The server maps the held command into its controller's action. A delta — Cartesian or joint — is a
        # one-shot relative motion, forwarded once then dropped: re-sending a stale delta would re-compose it
        # against the moving arm every tick (the eef drifts, or the joints walk toward their limits), so after
        # one step the arm holds its measured pose.
        channel = keys.arm_channel(keys.ROBOT_COMMAND, arm)
        cmd = self._held.get(channel)
        if isinstance(cmd, roboarm_command.CartesianDelta | roboarm_command.JointDelta):
            self._held.pop(channel)
        return {
            protocol.ARM_NAME: arm,
            protocol.ACTION_COMMAND: _wire_command(_in_env_control_frame(cmd, self.env_control_frame)),
            protocol.ACTION_GRIP: float(self._held.get(keys.arm_channel(keys.TARGET_GRIP, arm), 0.0)),
        }
