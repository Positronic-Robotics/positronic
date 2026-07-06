"""The ``EnvAdapter`` interface: the per-benchmark canonical<->raw mappings, on the client side.

``RemoteEnvControlSystem`` is a dumb translator — it moves data between pimm signals and this adapter.
The adapter is the smart half: it turns the Harness's command trajectories into the env's raw action
(owning trajectory playing and how to hold between waypoints), maps raw observations back to canonical
signals — policy-facing and privileged ground-truth kept separate — and reads the terminal. Each
benchmark ships one adapter (``vendors/``-style); the native ``MujocoSim`` fixture is the reference.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, final

import pimm
from positronic import geom
from positronic.drivers.roboarm import command as roboarm_command


def fresh_command_players() -> defaultdict[str, roboarm_command.TrajectoryPlayer]:
    """A trajectory player per command channel: ``robot_command`` accumulates the deltas due in one tick (a
    missed tick catches up instead of dropping motion), every other channel keeps the last value due."""
    players = defaultdict(roboarm_command.TrajectoryPlayer)
    players['robot_command'] = roboarm_command.TrajectoryPlayer(reduce=roboarm_command.reduce)
    return players


class EnvAdapter(ABC):
    """The mappings between the canonical embodiment contract and an env's raw wire payloads."""

    @abstractmethod
    def reset_token(self, context: dict[str, Any]) -> Any:
        """The per-trial RUN context -> the env's opaque reset token (an int for most, a blob for exact replay).

        Reads the context keys it needs (e.g. ``eval.seed``, ``eval.task_id``). Called at each trial start, so
        it is also where the adapter clears any per-trial command state.
        """

    @abstractmethod
    def action(self, commands: dict[str, pimm.Message], now_ns: int) -> dict[str, Any]:
        """The latest per-channel command messages + the clock -> the raw action the env steps.

        The adapter owns trajectory playing (sampling each channel's waypoints down to ``now_ns``) and
        what to do between waypoints — e.g. hold the last commanded value, the absolute-mode invariant.
        """

    @abstractmethod
    def observations(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        """An env's raw observation payload -> the canonical, policy-facing observation signals."""

    @abstractmethod
    def privileged(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        """An env's raw payload -> the privileged ground-truth signals: recorded, never fed to the policy.

        The split mirrors the Task's observations/privileged. The env exposes one raw payload; the adapter
        routes ground-truth (full sim state, a real scale) here so it can never reach the policy.
        """

    @abstractmethod
    def terminal(self, result: dict[str, Any]) -> dict[str, Any] | None:
        """A ``step`` result -> a non-empty ``done`` payload when the trial has ended, else ``None``.

        ``done`` is truthy-valued: a non-empty payload ends the trial and is recorded into the episode's
        static data; ``None`` or an empty ``{}`` keeps it running. ``{}`` is reserved as the non-terminal
        value ``reset`` republishes to clear the prior trial's terminal.
        """


def _wire_command(cmd: Any) -> dict[str, Any]:
    """The held command as a positronic-free payload the server decodes (no ``geom``/``roboarm`` on its side)."""
    match cmd:
        case roboarm_command.CartesianPosition(pose):
            return {'type': 'cartesian', 'pose': pose.as_vector(geom.Rotation.Representation.ROTATION_MATRIX)}
        case roboarm_command.JointPosition(positions):
            return {'type': 'joint_pos', 'q': positions}
        case roboarm_command.JointDelta(velocities):
            return {'type': 'joint_vel', 'dq': velocities}
        case roboarm_command.CartesianDelta(delta):
            return {'type': 'cartesian_delta', 'delta': delta.as_vector(geom.Rotation.Representation.ROTATION_MATRIX)}
        case None:
            return {'type': 'hold'}
        case other:
            raise ValueError(f'no wire encoding for robot_command {type(other).__name__}')


class WireCommandAdapter(EnvAdapter):
    """An adapter whose action is the shared wire payload ``{'command': <tagged dict>, 'grip': float}``.

    The command side of every remote benchmark adapter: it plays each command channel's trajectory down to
    the clock, holds the last waypoint between waypoints, and flattens the held arm command (a pose as
    ``[t(3), R(9)]``, joint positions, or per-step joint deltas) plus the gripper closure into one payload.
    All action *encoding* — how the tagged command becomes the env's native action — stays server-side with
    the env's own model. Subclasses implement ``_reset_token`` (the base clears the per-trial command state
    around it) and keep the observation and terminal mappings to themselves.
    """

    def __init__(self):
        self._reset_command_state()

    def _reset_command_state(self) -> None:
        self._players = fresh_command_players()
        self._held: dict[str, Any] = {}  # last sampled waypoint per channel — re-sent until it changes
        # Last commanded gripper closure, held across a cancelled grip trajectory: grip is an absolute [0, 1]
        # value with no 'hold' command to fall back on (unlike the arm), so cancelling must freeze it, not reopen.
        self._grip = 0.0

    @final
    def reset_token(self, context: dict[str, Any]) -> Any:
        self._reset_command_state()
        return self._reset_token(context)

    @abstractmethod
    def _reset_token(self, context: dict[str, Any]) -> Any:
        """The per-trial RUN context -> the env's opaque reset token; the command state is already cleared."""

    def action(self, commands: dict[str, pimm.Message], now_ns: int) -> dict[str, Any]:
        for name, msg in commands.items():
            player = self._players[name]
            if msg.updated and msg.data is not None:
                player.set(msg.data)
                if not msg.data:  # an empty trajectory cancels: stop replaying the held waypoint
                    self._held.pop(name, None)
            value = player.advance(now_ns)
            if value is not None:
                self._held[name] = value
        # The server maps the held command into its controller's action. Reset has no env-side
        # action, so it forwards as a hold; a CartesianDelta is a one-shot relative motion, forwarded once
        # then dropped so the held command never re-composes it against the moving eef.
        cmd = self._held.get('robot_command')
        match cmd:
            case roboarm_command.Reset():
                self._held.pop('robot_command')
                cmd = None
            case roboarm_command.CartesianDelta():
                self._held.pop('robot_command')
        if 'target_grip' in self._held:
            self._grip = float(self._held['target_grip'])
        return {'command': _wire_command(cmd), 'grip': self._grip}
