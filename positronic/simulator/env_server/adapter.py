"""The ``EnvAdapter`` interface: the per-benchmark canonical<->raw mappings, on the client side.

``RemoteEnvControlSystem`` is a dumb translator — it moves data between pimm signals and this adapter.
The adapter is the smart half: it turns the Harness's command trajectories into the env's raw action
(owning trajectory playing and how to hold between waypoints), maps raw observations back to canonical
signals — policy-facing and privileged ground-truth kept separate — and reads the terminal. Each
benchmark ships one adapter (``vendors/``-style); the native ``MujocoSim`` fixture is the reference.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import pimm
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
    def reset_token(self, seed: int | None) -> Any:
        """A per-trial seed -> the env's opaque reset token (an int for most, a blob for exact replay).

        Called at each trial start, so it is also where the adapter clears any per-trial command state.
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
