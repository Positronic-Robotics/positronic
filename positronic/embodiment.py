from dataclasses import dataclass
from typing import Any

import pimm
from positronic.dataset.serializers import Serializer

# Embodiment-level static meta: how recorded signals map to the canonical robot fields.
ROBOT_STATIC_META = {'joint_signal': 'robot_state.q', 'pose_signals': ['robot_state.ee_pose', 'robot_command.pose']}


@dataclass
class Observation:
    """A policy-facing signal source and the serializer for its canonical entries.

    The same serializer feeds the policy *and* records the signal — recording is
    canonical policy I/O. ``None`` passes the device value through unchanged. A
    serializer that returns ``None`` for a sample (e.g. ``robot_state`` while the arm
    is ``RESETTING``) means "not ready": that frame is neither fed to the policy nor
    recorded.

    TODO: the serializer (and most of this class's reason to exist) goes away once
    serialization is type-owned (steps 8-9): the value's domain type will own its
    policy- and dataset-side encoding, so the channel won't carry a serializer at all.
    """

    source: pimm.SignalEmitter
    serializer: Serializer | None


@dataclass
class Command:
    """A policy action channel: where its waypoints go and how it homes/records.

    ``home`` is the value emitted to send this channel to its safe state; ``serializer``
    serializes the channel's values, recorded under the channel's own key.
    """

    dest: pimm.SignalReceiver
    home: Any
    serializer: Serializer | None


@dataclass
class Embodiment:
    """The signal-dict contract the Harness drives, produced by a factory.

    Backed by 1 or N device control systems (not fused). Holds the observation
    serializers (which own the canonical key names), command channels, and home
    action; the Harness reads these to assemble policy inputs and demux actions.
    ``control_systems`` lists those devices for the runner to schedule, and
    ``simulated`` marks a sim embodiment (virtual clock, in-process scheduling).
    """

    descriptor: str
    observations: dict[str, Observation]
    commands: dict[str, Command]
    static_meta: dict[str, Any]
    meta_source: pimm.SignalEmitter | None
    control_systems: tuple[pimm.ControlSystem, ...] = ()
    simulated: bool = False

    @property
    def home(self) -> dict[str, Any]:
        """The home action: ``{command_name: home_value}`` for every channel."""
        return {name: cmd.home for name, cmd in self.commands.items()}
