from collections.abc import Callable
from dataclasses import dataclass, field
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


@dataclass
class Task:
    """The scenario layered on an embodiment: the policy-facing instruction plus
    the privileged ground-truth signals to record.

    ``instruction`` is the language goal sent to the policy on every call. ``timeout``
    is the per-trial time budget in seconds (sim-time for simulated embodiments,
    wall-clock for real). ``privileged`` maps a record key to the ground-truth source
    to capture (the sim's full ``save_state``, a real scale) — recorded but never fed
    to the policy.

    ``seed`` is the base seed for reproducible runs (``None`` → fully random); the
    runner derives per-trial seeds from it into the trial plan. ``reset`` re-randomizes
    the scene for a new trial from a per-trial seed; ``None`` on real embodiments,
    where reset is physical/human.
    """

    instruction: str
    timeout: float
    privileged: dict[str, Observation] = field(default_factory=dict)
    seed: int | None = None
    reset: Callable[[int | None], None] | None = None


@dataclass
class Eval:
    """An eval = embodiment + task, produced by a single config.

    For a sim eval that config holds the shared ``MujocoSim`` both are built from, so the
    embodiment stays pure robot while the task carries the scene's privileged signals.
    """

    embodiment: Embodiment
    task: Task
