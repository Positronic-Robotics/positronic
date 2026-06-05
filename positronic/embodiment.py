from dataclasses import dataclass
from typing import Any

import pimm
from positronic.dataset.serializers import Serializer, Serializers
from positronic.drivers.roboarm import command as roboarm_command

# Embodiment-level static meta: how recorded signals map to the canonical robot fields.
ROBOT_STATIC_META = {'joint_signal': 'robot_state.q', 'pose_signals': ['robot_state.ee_pose', 'robot_commands.pose']}

# Sentinel for ``Observation.to_record``: "record with the same serializer as the policy sees".
_SAME_AS_POLICY = object()


@dataclass
class Observation:
    """A policy-facing signal source and how it serializes to the policy and to disk.

    Two serializers because the policy and the dataset sometimes need *different*
    encodings of the same device value: ``to_policy`` builds the observation entries
    fed to the policy, ``to_record`` the entries written to the dataset. Today only
    ``robot_state`` actually differs — the obs side keeps the raw ``RobotStatus`` that
    ``ErrorRecovery`` matches on, while the record side drops ``RESETTING`` and emits
    ``.error``. Everywhere else the two coincide, so ``to_record`` defaults to
    ``to_policy`` and is passed explicitly only for that one split. ``None`` on either
    side passes the device value through unchanged.

    TODO: both serializers (and most of this class's reason to exist) collapse once
    serialization is type-owned (steps 7-9): the value's domain type will own its
    policy- and dataset-side encoding, so the channel won't carry serializers at all.
    """

    source: pimm.SignalEmitter
    to_policy: Serializer | None
    to_record: Serializer | None = _SAME_AS_POLICY

    def __post_init__(self):
        if self.to_record is _SAME_AS_POLICY:
            self.to_record = self.to_policy


@dataclass
class Command:
    """A policy action channel: where its waypoints go and how it homes/records.

    ``home`` is the value emitted to send this channel to its safe state. ``record_name``
    is the dataset signal name (it may differ from the action key, e.g. the ``robot_command``
    action records under ``robot_commands``). ``to_record`` serializes the channel's values.
    """

    dest: pimm.SignalReceiver
    home: Any
    record_name: str
    to_record: Serializer | None


@dataclass
class Privileged:
    """A ground-truth signal source, recorded but never fed to the policy."""

    source: pimm.SignalEmitter
    to_record: Serializer | None


@dataclass
class Embodiment:
    """The signal-dict contract the Harness drives, produced by a factory.

    Backed by 1 or N device control systems (not fused) — what satisfies the
    contract is an implementation detail. The Harness is name-free: it reads the
    serializers and command spec from here and never hardcodes a canonical key.
    """

    descriptor: str
    observations: dict[str, Observation]
    commands: dict[str, Command]
    privileged: dict[str, Privileged]
    static_meta: dict[str, Any]
    meta_source: pimm.SignalEmitter | None

    @property
    def home(self) -> dict[str, Any]:
        """The home action: ``{command_name: home_value}`` for every channel."""
        return {name: cmd.home for name, cmd in self.commands.items()}


def franka(
    robot_arm: pimm.ControlSystem,
    gripper: pimm.ControlSystem,
    *,
    descriptor: str,
    cameras: dict[str, pimm.SignalEmitter] | None = None,
    privileged: dict[str, pimm.SignalEmitter] | None = None,
    static_meta: dict[str, Any] | None = None,
) -> Embodiment:
    """Build a single-arm Franka + gripper embodiment from separate device CSs.

    Shared by the sim, real, and golden inference paths — they share the arm/gripper
    signal interface (``state``/``commands``/``robot_meta`` and ``grip``/``target_grip``).
    """
    observations = {
        # robot_state is the one channel whose policy/record encodings differ (status vs .error).
        'robot_state': Observation(robot_arm.state, Serializers.robot_state_obs, Serializers.robot_state),
        'grip': Observation(gripper.grip, None),
    }
    for name, emitter in (cameras or {}).items():
        observations[name] = Observation(emitter, Serializers.camera_images)

    commands = {
        'robot_command': Command(
            robot_arm.commands, roboarm_command.Reset(), 'robot_commands', Serializers.robot_command
        ),
        'target_grip': Command(gripper.target_grip, 0.0, 'target_grip', None),
    }

    privileged_specs = {name: Privileged(source, None) for name, source in (privileged or {}).items()}

    return Embodiment(
        descriptor=descriptor,
        observations=observations,
        commands=commands,
        privileged=privileged_specs,
        static_meta={**ROBOT_STATIC_META, **(static_meta or {})},
        meta_source=robot_arm.robot_meta,
    )
