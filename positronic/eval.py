from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pimm
from positronic import keys
from positronic.dataset.serializers import Serializer

# Embodiment-level static meta: how recorded signals map to the canonical robot fields.
ROBOT_STATIC_META = {keys.JOINT_SIGNALS: [keys.JOINTS], keys.POSE_SIGNALS: [keys.EE_POSE, keys.TARGET_EE_POSE]}


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


class Rollout:
    """One rollout: the goal the policy is told, the budget it runs under, and the scene it runs in.

    ``instruction`` is the language goal the policy conditions on, resolved live on every read: an embodiment
    that only learns its goal on reset (a remote env reporting it in meta) passes a source callable, a fixed
    scenario a plain string, and a rollout whose operator named no goal passes ``None``. ``timeout`` is the
    budget in seconds (sim-time for a simulated embodiment, wall-clock for a real one); ``None`` leaves the
    rollout unbounded, so only a directive ends it.

    ``scene`` is what the eval's ``reset`` reads to stage this rollout — a seed, a benchmark suite and task
    id, whatever the reset actuator needs. It is recorded with the episode and never fed to the policy, so a
    policy cannot condition on the ground truth its rollout is scored against.
    """

    def __init__(
        self,
        instruction: str | Callable[[], str] | None = None,
        timeout: float | None = None,
        scene: dict[str, Any] | None = None,
    ):
        self._instruction = instruction if callable(instruction) else (lambda: instruction)
        self.timeout = timeout
        self.scene = scene or {}

    @property
    def instruction(self) -> str | None:
        return self._instruction()


@dataclass
class Eval:
    """One embodiment, the rollouts to run on it, and the scene wiring they share.

    An eval config returns a list of these — one per embodiment — so a benchmark spanning several sims is a
    single selection. ``rollouts`` is the plan the self-driving Harness works through; ``None`` leaves
    the lifecycle to an operator's directives.

    For a sim eval the config holds the shared ``MujocoSim`` the embodiment and the wiring below are both
    built from, so the embodiment stays pure robot. ``reset`` stages a rollout's scene from its ``scene``
    payload; ``None`` on a real embodiment, where a human stages it. ``privileged`` maps a record key to the
    ground-truth source to capture (the sim's full ``save_state``, a real scale) — recorded but never fed to
    the policy. ``done`` is the terminating signal: a source delivering a dict payload when a rollout ends,
    which the Harness reads to stop early and records into the episode's static data.
    """

    embodiment: Embodiment
    rollouts: list[Rollout] | None = None
    reset: Callable[[dict[str, Any]], None] | None = None
    privileged: dict[str, Observation] = field(default_factory=dict)
    done: pimm.SignalEmitter | None = None
