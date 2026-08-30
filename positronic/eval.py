from collections.abc import Callable, Iterable
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
    serializer that returns ``None`` drops the value from both the recording and the
    policy input.

    TODO(#638): the serializer (and most of this class's reason to exist) goes away
    once serialization is type-owned: the value's domain type will own its policy- and
    dataset-side encoding, so the channel won't carry a serializer at all.
    """

    source: pimm.ControlSystemEmitter
    serializer: Serializer | None


@dataclass
class Command:
    """A policy action channel: where its waypoints go, and the serializer that records them under the
    channel's own key."""

    dest: pimm.ControlSystemReceiver
    serializer: Serializer | None


@dataclass
class Embodiment:
    """The signal-dict contract the Harness drives, produced by a factory.

    Backed by 1 or N device control systems. Holds the observation
    serializers (which own the canonical key names), command channels, and what
    a trial readies; the Harness reads these to assemble policy inputs and
    demux actions. ``control_systems`` lists those devices for the runner to
    schedule, and ``simulated`` marks a sim embodiment (virtual clock, in-process
    scheduling).
    """

    descriptor: str
    observations: dict[str, Observation]
    commands: dict[str, Command]
    # Everything a trial readies before it opens, keyed by ``keys.ARM``, ``keys.SCENE`` and the like. One
    # handler serves one caller, so a device backing several command channels appears here once.
    prepare_handlers: dict[str, pimm.calls.ControlSystemHandler[Any, None]]
    static_meta: dict[str, Any]
    meta_source: pimm.ControlSystemEmitter | None
    control_systems: tuple[pimm.ControlSystem, ...] = ()
    simulated: bool = False


@dataclass
class Task:
    """One trial: the goal the policy conditions on, the time budget it runs under, and what sets it up."""

    instruction_source: str | Callable[[], str]
    # Time budget for a rollout; ``None`` ends on ``Eval.done`` alone.
    timeout_sec: float | None
    # What to ask for, keyed as ``Embodiment.prepare_handlers`` is; a handler this does not name goes unasked
    prepare_args: dict[str, Any] = field(default_factory=dict)
    # What the episode records as this trial's identity: its seed, its place in the sweep, its scene
    meta: dict[str, Any] = field(default_factory=dict)
    # Sim-only: a real rig cannot pretend the time is paused when inference is run.
    charge_inference_time: bool = False

    @property
    def instruction(self) -> str:
        """An embodiment that learns its task on reset has one to report only once that reset has run."""
        src = self.instruction_source
        return src if isinstance(src, str) else src()


# rules-allow: stale-doc — a plan is read for what is left, which needs the landed steps marked
# rules-allow: diff-comments — the marks are the plan's state, not a note about an edit
# TODO: one lifecycle for every trial: sim or real, attended or not.
# Roadmap:
# [✓] ``privileged``, ``done`` and ``reset`` are per-run: they move to ``Eval``.
# [✓] ``trials`` becomes ``list[Task]``, one per trial; a ``Task`` carries what ``reset`` is called with.
# [✓] A driver walks the list and calls ``perform_task(task)``; the Harness keeps no plan.
# [✓] ``charge_inference_time`` is a ``Task`` field, not a context key.
# [✓] Split the reset token from the policy input: the instruction is all a trial gives the policy.
# [✓] Homing becomes a ``prepare`` call the arm and gripper answer once in place; ``Command.home`` goes with it.
# [✓] A trial's reset is every ``prepare`` it asks for — scene, arm, gripper — and it opens once all answer.
# [✓] ``command.Reset`` goes: a robot is moved by a ``prepare`` call that names where to go.
# [✓] The recorder keeps what is on the wire when it opens, and a producer publishes the scene as it answers.
# [✓] A config makes each attended trial, so the keyboard path makes none of its own.
# [✓] A task source makes the plan when the world starts.
# [✓] One runner builds the world for both.
# [✓] A session reports the model's meta, so a policy holds none.
# [✓] The episode call carries the session that runs it, so the Harness holds no policy.
# [✓] The episode call names where it records, so the Harness holds no dataset.
# [ ] A benchmark answers ``tasks(spec)``, so positronic holds no task table of its own.
# [ ] A task carries its instruction as data, so no trial reads it after the reset.
# [ ] A sim eval config names a selection, and a spec flag narrows it to one task.
# [ ] A task names its model, and a routed policy serves the mix.
# [ ] A session logs the trials it ran, and a later run replays that log.
@dataclass
class Eval:
    """An eval = embodiment + the tasks to run on it, produced by a single config.

    ``privileged`` and ``done`` are per-run, not per-task: the World wires the signals once, before it runs.
    ``tasks`` makes the sweep — one entry per (scenario, seed). A driver calls it when it starts, so a
    source that asks its env what tasks it has reaches a live one.
    """

    embodiment: Embodiment
    tasks: Callable[[], Iterable[Task]]
    privileged: dict[str, Observation] = field(default_factory=dict)
    done: pimm.ControlSystemEmitter | None = None
