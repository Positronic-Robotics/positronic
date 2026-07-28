from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pimm
from positronic import keys
from positronic.dataset.serializers import Serializer

# Embodiment-level static meta: how recorded signals map to the canonical robot fields.
ROBOT_STATIC_META = {'joint_signal': keys.JOINTS, 'pose_signals': [keys.EE_POSE, 'robot_command.pose']}

# Per-trial context keys the trial builders set and the sim adapters read from each trial's reset context.
EVAL_SEED = 'eval.seed'
EVAL_EPISODE_INDEX = 'eval.episode_index'
EVAL_TRIAL_INDEX = 'eval.trial_index'
EVAL_TRIAL_COUNT = 'eval.trial_count'


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


class Task:
    """The scenario layered on an embodiment: the policy-facing instruction plus
    the privileged ground-truth signals to record.

    ``instruction`` is the language goal the policy conditions on, resolved live on every read: an embodiment
    that only learns its task on reset (a remote env reporting it in meta) passes a source callable, while a
    fixed scenario passes a plain string (wrapped as a constant). ``timeout`` is the per-trial time budget in
    seconds (sim-time for simulated embodiments, wall-clock for real). ``privileged`` maps a record key to the
    ground-truth source to capture (the sim's full ``save_state``, a real scale) — recorded but never fed to
    the policy.

    Two tiers own the trial's end. For a benchmark sim the env owns the horizon — the task's native horizon is
    part of its definition, and the env enforces it and reports expiry as a terminal ``done``; ``timeout`` is
    then only a runaway-cost safety net, set deliberately longer than any healthy sim horizon so it fires solely
    when the sim never terminates. For a real or attended eval there is no such terminal, so ``timeout`` is the
    actual budget. A ``done`` within budget ends the trial and records ``eval.terminated`` True; the budget
    lapsing records it False, distinguishing a safety-net cutoff from a genuine terminal.

    ``reset`` re-randomizes the scene for a new trial from the per-trial run context, reading the keys it
    needs (e.g. ``eval.seed``, and ``eval.task_id`` for a multi-task suite); ``None`` on real embodiments,
    where reset is physical/human.

    ``done`` is the optional terminating signal: a source that delivers a dict payload when the
    trial ends. The Harness reads it to stop the trial early and records the payload into the
    episode's static data.

    ``horizon`` is the optional trial time budget (sim-seconds) — the sim-enforced episode horizon plus the env's
    own framing overhead — resolved live per trial like ``instruction`` from what a benchmark env reports at reset
    (see the two-tier design above). The Harness reads it once a trial is armed and rejects a ``timeout`` that is
    not strictly longer, so the safety net cannot silently truncate a valid episode. ``None`` for real/attended
    tasks and envs that enforce no horizon.
    """

    def __init__(
        self,
        instruction: str | Callable[[], str],
        timeout: float,
        privileged: dict[str, Observation] | None = None,
        reset: Callable[[dict[str, Any]], None] | None = None,
        done: pimm.SignalEmitter | None = None,
        horizon: Callable[[], float | None] | None = None,
    ):
        self._instruction = (lambda: instruction) if isinstance(instruction, str) else instruction
        self.timeout = timeout
        self.privileged = privileged or {}
        self.reset = reset
        self.done = done
        self._horizon = horizon

    @property
    def instruction(self) -> str:
        return self._instruction()

    @property
    def horizon(self) -> float | None:
        """The env's sim-enforced episode horizon (sim-seconds) for the current trial, or ``None`` if none."""
        return self._horizon() if self._horizon is not None else None


@dataclass
class Eval:
    """An eval = embodiment + task + the trial sweep, produced by a single config.

    For a sim eval that config holds the shared ``MujocoSim`` both are built from, so the
    embodiment stays pure robot while the task carries the scene's privileged signals. ``trials`` is the
    sequence of RUN contexts the self-driving Harness runs — one per (task variant, seed) the config sweeps;
    empty for an attended/real eval, driven by an operator rather than a trial plan.
    """

    embodiment: Embodiment
    task: Task
    trials: list[dict[str, Any]] = field(default_factory=list)
