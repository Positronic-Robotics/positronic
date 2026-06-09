from dataclasses import dataclass, field

from positronic.embodiment import Embodiment, Observation


@dataclass
class Task:
    """The scenario layered on an embodiment: the policy-facing instruction plus
    the privileged ground-truth signals to record.

    ``instruction`` is the language goal sent to the policy on every call. ``privileged``
    maps a record key to the ground-truth source to capture (the sim's full ``save_state``,
    a real scale) — recorded but never fed to the policy.
    """

    instruction: str
    privileged: dict[str, Observation] = field(default_factory=dict)


@dataclass
class Eval:
    """An eval = embodiment + task, produced by a single config.

    For a sim eval that config holds the shared ``MujocoSim`` both are built from, so the
    embodiment stays pure robot while the task carries the scene's privileged signals.
    """

    embodiment: Embodiment
    task: Task
