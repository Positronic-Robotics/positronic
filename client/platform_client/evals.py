"""The one name a caller picks a run by, and the definitions of the public evals.

A name is a task suite AND the embodiment that runs it, so there is no second axis to get wrong.
The platform owns the set, so a name this client has never heard of still reaches the server.

The platform holds the definitions of the evals it does not publish.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler, JsonValue, PositiveInt
from pydantic_core import core_schema


class EvalRef(str):
    """The name of an eval the platform offers, as a validated `str`."""

    __slots__ = ()

    def __new__(cls, value: str) -> EvalRef:
        if not value or any(c.isspace() for c in value):
            raise ValueError(f'not an eval name: {value!r}')
        return super().__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(cls, source: type, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(cls, core_schema.str_schema())


class ScorerRef(str):
    """The name of the rule that scores a run's recorded episodes, as a validated `str`."""

    __slots__ = ()

    def __new__(cls, value: str) -> ScorerRef:
        if not value or any(c.isspace() for c in value):
            raise ValueError(f'not a scorer name: {value!r}')
        return super().__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(cls, source: type, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(cls, core_schema.str_schema())


_DEFINITION_CONFIG = ConfigDict(extra='forbid', frozen=True)


class EvalTask(BaseModel):
    """One task of an eval, and its trials in run order."""

    model_config = _DEFINITION_CONFIG

    name: str = Field(min_length=1)
    # Each trial's initial condition, in the keys of the config that runs it.
    trials: list[dict[str, JsonValue]] = Field(min_length=1)


class EvalDefinition(BaseModel):
    """What a run of a named eval runs."""

    model_config = _DEFINITION_CONFIG

    # The positronic eval config, as `positronic eval run --eval=` takes it.
    config: str = Field(pattern=r'^\.[A-Za-z_][A-Za-z0-9_.]*$')
    # The config's `--eval.<name>=` overrides, which expand to `tasks`.
    args: dict[str, JsonValue]
    tasks: list[EvalTask] = Field(min_length=1)
    # The wall-clock limit of one platform run. A local run does not stop at it.
    time_limit_s: PositiveInt
    scorer: ScorerRef
    # The positronic commit the platform runs the config at.
    positronic_revision: str = Field(pattern=r'^[0-9a-f]{40}$')


# The MolmoSpaces evals: the config they name, its task and trial key as positronic spells them
# (`positronic/simulator/molmo_spaces/keys.py`), and the positronic commit the platform's sim image holds.
MOLMO_CONFIG = '.sim.molmo.benchmarks'
MOLMO_TASK = 'franka_pick_droid_mini'
MOLMO_EPISODE_INDEX_KEY = 'molmo.episode_index'
MOLMO_SCORER = ScorerRef('molmo')
MOLMO_POSITRONIC_REVISION = 'a9e13e8a1fbe624ff62b49e25628b39859619986'


def _franka_pick_mini(episodes: list[int]) -> EvalDefinition:
    """`episodes` of the one benchmark the MolmoSpaces evals run, once each."""
    return EvalDefinition(
        config=MOLMO_CONFIG,
        # Parameters of positronic's `benchmarks` config, which `positronic/cli/eval/tests/test_run.py` checks.
        args={
            'suite': 'molmospaces-bench-v2',
            'scene_dataset': 'procthor-10k',
            'task_config': 'FrankaPickDroidMiniBench',
            'benchmark': 'FrankaPickDroidMiniBench_json_benchmark_20251231',
            'episodes': list(episodes),
            'trial_count': 1,
        },
        tasks=[EvalTask(name=MOLMO_TASK, trials=[{MOLMO_EPISODE_INDEX_KEY: i} for i in episodes])],
        time_limit_s=7200,
        scorer=MOLMO_SCORER,
        positronic_revision=MOLMO_POSITRONIC_REVISION,
    )


_PUBLIC_EVALS: dict[EvalRef, EvalDefinition] = {
    EvalRef('molmo.franka_pick_mini'): _franka_pick_mini(list(range(20))),
    EvalRef('molmo.franka_pick_mini_smoke'): _franka_pick_mini([0, 1, 2, 3, 4]),
}

PUBLIC_EVALS: tuple[EvalRef, ...] = tuple(_PUBLIC_EVALS)


def public_eval(name: EvalRef) -> EvalDefinition:
    """The definition of the public eval `name`, as a copy, or LookupError naming the public evals."""
    definition = _PUBLIC_EVALS.get(name)
    if definition is None:
        raise LookupError(f'{name!r} is not a public eval; the public evals are {", ".join(PUBLIC_EVALS)}')
    return definition.model_copy(deep=True)
