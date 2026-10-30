"""The one name a caller picks a run by, and the definitions of the public evals.

A name is a task suite AND the embodiment that runs it, so there is no second axis to get wrong.
The platform owns the set, so a name that this package does not define still reaches the platform.

The platform holds the definitions of the evals it does not publish.
"""

from __future__ import annotations

from eval_vocabulary.episode import SEED
from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler, JsonValue, PositiveInt
from pydantic_core import core_schema

from positronic_evals.labels import ABC_LABELER, MOLMO_LABELER
from positronic_evals.scoring import MeanEpisodeScore, Scoring


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
    scoring: Scoring
    # The positronic commit the platform runs the config at.
    positronic_revision: str = Field(pattern=r'^[0-9a-f]{40}$')


# The MolmoSpaces evals: the config they name, its task and trial key as positronic spells them
# (`positronic/simulator/molmo_spaces/keys.py`), their scoring, and the positronic commit the platform's sim
# image holds. A run ranks on its success rate.
MOLMO_CONFIG = '.sim.molmo.benchmarks'
MOLMO_TASK = 'franka_pick_droid_mini'
MOLMO_EPISODE_INDEX_KEY = 'molmo.episode_index'
MOLMO_SCORING = Scoring(labelers=(MOLMO_LABELER,), primary=MeanEpisodeScore())
MOLMO_POSITRONIC_REVISION = 'a9e13e8a1fbe624ff62b49e25628b39859619986'


def _franka_pick_mini(episodes: list[int]) -> EvalDefinition:
    """`episodes` of the one benchmark the MolmoSpaces evals run, once each."""
    return EvalDefinition(
        config=MOLMO_CONFIG,
        # Each key must name a parameter of positronic's `benchmarks` config.
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
        scoring=MOLMO_SCORING,
        positronic_revision=MOLMO_POSITRONIC_REVISION,
    )


# The ABC evals on the bimanual i2rt YAM. A run ranks on the mean of its task success rates, and a timeout scores 0.
ABC_CONFIG = '.sim.amazon_abc.all_tasks'
# The 29 tasks that ABC ships a success evaluator for at the ABC commit that the config pins. ABC scores none of its
# other four tasks.
ABC_SCORED_TASKS = (
    'throw_plastic_bottles_in_bin',
    'put_plastic_bottles_in_bin',
    'conveyor_pick',
    'count_into_opaque_box',
    'count_one_into_opaque_box',
    'count_two_into_opaque_box',
    'count_three_into_opaque_box',
    'count_four_into_opaque_box',
    'put_relative',
    'grab_specific_object_from_clutter',
    'ball_tray_balancing',
    'inhand_transfer_item_to_other_side',
    'multi_drawer_search',
    'pouring',
    'hang_mug_on_mug_rack',
    'turn_mug_right_side_up',
    'sweep_away_paper_scraps_from_table',
    'sort_nuts_and_bolts',
    'sort_lego_blocks',
    'load_plates_into_dish_rack',
    'set_up_chess_pieces_on_board',
    'spell_cat',
    'spell_dog',
    'spell_fish',
    'spell_bair',
    'spell_xdof',
    'spell_abc',
    'spell_yam',
    'spell_agi',
)
# The five tasks that ABC's multi-task sim policy trained on.
ABC_MULTITASK_TASKS = (
    'put_plastic_bottles_in_bin',
    'sweep_away_paper_scraps_from_table',
    'turn_mug_right_side_up',
    'load_plates_into_dish_rack',
    'hang_mug_on_mug_rack',
)
# ABC's own sim eval draws world `i` of a task from seed `20260511 + i`, so trial `i` here runs the same scene.
ABC_FIRST_SEED = 20260511
ABC_TRIALS_PER_TASK = 20
# The render size and the camera view change what a policy sees, so a definition pins them rather than take a
# config default.
ABC_CAMERA_HEIGHT, ABC_CAMERA_WIDTH = 260, 416
# The value of `positronic.simulator.amazon_abc.mapping.CameraView.ZED_X`: a fit to the ZED-X stations of ABC-130k.
ABC_CAMERA_VIEW = 'zed_x'
ABC_SCORING = Scoring(labelers=(ABC_LABELER,), primary=MeanEpisodeScore(per_task=True))
# The `main` commit that added the camera view to the ABC config.
ABC_POSITRONIC_REVISION = 'b2499cbe5565dc3cbd9cd8fcbdf29f588a6aaa88'


def _abc(tasks: tuple[str, ...], time_limit_s: int) -> EvalDefinition:
    """`tasks`, each on the first `ABC_TRIALS_PER_TASK` scenes of ABC's own sim eval."""
    seeds = range(ABC_FIRST_SEED, ABC_FIRST_SEED + ABC_TRIALS_PER_TASK)
    return EvalDefinition(
        config=ABC_CONFIG,
        # Each key must name a parameter of positronic's `all_tasks` config.
        args={
            'task': list(tasks),
            'seed': ABC_FIRST_SEED,
            'trial_count': ABC_TRIALS_PER_TASK,
            'camera_height': ABC_CAMERA_HEIGHT,
            'camera_width': ABC_CAMERA_WIDTH,
            'camera_view': ABC_CAMERA_VIEW,
        },
        tasks=[EvalTask(name=task, trials=[{SEED: seed} for seed in seeds]) for task in tasks],
        time_limit_s=time_limit_s,
        scoring=ABC_SCORING,
        positronic_revision=ABC_POSITRONIC_REVISION,
    )


_PUBLIC_EVALS: dict[EvalRef, EvalDefinition] = {
    # Episode 6 hides the start grasp site from the right shoulder camera; the sim server refuses it.
    EvalRef('molmo.franka_pick_mini'): _franka_pick_mini([*range(6), *range(7, 21)]),
    EvalRef('molmo.franka_pick_mini_smoke'): _franka_pick_mini([0, 1, 2, 3, 4]),
    EvalRef('abc.yam_bimanual'): _abc(ABC_SCORED_TASKS, time_limit_s=6 * 3600),
    EvalRef('abc.yam_bimanual_multitask'): _abc(ABC_MULTITASK_TASKS, time_limit_s=2 * 3600),
}

PUBLIC_EVALS: tuple[EvalRef, ...] = tuple(_PUBLIC_EVALS)


def public_eval(name: EvalRef) -> EvalDefinition:
    """The definition of the public eval `name`, as a copy, or LookupError naming the public evals."""
    definition = _PUBLIC_EVALS.get(name)
    if definition is None:
        raise LookupError(f'{name!r} is not a public eval; the public evals are {", ".join(PUBLIC_EVALS)}')
    return definition.model_copy(deep=True)
