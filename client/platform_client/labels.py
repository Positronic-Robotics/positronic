"""One labeler's account of one recorded episode, and the labelers of the public evals.

A `Label` holds one episode's timed events, its end and its totals, and its views are properties of the events.
`label` gives an episode directory the label of the first labeler, in precedence order, that labels it.
"""

from __future__ import annotations

import json
import logging
import math
from bisect import bisect_right
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import IntEnum, unique
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, ClassVar, Literal, NamedTuple, NewType, Self

from eval_vocabulary.episode import (
    DURATION,
    ENDED_BY,
    ENDED_BY_OPERATOR,
    LEGACY_TIME_COLUMN,
    META_FILE,
    SIGNAL_FILE_SUFFIX,
    STATIC_FILE,
    SUCCESS,
    TASK,
    TERMINATED,
    UID,
)
from platform_client.slug import Slugged
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)
from pydantic_core import core_schema

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # The `pyarrow` extra. Only a recording that has no `eval.duration` needs it.
    pa = pq = None

log = logging.getLogger(__name__)


class LabelerRef(str):
    """The name of a labeler, as a validated `str`."""

    __slots__ = ()

    def __new__(cls, value: str) -> LabelerRef:
        if not value or any(c.isspace() for c in value):
            raise ValueError(f'not a labeler name: {value!r}')
        return super().__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(cls, source: type, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(cls, core_schema.str_schema())


# rules-allow: earn-its-place — the API word; `eval_vocabulary.outcome` keeps the recorded word; a labeler maps them
@unique
class Outcome(IntEnum):
    """How an episode ended, as the events of its label say."""

    INVALID = 0
    success = 1
    fail = 2
    out_of_time = 3
    safety = 4


# rules-allow: earn-its-place — the API word; `eval_vocabulary.progress` keeps the recorded word; a labeler maps them
@unique
class Stage(IntEnum):
    """The ladder a task passes through. A higher number is a higher stage."""

    INVALID = 0
    floating = 10
    reaching = 20
    contact = 30
    control = 40
    at_target = 50


_LABEL_CONFIG = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)

# An item as the task's object list names it.
ItemName = NewType('ItemName', str)


class StageItems(BaseModel):
    """The items that one stage mark applies to, for example the three spoons that the arm controls."""

    model_config = _LABEL_CONFIG

    count: PositiveInt
    # The count of each item name, where the labeler tells the items apart.
    names: dict[ItemName, PositiveInt] = {}

    @model_validator(mode='after')
    def _names_add_up_to_the_count(self) -> Self:
        if self.names and sum(self.names.values()) != self.count:
            raise ValueError(f'the counts in {self.names} add up to {sum(self.names.values())}, not to {self.count}')
        return self


class StageReached(BaseModel):
    """The arm reached `stage`. A stage repeats when the operator marks the ladder again for the next item."""

    model_config = _LABEL_CONFIG

    kind: Literal['stage_reached'] = 'stage_reached'
    seconds: NonNegativeFloat
    stage: Slugged[Stage]
    # None: the labeler does not state the items, as where the task moves one item at a time.
    items: StageItems | None = None


class ItemsDelivered(BaseModel):
    """The count of items delivered became `delivered`."""

    model_config = _LABEL_CONFIG

    kind: Literal['items_delivered'] = 'items_delivered'
    seconds: NonNegativeFloat
    delivered: NonNegativeInt


class SubtaskAchieved(BaseModel):
    """The count of subtasks achieved became `achieved`."""

    model_config = _LABEL_CONFIG

    kind: Literal['subtask_achieved'] = 'subtask_achieved'
    seconds: NonNegativeFloat
    achieved: NonNegativeInt
    # The env's weighted score of the subtasks achieved, where the env grades one.
    score: Annotated[float, Field(ge=0, le=1)] | None = None


class Success(BaseModel):
    """The labeler states that the policy did the task."""

    model_config = _LABEL_CONFIG
    outcome: ClassVar[Outcome] = Outcome.success

    kind: Literal['success'] = 'success'
    seconds: NonNegativeFloat


class Fail(BaseModel):
    """The labeler states that the policy failed the task."""

    model_config = _LABEL_CONFIG
    outcome: ClassVar[Outcome] = Outcome.fail

    kind: Literal['fail'] = 'fail'
    seconds: NonNegativeFloat
    reason: str | None = None  # the labeler's words


class Safety(BaseModel):
    """The labeler states that the policy was stopped for safety."""

    model_config = _LABEL_CONFIG
    outcome: ClassVar[Outcome] = Outcome.safety

    kind: Literal['safety'] = 'safety'
    seconds: NonNegativeFloat
    reason: str | None = None  # the labeler's words


EndEvent = Success | Fail | Safety
Event = Annotated[
    StageReached | ItemsDelivered | SubtaskAchieved | Success | Fail | Safety, Field(discriminator='kind')
]


class Attempt(NamedTuple):
    """The part of an episode in which `done` items, or subtasks, are done."""

    done: int
    start_seconds: float
    end_seconds: float  # the next change of the count, or the episode end
    stage_seconds: dict[Stage, float]  # the first mark of each stage in this part, from the trial start


def _first_marks(marks: Iterable[StageReached]) -> dict[Stage, float]:
    first: dict[Stage, float] = {}
    for mark in marks:
        first.setdefault(mark.stage, mark.seconds)
    return first


# The order of two end events at the same time: the first in this tuple decides the outcome.
_SAME_TIME_ORDER = (Outcome.safety, Outcome.fail, Outcome.success)


class Label(BaseModel):
    """One labeler's account of one episode.

    Each event carries its seconds from the trial start. The events are in ascending order of time, and none
    falls after `seconds_to_end`. A fact that the source does not time sits at the episode end.
    """

    model_config = _LABEL_CONFIG

    uid: str = Field(min_length=1)  # the id that the episode's meta records
    task: str  # the task a metric groups by
    # The labeler that gave the label. None: the publisher withholds the name.
    # rules-allow: earn-its-place — a stored label keeps its labeler, and `label()` returns its own key for precedence
    annotator: LabelerRef | None = None
    seconds_to_end: NonNegativeFloat
    total_items: PositiveInt | None = None
    total_subtasks: PositiveInt | None = None
    # Items lost for the rest of the episode: dropped outside the workspace, or under way at a safety stop.
    # No source times them.
    lost_items: NonNegativeInt = 0
    events: tuple[Event, ...] = ()
    notes: str | None = None  # the operator's notes

    @model_validator(mode='after')
    def _events_are_ordered_within_the_episode(self) -> Self:
        times = [event.seconds for event in self.events]
        if any(later < earlier for earlier, later in zip(times, times[1:], strict=False)):
            raise ValueError(f'the events are not in ascending order of time: {times}')
        if times and times[-1] > self.seconds_to_end:
            raise ValueError(f'an event at {times[-1]} s falls after the episode end at {self.seconds_to_end} s')
        return self

    def until(self, seconds: float) -> Label:
        """The label as it stood at `seconds`: the events at or before it, and the earlier of the two ends."""
        if seconds < 0:
            raise ValueError(f'a label cannot stand at {seconds} s, before the trial start')
        kept = tuple(event for event in self.events if event.seconds <= seconds)
        return self.model_copy(update={'events': kept, 'seconds_to_end': min(self.seconds_to_end, seconds)})

    def _deciding_end(self) -> EndEvent | None:
        ends = [event for event in self.events if isinstance(event, EndEvent)]
        return min(ends, key=lambda end: (end.seconds, _SAME_TIME_ORDER.index(end.outcome)), default=None)

    @property
    def outcome(self) -> Outcome:
        """The outcome that the earliest end event states, or `out_of_time` when there is no end event."""
        end = self._deciding_end()
        return end.outcome if end is not None else Outcome.out_of_time

    @property
    def seconds_to_outcome(self) -> float | None:
        """The time of the end event that decides `outcome`. None for `out_of_time`."""
        end = self._deciding_end()
        return end.seconds if end is not None else None

    @property
    def seconds_to_success(self) -> float | None:
        return self.seconds_to_outcome if self.outcome is Outcome.success else None

    @property
    def stage_seconds(self) -> dict[Stage, float]:
        """The first mark of each stage."""
        return _first_marks(event for event in self.events if isinstance(event, StageReached))

    @property
    def furthest_stage(self) -> Stage | None:
        return max(self.stage_seconds, default=None)

    @property
    def placements(self) -> tuple[float, ...]:
        """The time of each item's delivery, one time per item. A fall in the count takes back the latest items."""
        placed: list[float] = []
        for event in self.events:
            if isinstance(event, ItemsDelivered):
                placed = placed[: event.delivered] + [event.seconds] * (event.delivered - len(placed))
        return tuple(placed)

    @property
    def delivered(self) -> int:
        """The last count of items delivered."""
        counts = [event.delivered for event in self.events if isinstance(event, ItemsDelivered)]
        return counts[-1] if counts else 0

    @property
    def subtask_score(self) -> float | None:
        """The score of the last subtask event."""
        scores = [event.score for event in self.events if isinstance(event, SubtaskAchieved)]
        return scores[-1] if scores else None

    @property
    def attempts(self) -> tuple[Attempt, ...]:
        """The episode, split at each change of the items count, else of the subtasks count."""
        counts = [(event.seconds, event.delivered) for event in self.events if isinstance(event, ItemsDelivered)]
        if not counts:
            counts = [(event.seconds, event.achieved) for event in self.events if isinstance(event, SubtaskAchieved)]
        starts = [(0.0, 0)]
        for seconds, done in counts:
            if done != starts[-1][1]:
                starts.append((seconds, done))
        ends = [seconds for seconds, _ in starts[1:]] + [self.seconds_to_end]
        marks: list[list[StageReached]] = [[] for _ in starts]
        start_times = [seconds for seconds, _ in starts]
        for event in self.events:
            if isinstance(event, StageReached):
                marks[bisect_right(start_times, event.seconds) - 1].append(event)
        return tuple(
            Attempt(done, start, end, _first_marks(part))
            for (start, done), end, part in zip(starts, ends, marks, strict=True)
        )


# Gives one recorded episode its label, or None when this labeler has nothing to say about it.
Labeler = Callable[[Path], Label | None]


def _read_object(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding='utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        loaded = None
    if not isinstance(loaded, dict):
        # One damaged upload leaves the other episodes labelled, so this is logged, not raised.
        log.error('%s does not hold a JSON object; this episode has no label', path)
        return None
    return loaded


def read_static(episode: Path) -> dict[str, object] | None:
    """The statics that `episode` recorded, or None when it recorded none or they do not parse."""
    return _read_object(episode / STATIC_FILE)


def recorded_uid(episode: Path) -> str | None:
    """The id that the meta of `episode` records, or None when the meta does not record one."""
    meta = _read_object(episode / META_FILE)
    if meta is None:
        return None
    uid = meta.get(UID)
    if not isinstance(uid, str) or not uid:
        log.error('%s holds %s=%r, which is not an episode id; this episode has no label', META_FILE, UID, uid)
        return None
    return uid


def recorded_task_and_outcome(static: Mapping[str, object], task_key: str) -> tuple[str, Outcome] | None:
    """The task under `task_key` and how the trial ended, or None when either is missing or of the wrong type.

    A trial that ran out of time ended `out_of_time`, and the success that it recorded does not count. A success of
    the wrong type is still logged.
    """
    terminated = static.get(TERMINATED)
    if TERMINATED in static and not isinstance(terminated, bool):
        log.error('%s=%r is not a bool; reading it as absent', TERMINATED, terminated)
    success = static.get(SUCCESS)
    if terminated is False:
        outcome = Outcome.out_of_time
    elif isinstance(success, bool):
        outcome = Outcome.success if success else Outcome.fail
    else:
        outcome = None
    task = static.get(task_key)
    recorded = (task, outcome) if isinstance(task, str) and outcome is not None else None
    misrecorded = [
        f'{key}={static[key]!r} is not a {kind.__name__}'
        for key, kind in ((SUCCESS, bool), (task_key, str))
        if key in static and not isinstance(static[key], kind)
    ]
    if misrecorded:
        # One damaged record leaves the other episodes labelled, so this is logged, not raised.
        verdict = 'this episode has no label' if recorded is None else 'the trial ran out of time'
        log.error('%s; %s', '; '.join(misrecorded), verdict)
    return recorded


def recorded_seconds_to_end(episode: Path, static: Mapping[str, object]) -> float | None:
    """The seconds from the trial's start to its end, or None when the recording does not say.

    A recording that has no `eval.duration` gives the time from the first to the last sample of its signals, and
    reading the signals needs the `pyarrow` extra.
    """
    duration = static.get(DURATION)
    if isinstance(duration, int | float) and not isinstance(duration, bool) and 0 <= duration < math.inf:
        return float(duration)
    if DURATION in static:
        log.error('%s=%r is not a count of seconds; reading it as absent', DURATION, duration)
    return _signal_seconds(episode)


def _signal_seconds(episode: Path) -> float | None:
    if pa is None or pq is None:
        raise ModuleNotFoundError(f'{episode} records no {DURATION}, and reading its signals needs the `pyarrow` extra')
    firsts, lasts = [], []
    for signal in sorted(episode.glob(f'*{SIGNAL_FILE_SUFFIX}')):
        try:
            if LEGACY_TIME_COLUMN not in pq.read_schema(signal).names:
                continue
            times = pq.read_table(signal, columns=[LEGACY_TIME_COLUMN]).column(LEGACY_TIME_COLUMN)
        except pa.ArrowInvalid:
            # One damaged upload leaves the other episodes labelled, so this is logged, not raised.
            log.error('%s does not parse; this episode has no label', signal)
            return None
        if len(times):
            firsts.append(times[0].as_py())
            lasts.append(times[-1].as_py())
    if not lasts:
        log.error(
            '%s records no %s and no sample in a %r column; this episode has no label',
            episode,
            DURATION,
            LEGACY_TIME_COLUMN,
        )
        return None
    return (max(lasts) - min(firsts)) / 1e9


MOLMO_LABELER = LabelerRef('molmo')
# The end event of each outcome that one states. `out_of_time` has none.
_END_EVENTS: Mapping[Outcome, type[EndEvent]] = MappingProxyType({end.outcome: end for end in (Success, Fail, Safety)})


def molmo_label(episode: Path) -> Label | None:
    """MolmoSpaces' labeler: a trial succeeds or fails at its end, or runs out of time.

    A trial that an operator ended has no label.
    """
    static = read_static(episode)
    if static is None or static.get(ENDED_BY) == ENDED_BY_OPERATOR:
        return None
    recorded = recorded_task_and_outcome(static, TASK)
    uid = recorded_uid(episode) if recorded is not None else None
    seconds_to_end = recorded_seconds_to_end(episode, static) if uid is not None else None
    if recorded is None or uid is None or seconds_to_end is None:
        return None
    task, outcome = recorded
    end = _END_EVENTS.get(outcome)
    events = (end(seconds=seconds_to_end),) if end is not None else ()
    return Label(uid=uid, task=task, annotator=MOLMO_LABELER, seconds_to_end=seconds_to_end, events=events)


PUBLIC_LABELERS: Mapping[LabelerRef, Labeler] = MappingProxyType({MOLMO_LABELER: molmo_label})


def label(
    episode: Path, labelers: Sequence[LabelerRef], table: Mapping[LabelerRef, Labeler] = PUBLIC_LABELERS
) -> tuple[LabelerRef, Label] | None:
    """The first labeler in `labelers` that labels `episode`, with its label.

    LookupError when `table` does not hold one of `labelers`.
    """
    missing = [ref for ref in labelers if ref not in table]
    if missing:
        raise LookupError(f'no labeler {", ".join(map(repr, missing))}; the labelers are {", ".join(sorted(table))}')
    for ref in labelers:
        given = table[ref](episode)
        if given is not None:
            return ref, given
    return None
