"""One labeler's account of one recorded episode, and the labelers that this package publishes.

A `Label` holds one episode's timed events, its end and its totals, and its views are properties of the events.
`label` gives the first labeler, in precedence order, that labels an episode directory, with its label.
"""

from __future__ import annotations

import json
import logging
from bisect import bisect_right
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import IntEnum, unique
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, ClassVar, Literal, NamedTuple, NewType, Self

import pyarrow as pa
import pyarrow.parquet as pq
from eval_vocabulary import episode as vocabulary
from eval_vocabulary.episode import (
    BENCHMARK_TASK,
    EDITS_FILE,
    ENDED_BY,
    ENDED_BY_OPERATOR,
    LEGACY_TIME_COLUMN,
    META_FILE,
    SIGNAL_FILE_SUFFIX,
    STATIC_FILE,
    SUBTASK_COMPLETED_INDEX,
    SUBTASK_SCORE_INDEX,
    SUBTASK_SIGNAL,
    SUBTASK_TOTAL_INDEX,
    SUCCESS,
    TASK,
    TERMINATED,
    TIMESTAMP_COLUMN_PREFIX,
    UID,
    VALUE_COLUMN,
    EditOp,
)
from eval_vocabulary.outcome import Outcome as CountedOutcome
from eval_vocabulary.outcome import counted_outcome
from eval_vocabulary.timeline import EMITTED_WORLD
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

from positronic_evals.slug import Slugged

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

    @model_validator(mode='after')
    def _counts_stay_within_their_totals(self) -> Self:
        item_counts = [(self.lost_items, 'items lost')]
        subtask_counts: list[tuple[int, str]] = []
        for event in self.events:
            if isinstance(event, ItemsDelivered):
                item_counts.append((event.delivered, 'items delivered'))
            elif isinstance(event, StageReached) and event.items is not None:
                item_counts.append((event.items.count, 'items at a stage mark'))
            elif isinstance(event, SubtaskAchieved):
                subtask_counts.append((event.achieved, 'subtasks achieved'))
        for total, counts in ((self.total_items, item_counts), (self.total_subtasks, subtask_counts)):
            for count, what in counts:
                if total is not None and count > total:
                    raise ValueError(f'{count} {what} is more than the total of {total}')
        if self.total_items is not None and self.delivered + self.lost_items > self.total_items:
            raise ValueError(
                f'{self.delivered} items delivered and {self.lost_items} items lost are more than the total of '
                f'{self.total_items}'
            )
        return self

    def until(self, seconds: float) -> Label:
        """The label as it stood at `seconds`: the events and the losses at or before it, and the earlier end."""
        if not seconds >= 0:
            raise ValueError(f'a label cannot stand at {seconds} s, which is not a time from the trial start')
        kept = tuple(event for event in self.events if event.seconds <= seconds)
        lost = self.lost_items if seconds >= self.seconds_to_end else 0
        return self.model_copy(
            update={'events': kept, 'seconds_to_end': min(self.seconds_to_end, seconds), 'lost_items': lost}
        )

    # The order of two end events at the same time: the first in this tuple decides the outcome.
    _SAME_TIME_ORDER: ClassVar[tuple[Outcome, ...]] = (Outcome.safety, Outcome.fail, Outcome.success)

    def _deciding_end(self) -> EndEvent | None:
        ends = [event for event in self.events if isinstance(event, EndEvent)]
        return min(ends, key=lambda end: (end.seconds, self._SAME_TIME_ORDER.index(end.outcome)), default=None)

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

    @staticmethod
    def _first_marks(marks: Iterable[StageReached]) -> dict[Stage, float]:
        first: dict[Stage, float] = {}
        for mark in marks:
            first.setdefault(mark.stage, mark.seconds)
        return first

    @property
    def stage_seconds(self) -> dict[Stage, float]:
        """The first mark of each stage."""
        return self._first_marks(event for event in self.events if isinstance(event, StageReached))

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
            Attempt(done, start, end, self._first_marks(part))
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


def _edits(log_path: Path, uid: str) -> list[tuple[EditOp, dict[str, object]]] | None:
    """The edits that the log at `log_path` makes to the episode `uid`, in order, or None when a record does not parse.

    A `DROP` or an `UNDROP` carries no statics.
    """
    try:
        lines = log_path.read_text(encoding='utf-8').splitlines()
    except UnicodeDecodeError:
        log.error('%s is not UTF-8 text; this episode has no label', log_path)
        return None
    edits: list[tuple[EditOp, dict[str, object]]] = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            record = None
        match record:
            case {
                vocabulary.EDIT_OP: EditOp.SET_STATIC,
                vocabulary.EDIT_FORMAT: vocabulary.EDIT_FORMAT_VERSION,
                vocabulary.EDIT_EPISODE: str(episode),
                vocabulary.EDIT_DATA: dict(data),
            }:
                edit = (EditOp.SET_STATIC, data)
            case {
                vocabulary.EDIT_OP: (EditOp.DROP | EditOp.UNDROP) as op,
                vocabulary.EDIT_FORMAT: vocabulary.EDIT_FORMAT_VERSION,
                vocabulary.EDIT_EPISODE: str(episode),
            }:
                edit = (EditOp(op), {})
            case _:
                # One damaged upload leaves the other episodes labelled, so this is logged, not raised.
                log.error('%s holds a record that is not an edit, %r; this episode has no label', log_path, line)
                return None
        if episode == uid:
            edits.append(edit)
    return edits


def read_static(episode: Path) -> dict[str, object] | None:
    """The statics that `episode` recorded, with the edits of its own edit log over them.

    None when it recorded none, when they or the log do not parse, or when the log drops the episode. The edit logs
    of the dataset directories above the episode do not apply.
    """
    static = _read_object(episode / STATIC_FILE)
    log_path = episode / EDITS_FILE
    if static is None or not log_path.is_file():
        return static
    uid = recorded_uid(episode)
    edits = _edits(log_path, uid) if uid is not None else None
    if edits is None:
        return None
    dropped = False
    for op, data in edits:
        if op is EditOp.SET_STATIC:
            static |= data
        else:
            dropped = op is EditOp.DROP
    return None if dropped else static


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


# The outcome of a label for each outcome that `counted_outcome` gives.
_LABEL_OUTCOMES: Mapping[CountedOutcome, Outcome] = MappingProxyType({
    CountedOutcome.SUCCESS: Outcome.success,
    CountedOutcome.FAIL: Outcome.fail,
    CountedOutcome.OUT_OF_TIME: Outcome.out_of_time,
    CountedOutcome.SAFETY: Outcome.safety,
    CountedOutcome.SYSTEM: Outcome.fail,
})


def recorded_task_and_outcome(static: Mapping[str, object], task_key: str) -> tuple[str, Outcome] | None:
    """The task under `task_key` and the outcome that `counted_outcome` gives, or None when either is missing.

    `UNSCORED` and `DISCARDED` give no outcome. A success or a timeout marker of the wrong type is logged, and
    `counted_outcome` reads it as absent.
    """
    terminated = static.get(TERMINATED)
    if TERMINATED in static and not isinstance(terminated, bool):
        log.error('%s=%r is not a bool; reading it as absent', TERMINATED, terminated)
    outcome = _LABEL_OUTCOMES.get(counted_outcome(static))
    task = static.get(task_key)
    recorded = (task, outcome) if isinstance(task, str) and outcome is not None else None
    misrecorded = [
        f'{key}={static[key]!r} is not a {kind.__name__}'
        for key, kind in ((SUCCESS, bool), (task_key, str))
        if key in static and not isinstance(static[key], kind)
    ]
    if misrecorded:
        # One damaged record leaves the other episodes labelled, so this is logged, not raised.
        verdict = 'this episode has no label' if recorded is None else f'the trial counts as {recorded[1].name}'
        log.error('%s; %s', '; '.join(misrecorded), verdict)
    return recorded


_WORLD_TIME_COLUMN = f'{TIMESTAMP_COLUMN_PREFIX}{EMITTED_WORLD}'


def _world_time_column(signal: Path) -> str | None:
    """The column of `signal` that holds each sample's world time, or None. ArrowInvalid when it does not parse."""
    names = pq.read_schema(signal).names
    return next((name for name in (_WORLD_TIME_COLUMN, LEGACY_TIME_COLUMN) if name in names), None)


def recorded_span(episode: Path) -> tuple[int, int] | None:
    """The world times of the first and the last sample of the episode's signals, in nanoseconds.

    The last can fall up to one control step after the end of the trial.
    None when no signal file holds a world time, or one does not parse.
    """
    firsts, lasts = [], []
    for signal in sorted(episode.glob(f'*{SIGNAL_FILE_SUFFIX}')):
        try:
            column = _world_time_column(signal)
            if column is None:
                continue
            times = pq.read_table(signal, columns=[column]).column(column)
        except pa.ArrowInvalid:
            # One damaged upload leaves the other episodes labelled, so this is logged, not raised.
            log.error('%s does not parse; this episode has no label', signal)
            return None
        if len(times):
            firsts.append(times[0].as_py())
            lasts.append(times[-1].as_py())
    if not lasts:
        log.error(
            '%s holds no sample in a %r or a %r column; this episode has no label',
            episode,
            _WORLD_TIME_COLUMN,
            LEGACY_TIME_COLUMN,
        )
        return None
    return min(firsts), max(lasts)


# The end event of each outcome that one states. `out_of_time` has none.
_END_EVENTS: Mapping[Outcome, type[EndEvent]] = MappingProxyType({end.outcome: end for end in (Success, Fail, Safety)})


class _SimTrial(NamedTuple):
    """What a sim episode records of its trial."""

    uid: str
    task: str
    outcome: Outcome
    start_ns: int  # the world time of the first sample of the episode's signals
    seconds_to_end: float

    @property
    def end_events(self) -> tuple[EndEvent, ...]:
        """The end event that the outcome states, at the end of the trial."""
        end = _END_EVENTS.get(self.outcome)
        return (end(seconds=self.seconds_to_end),) if end is not None else ()


def _recorded_sim_trial(episode: Path, task_key: str) -> _SimTrial | None:
    """The trial that `episode` records, with its task under `task_key`.

    None when an operator ended the trial, when the episode gives no outcome that a label states, when its edit log
    drops it, or when it records no id or no world time.
    """
    static = read_static(episode)
    if static is None or static.get(ENDED_BY) == ENDED_BY_OPERATOR:
        return None
    recorded = recorded_task_and_outcome(static, task_key)
    uid = recorded_uid(episode) if recorded is not None else None
    span = recorded_span(episode) if uid is not None else None
    if recorded is None or uid is None or span is None:
        return None
    start_ns, end_ns = span
    return _SimTrial(uid, *recorded, start_ns, (end_ns - start_ns) / 1e9)


MOLMO_LABELER = LabelerRef('molmo')


def molmo_label(episode: Path) -> Label | None:
    """MolmoSpaces' labeler: a trial ends with the outcome that `counted_outcome` gives, at the end of its signals.

    A trial that an operator ended has no label, and neither has an episode that `counted_outcome` leaves unscored.
    """
    trial = _recorded_sim_trial(episode, TASK)
    if trial is None:
        return None
    return Label(
        uid=trial.uid,
        task=trial.task,
        annotator=MOLMO_LABELER,
        seconds_to_end=trial.seconds_to_end,
        events=trial.end_events,
    )


ROBOLAB_LABELER = LabelerRef('robolab')


class _SubtaskSample(NamedTuple):
    """One sample of the `subtask` signal."""

    time_ns: int  # on the world clock
    completed: int
    total: int
    score: float


def _subtask_samples(episode: Path) -> list[_SubtaskSample]:
    """Each sample of the `subtask` signal, which `recorded_span` has parsed."""
    signal = episode / f'{SUBTASK_SIGNAL}{SIGNAL_FILE_SUFFIX}'
    column = _world_time_column(signal) if signal.is_file() else None
    if column is None:
        return []
    table = pq.read_table(signal, columns=[column, VALUE_COLUMN])
    return [
        _SubtaskSample(
            time_ns,
            int(vector[SUBTASK_COMPLETED_INDEX]),
            int(vector[SUBTASK_TOTAL_INDEX]),
            float(vector[SUBTASK_SCORE_INDEX]),
        )
        for time_ns, vector in zip(
            table.column(column).to_pylist(), table.column(VALUE_COLUMN).to_pylist(), strict=True
        )
    ]


def robolab_label(episode: Path) -> Label | None:
    """RoboLab's labeler: a trial ends as in `molmo_label`, and the `subtask` signal gives its progress.

    A `SubtaskAchieved` event sits at each sample where the count of subtasks completed changes, with the score
    of that sample. A change of the score alone is not an event, so the label's `subtask_score` is the score at
    the last change of the count. A trial whose signal holds no sample has no label.
    """
    trial = _recorded_sim_trial(episode, BENCHMARK_TASK)
    samples = _subtask_samples(episode) if trial is not None else []
    if trial is None or not samples:
        return None
    events: list[SubtaskAchieved] = []
    achieved = 0
    for sample in samples:
        if sample.completed != achieved:
            achieved = sample.completed
            seconds = (sample.time_ns - trial.start_ns) / 1e9
            events.append(SubtaskAchieved(seconds=seconds, achieved=achieved, score=sample.score))
    return Label(
        uid=trial.uid,
        task=trial.task,
        annotator=ROBOLAB_LABELER,
        seconds_to_end=trial.seconds_to_end,
        # The env records zeros on the frame of a reset, and a total of 0 states no total.
        total_subtasks=samples[-1].total or None,
        events=(*events, *trial.end_events),
    )


PUBLIC_LABELERS: Mapping[LabelerRef, Labeler] = MappingProxyType({
    MOLMO_LABELER: molmo_label,
    ROBOLAB_LABELER: robolab_label,
})


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
