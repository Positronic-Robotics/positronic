"""One labeler's account of one recorded episode, and the labelers that this package publishes.

A `Label` holds one episode's timed events, its end and its totals, and its views are properties of the events.
`label` gives the first labeler, in precedence order, that labels an episode directory, with its label.
"""

from __future__ import annotations

import json
import logging
import math
from bisect import bisect_right
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from enum import IntEnum, unique
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Literal, NamedTuple, NewType, Self

import pyarrow as pa
import pyarrow.parquet as pq
from eval_vocabulary.episode import (
    ABC_REWARD_SIGNAL,
    BENCHMARK_TASK,
    EE_POSE_SUFFIX,
    ENDED_BY,
    ENDED_BY_OPERATOR,
    GT_STATE_GRABBED,
    GT_STATE_GRIPPER_CONTACT,
    GT_STATE_OBJECTS,
    GT_STATE_POSE,
    GT_STATE_PROGRESS,
    GT_STATE_SIGNAL,
    LEGACY_FRAMES_TIME_COLUMN,
    LEGACY_TIME_COLUMN,
    META_FILE,
    ROBOT_STATE,
    SIGNAL_FILE_SUFFIX,
    STATIC_FILE,
    SUBTASK_COMPLETED_INDEX,
    SUBTASK_SCORE_INDEX,
    SUBTASK_SIGNAL,
    SUBTASK_TOTAL_INDEX,
    SUCCESS,
    TASK,
    TERMINATED,
    TIMELINE_METADATA_KEY,
    TIMESTAMP_COLUMN_PREFIX,
    UID,
    VALUE_COLUMN,
)
from eval_vocabulary.progress import MARK_CHANNELS
from eval_vocabulary.timeline import EMITTED_WORLD, RECORDED_TIME, recording_timeline
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


_WORLD_TIME_COLUMN = f'{TIMESTAMP_COLUMN_PREFIX}{EMITTED_WORLD}'


def _world_time_column(signal: Path) -> str | None:
    """The column of `signal` that holds each sample's world time, or None. ArrowInvalid when it does not parse."""
    names = pq.read_schema(signal).names
    return next((name for name in (_WORLD_TIME_COLUMN, LEGACY_TIME_COLUMN) if name in names), None)


def _signal_times(
    episode: Path, time_column: Callable[[Path], str | None]
) -> list[tuple[Path, pa.ChunkedArray]] | None:
    """Each signal of `episode` with a sample in the column that `time_column` picks, with the times in that column.

    None when a signal file does not parse.
    """
    read = []
    for signal in sorted(episode.glob(f'*{SIGNAL_FILE_SUFFIX}')):
        try:
            column = time_column(signal)
            if column is None:
                continue
            times = pq.read_table(signal, columns=[column]).column(column)
        except pa.ArrowInvalid:
            # One damaged upload leaves the other episodes read, so this is logged, not raised.
            log.error('%s does not parse; this episode is not read', signal)
            return None
        if len(times):
            read.append((signal, times))
    return read


def recorded_span(episode: Path) -> tuple[int, int] | None:
    """The world times of the first and the last sample of the episode's signals, in nanoseconds.

    The last can fall up to one control step after the end of the trial.
    None when no signal file holds a world time, or one does not parse.
    """
    read = _signal_times(episode, _world_time_column)
    if read is None:
        return None
    if not read:
        log.error(
            '%s holds no sample in a %r or a %r column; this episode has no label',
            episode,
            _WORLD_TIME_COLUMN,
            LEGACY_TIME_COLUMN,
        )
        return None
    return min(times[0].as_py() for _, times in read), max(times[-1].as_py() for _, times in read)


# The one time column of a file in the format before named timelines: a signal's, then a video frames index's.
_LEGACY_TIME_COLUMNS = (LEGACY_TIME_COLUMN, LEGACY_FRAMES_TIME_COLUMN)


def _recording_time_column(signal: Path) -> str | None:
    """The column of `signal` on the timeline that `recording_timeline` picks from those it carries, or None.

    A file in the format before named timelines carries its one time column on the timeline that its metadata names,
    else on `RECORDED_TIME`. ArrowInvalid when the file does not parse.
    """
    schema = pq.read_schema(signal)
    names = schema.names
    timelines = {
        name.removeprefix(TIMESTAMP_COLUMN_PREFIX): name for name in names if name.startswith(TIMESTAMP_COLUMN_PREFIX)
    }
    if not timelines:
        legacy = next((name for name in _LEGACY_TIME_COLUMNS if name in names), None)
        declared = (schema.metadata or {}).get(TIMELINE_METADATA_KEY, RECORDED_TIME.encode()).decode()
        timelines = {declared: legacy} if legacy is not None else {}
    return timelines.get(recording_timeline(timelines))


# The signals of the marks that a person makes. A mark is written when it is made, so it is never a held reading.
_MARK_SIGNAL_PREFIXES = tuple(f'{channel}.' for channel in MARK_CHANNELS)
# A first sample that the next sample of its signal follows by more than this is a held reading. A held reading
# arrived before the try, and a recording can carry it at its arrival time.
_HELD_READING_GAP_NS = 1_000_000_000


def _time_that_starts_the_try(signal: Path, times: pa.ChunkedArray) -> int | None:
    """The world time of the first sample of `signal`, or of its second where the first is a held reading.

    None for a signal other than a mark with one sample.
    """
    first = times[0].as_py()
    if signal.stem.startswith(_MARK_SIGNAL_PREFIXES):
        return first
    if len(times) < 2:
        return None
    second = times[1].as_py()
    return first if second - first <= _HELD_READING_GAP_NS else second


class TryWindow(NamedTuple):
    """The world times of the first and the last sample of a try, in nanoseconds."""

    start_ns: int
    end_ns: int


def recorded_try_window(episode: Path) -> TryWindow | None:
    """The world times of the first and the last sample of the try that `episode` records.

    None when no signal places the start of the try, or a signal file does not parse.
    """
    read = _signal_times(episode, _recording_time_column)
    if read is None:
        return None
    starts = [start for signal, times in read if (start := _time_that_starts_the_try(signal, times)) is not None]
    if not starts:
        log.error('%s holds no signal that places the start of its try; this episode has no try window', episode)
        return None
    return TryWindow(min(starts), max(times[-1].as_py() for _, times in read))


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

    None when an operator ended the trial, or the episode records no verdict, no id or no world time.
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


def _end_label(episode: Path, task_key: str, annotator: LabelerRef) -> Label | None:
    """The label of a trial that succeeds or fails at its end, or runs out of time, with its task under `task_key`."""
    trial = _recorded_sim_trial(episode, task_key)
    if trial is None:
        return None
    return Label(
        uid=trial.uid,
        task=trial.task,
        annotator=annotator,
        seconds_to_end=trial.seconds_to_end,
        events=trial.end_events,
    )


MOLMO_LABELER = LabelerRef('molmo')


def molmo_label(episode: Path) -> Label | None:
    """MolmoSpaces' labeler: a trial succeeds or fails at its end, or runs out of time.

    A trial that an operator ended has no label.
    """
    return _end_label(episode, TASK, MOLMO_LABELER)


ABC_LABELER = LabelerRef('abc')


def abc_label(episode: Path) -> Label | None:
    """ABC's labeler: a trial ends as in `molmo_label`, and its task is the task id that ABC names."""
    return _end_label(episode, BENCHMARK_TASK, ABC_LABELER)


def abc_max_progress(episode: Path) -> float | None:
    """The largest progress that ABC's evaluator recorded in `episode`, from 0 to 1, as ABC scores progress.

    None when the episode records no progress sample, or its signal does not parse.
    """
    signal = episode / f'{ABC_REWARD_SIGNAL}{SIGNAL_FILE_SUFFIX}'
    if not signal.is_file():
        return None
    try:
        rewards = pq.read_table(signal, columns=[VALUE_COLUMN]).column(VALUE_COLUMN)
    except pa.ArrowInvalid:
        # One damaged upload leaves the other episodes measured, so this is logged, not raised.
        log.error('%s does not parse; this episode has no progress', signal)
        return None
    return max((reward for reward in rewards.to_pylist() if reward is not None), default=None)


ROBOLAB_LABELER = LabelerRef('robolab')


class _SubtaskSample(NamedTuple):
    """One sample of the `subtask` signal."""

    time_ns: int  # on the world clock
    completed: int
    total: int
    score: float


def _timed_values(signal: Path) -> list[tuple[int, Any]]:
    """Each sample of `signal`, which `recorded_span` has parsed, as its world time and its value.

    Empty when the episode does not record the signal, or the signal holds no values.
    """
    column = _world_time_column(signal) if signal.is_file() else None
    if column is None:
        return []
    if VALUE_COLUMN not in pq.read_schema(signal).names:
        # One damaged upload leaves the other signals read, so this is logged, not raised.
        log.error('%s holds no %s column; it counts as not recorded', signal, VALUE_COLUMN)
        return []
    table = pq.read_table(signal, columns=[column, VALUE_COLUMN])
    return list(zip(table.column(column).to_pylist(), table.column(VALUE_COLUMN).to_pylist(), strict=True))


def _subtask_samples(episode: Path) -> list[_SubtaskSample]:
    """Each sample of the `subtask` signal."""
    return [
        _SubtaskSample(
            time_ns,
            int(vector[SUBTASK_COMPLETED_INDEX]),
            int(vector[SUBTASK_TOTAL_INDEX]),
            float(vector[SUBTASK_SCORE_INDEX]),
        )
        for time_ns, vector in _timed_values(episode / f'{SUBTASK_SIGNAL}{SIGNAL_FILE_SUFFIX}')
    ]


class _Condition(NamedTuple):
    """One condition of a chain that RoboLab tracks, as a `gt_state` progress signal names it."""

    subtask: int
    group: str
    position: int  # in the group's chain
    name: str


class _SceneState(NamedTuple):
    """The samples of an episode's `gt_state` signals, by object and by condition."""

    poses: dict[str, list[tuple[int, Any]]]
    contacts: dict[str, list[tuple[int, Any]]]
    progress: dict[_Condition, list[tuple[int, Any]]]


def _parsed_condition(name: str) -> _Condition | None:
    """The condition that a progress signal's `<subtask>.<group>.<index>.<condition>` names. A group can hold dots."""
    subtask, _, rest = name.partition('.')
    group, index, condition = (rest.rsplit('.', 2) + ['', '', ''])[:3]
    if not (subtask.isdecimal() and index.isdecimal() and group and condition):
        return None
    return _Condition(int(subtask), group, int(index), condition)


def _scene_state(episode: Path) -> _SceneState:
    """The `gt_state` signals that `episode` records. A field that this labeler does not read is left out."""
    objects = f'{GT_STATE_SIGNAL}.{GT_STATE_OBJECTS}.'
    progress = f'{GT_STATE_SIGNAL}.{GT_STATE_PROGRESS}.'
    state = _SceneState({}, {}, {})
    for signal in sorted(episode.glob(f'{GT_STATE_SIGNAL}.*{SIGNAL_FILE_SUFFIX}')):
        name = signal.name.removesuffix(SIGNAL_FILE_SUFFIX)
        if name.startswith(objects):
            item, _, field = name.removeprefix(objects).rpartition('.')
            fields = {GT_STATE_POSE: state.poses, GT_STATE_GRIPPER_CONTACT: state.contacts}
            if field in fields:
                fields[field][item] = _timed_values(signal)
        elif name.startswith(progress):
            condition = _parsed_condition(name.removeprefix(progress))
            if condition is None:
                # One misnamed signal leaves the episode's other stages labelled, so this is logged, not raised.
                log.error('%s names no condition as <subtask>.<group>.<index>.<condition>; it gives no stage', signal)
                continue
            state.progress[condition] = _timed_values(signal)
    return state


class _Sample(NamedTuple):
    """One sample of a signal: its world time and its value."""

    time_ns: int
    value: Any


class _Series:
    """The samples of one signal, in time order, read by the frame they fall in."""

    # The signals of one recorded frame share a world time, or differ by microseconds when the env emits them one
    # after another. The frames are 1/15 s apart.
    _SAME_FRAME_NS = 10_000_000

    def __init__(self, samples: Sequence[tuple[int, Any]]):
        self._times = [time_ns for time_ns, _ in samples]
        self._values = [value for _, value in samples]

    def sample_at(self, time_ns: int) -> _Sample | None:
        """The last sample in the frame at `time_ns` or before it, or None."""
        index = bisect_right(self._times, time_ns + self._SAME_FRAME_NS) - 1
        return _Sample(self._times[index], self._values[index]) if index >= 0 else None


class _Mark(NamedTuple):
    """The frame where the arm is at `stage` in `subtask`, at the time of the frame's `subtask` sample."""

    subtask: int
    stage: Stage
    time_ns: int


class _StageReader:
    """Reads the stage marks of one RoboLab episode from its `gt_state`, `subtask` and end effector signals."""

    # The arm reaches an object when its end effector comes this near to the object's origin.
    _REACH_M = 0.10

    def __init__(self, episode: Path, subtasks: list[_SubtaskSample]):
        self._state = _scene_state(episode)
        self._counts = _Series([(sample.time_ns, sample.completed) for sample in subtasks])
        self._ee = _Series(_timed_values(episode / f'{ROBOT_STATE}{EE_POSE_SUFFIX}{SIGNAL_FILE_SUFFIX}'))
        self._chain_ends: dict[tuple[int, str], int] = {}
        for condition in self._state.progress:
            key = (condition.subtask, condition.group)
            self._chain_ends[key] = max(self._chain_ends.get(key, 0), condition.position)
        # Per subtask, the groups of its chains that name an object, or every object when no group does. A subtask
        # that records no chain, as after the last subtask, has none.
        self._objects: dict[int, set[str]] = {}
        for subtask, group in self._chain_ends:
            self._objects.setdefault(subtask, set()).add(group)
        everything = set(self._state.poses)
        for subtask, groups in self._objects.items():
            self._objects[subtask] = (groups & everything) or everything

    def frame_at(self, time_ns: int) -> tuple[int, int]:
        """The current subtask at `time_ns`, and the time of its frame's `subtask` sample, where attempts split."""
        sample = self._counts.sample_at(time_ns)
        return (sample.value, sample.time_ns) if sample is not None else (0, time_ns)

    def marks(self) -> Iterator[_Mark]:
        yield from self._condition_marks()
        yield from self._contact_marks()
        yield from self._reach_marks()

    def _condition_marks(self) -> Iterator[_Mark]:
        for condition, samples in self._state.progress.items():
            last = condition.position == self._chain_ends[(condition.subtask, condition.group)]
            for time_ns, done in samples:
                subtask, frame_ns = self.frame_at(time_ns)
                if done is not True or subtask != condition.subtask:
                    continue
                if condition.name == GT_STATE_GRABBED:
                    yield _Mark(subtask, Stage.control, frame_ns)
                if last:
                    yield _Mark(subtask, Stage.at_target, frame_ns)

    def _contact_marks(self) -> Iterator[_Mark]:
        for item, samples in self._state.contacts.items():
            for time_ns, touching in samples:
                subtask, frame_ns = self.frame_at(time_ns)
                if touching is True and item in self._objects.get(subtask, ()):
                    yield _Mark(subtask, Stage.contact, frame_ns)

    def _reach_marks(self) -> Iterator[_Mark]:
        for item, samples in self._state.poses.items():
            for time_ns, pose in samples:
                subtask, frame_ns = self.frame_at(time_ns)
                effector = self._ee.sample_at(time_ns)
                if item not in self._objects.get(subtask, ()) or effector is None:
                    continue
                if isinstance(effector.value, list) and isinstance(pose, list):
                    if math.dist(effector.value[:3], pose[:3]) <= self._REACH_M:
                        yield _Mark(subtask, Stage.reaching, frame_ns)


def _robolab_stages(episode: Path, start_ns: int, subtasks: list[_SubtaskSample]) -> list[StageReached]:
    """The first mark of each stage in each subtask, from the episode's `gt_state` signals."""
    first: dict[tuple[int, Stage], int] = {}
    for mark in _StageReader(episode, subtasks).marks():
        key = (mark.subtask, mark.stage)
        first[key] = min(first.get(key, mark.time_ns), mark.time_ns)
    ordered = sorted(first.items(), key=lambda entry: (entry[1], entry[0][1]))
    return [StageReached(seconds=(time_ns - start_ns) / 1e9, stage=stage) for (_, stage), time_ns in ordered]


def robolab_label(episode: Path) -> Label | None:
    """RoboLab's labeler: a trial ends as in `molmo_label`, the `subtask` signal gives its progress, and the
    `gt_state` signals give its stages.

    A `SubtaskAchieved` event sits at each sample where the count of subtasks completed changes, with the score
    of that sample. A change of the score alone is not an event, so the label's `subtask_score` is the score at
    the last change of the count. `_robolab_stages` gives the `StageReached` events. An episode that records no
    `gt_state` has none. A trial whose `subtask` signal holds no sample has no label.
    """
    trial = _recorded_sim_trial(episode, BENCHMARK_TASK)
    samples = _subtask_samples(episode) if trial is not None else []
    if trial is None or not samples:
        return None
    events: list[Event] = []
    achieved = 0
    for sample in samples:
        if sample.completed != achieved:
            achieved = sample.completed
            seconds = (sample.time_ns - trial.start_ns) / 1e9
            events.append(SubtaskAchieved(seconds=seconds, achieved=achieved, score=sample.score))
    events += _robolab_stages(episode, trial.start_ns, samples)
    events.sort(key=lambda event: (event.seconds, isinstance(event, StageReached)))
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
    ABC_LABELER: abc_label,
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
