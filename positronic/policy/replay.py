"""Offline replay of a journaled episode, and branches that change chosen activity outcomes.

``verify`` reruns a fresh policy against a journal and stops at the first event that differs.
``branch`` writes a new journal in which chosen outcomes are replaced or recomputed, and reports the scopes
that changed. Neither starts a world, prepares a robot, or runs a recorded activity. A ``RerunActivity``
runs its own function, and only when it allows execution.

A replay supplies what the harness supplied: the times of the startup, the turns and the close, the
observations, the published outcomes, the emitted commands and the termination. It compares what the
policy produces. A match shows that the policy reproduces from these inputs. The caller supplies the same
policy code and initial state. A policy that reads a clock, a random generator, a file or a network
outside its runtime, observations and answers can diverge, and the journal cannot show why. A branch
keeps the recorded times and observations: it shows the decisions of the policy against the recorded
history, not what a robot would do under the changed commands.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from positronic.policy.base import Answer, Obs, Policy, ProcessorRun, Runtime, Step
from positronic.policy.harness import clamp_wake
from positronic.policy.journal import (
    Activity,
    Cancelled,
    CancelRequested,
    Capture,
    Closing,
    CommandEmitted,
    Ended,
    Event,
    Finished,
    Journal,
    JournalAnswer,
    JournalSink,
    JournalWriter,
    Parent,
    Primed,
    Published,
    Raised,
    Recording,
    ReplacedResult,
    Reran,
    ResultRead,
    Returned,
    Started,
    StartFailed,
    Startup,
    StepReturned,
    Stopped,
    Submitted,
    TurnFailed,
    TurnLog,
    TurnStarted,
    qualified_name,
)


class ReplayError(Exception):
    """A replay cannot go on."""


class ReplayDivergence(ReplayError):
    """The replay produced an event that differs from the journal's event at ``index``."""

    def __init__(self, index: int, expected: Event | None, actual: Event | None, detail: str = '') -> None:
        super().__init__(f'Replay diverged at journal event {index}: expected {expected!r}, got {actual!r}{detail}')
        self.index = index
        self.expected = expected
        self.actual = actual


class MissingResult(ReplayError):
    """A branch submitted an activity that the source journal has no outcome for."""


class MissingInput(ReplayError):
    """A rerun needs an input that the source journal did not retain."""


class ExecutionRefused(ReplayError):
    """A rerun was requested without permission to run its function."""


class ReplayRuntime(Runtime):
    """A runtime whose scopes and activity outcomes come from a journal. It never calls a submitted function."""

    def __init__(self, journal: Journal, sink: JournalSink) -> None:
        self.log = TurnLog(journal, sink, self._cancel_work)

    @property
    def time_ns(self) -> int:
        return self.log.time_ns

    @property
    def tick(self) -> int:
        return self.log.tick

    @property
    def invocation(self) -> int:
        return self.log.invocation

    @property
    def journaled(self) -> bool:
        return True

    def submit(self, function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Answer[Any]:
        if not isinstance(function, Activity):
            raise TypeError(f'A journaled episode submits only an Activity, not {function!r}')
        answer, _, _ = self.log.submit(function, args, kwargs)
        return answer

    @staticmethod
    def _cancel_work(submission: int) -> None:
        """The journal holds the outcome of every cancellation request, so there is no work to stop."""

    def begin_turn(self, turn: TurnStarted, observation: bytes) -> Obs:
        return self.log.begin(turn.invocation, turn.time_ns, turn.tick, turn.wake, observation)

    def end_turn(self, step: Step) -> StepReturned:
        return self.log.end(step, clamp_wake(self.time_ns, step.resume_at_ns))


@dataclass(frozen=True)
class _Turn:
    started: TurnStarted
    published: tuple[Published, ...]
    ended: StepReturned | TurnFailed
    emitted: tuple[CommandEmitted, ...] = ()


@dataclass(frozen=True)
class _Plan:
    """The harness inputs of the closed scopes of a journal, and the index after the events they cover.

    ``startup`` is ``None`` when the journal ends before the startup does. ``end`` is ``None`` when the
    journal ends before its ``Ended`` event.
    """

    startup: tuple[Startup, Primed | StartFailed] | None
    turns: tuple[_Turn, ...]
    end: tuple[Closing, Ended] | None
    covered: int


# For each kind of event, the scope events (those that open or close a scope) that can come last before it.
_FOLLOWS: dict[type, tuple[type, ...]] = {
    Startup: (Started,),
    Primed: (Startup,),
    StartFailed: (Startup,),
    TurnStarted: (Primed, StepReturned),
    Published: (TurnStarted,),
    Submitted: (Startup, TurnStarted),
    ResultRead: (Startup, TurnStarted, Closing),
    CancelRequested: (Startup, TurnStarted, Closing),
    StepReturned: (TurnStarted,),
    CommandEmitted: (StepReturned,),
    TurnFailed: (TurnStarted,),
    Closing: (Primed, StartFailed, StepReturned, TurnFailed),
    Ended: (Closing,),
}
_SCOPE_EVENTS = Startup | Primed | StartFailed | TurnStarted | StepReturned | TurnFailed | Closing | Ended
# The events that carry the invocation current in their scope: -1 at startup, the last turn's at close.
_INVOCATION_EVENTS = Published | Submitted | ResultRead | CancelRequested | StepReturned | CommandEmitted | TurnFailed


def _plan(recording: Recording) -> _Plan:
    """Read the scopes of ``recording``. Raise ``ValueError`` at an event out of order."""
    events = recording.events
    startup: tuple[Startup, Primed | StartFailed] | None = None
    turns: list[_Turn] = []
    published: list[Published] = []
    end: tuple[Closing, Ended] | None = None
    last: Event = events[0]
    invocation = -1
    covered = 1
    for index, event in enumerate(events[1:], start=1):
        if isinstance(event, TurnStarted):
            invocation += 1
        if not isinstance(last, _FOLLOWS.get(type(event), ())) or (
            isinstance(event, _INVOCATION_EVENTS) and event.invocation != invocation
        ):
            raise ValueError(f'{recording.journal.path}: event {index} {event!r} cannot follow {last!r}')
        match event:
            case Primed() | StartFailed():
                assert isinstance(last, Startup)
                startup, covered = (last, event), index + 1
            case TurnStarted():
                published = []
            case Published():
                published.append(event)
            case StepReturned() | TurnFailed():
                assert isinstance(last, TurnStarted)
                turns.append(_Turn(last, tuple(published), event))
                covered = index + 1
            case CommandEmitted():
                turns[-1] = replace(turns[-1], emitted=(*turns[-1].emitted, event))
                covered = index + 1
            case Ended():
                assert isinstance(last, Closing)
                end, covered = (last, event), index + 1
        if isinstance(event, _SCOPE_EVENTS):
            last = event
    return _Plan(startup, tuple(turns), end, covered)


def _check_replayable(policy: Policy, recording: Recording) -> None:
    if qualified_name(policy) != recording.started.policy:
        raise ReplayError(f'The journal records policy {recording.started.policy}, not {qualified_name(policy)}')


def _close_unrecorded(run: ProcessorRun[Obs, Step] | None) -> None:
    """Close ``run`` where the journal records no close, and log an error of its finalizers."""
    if run is None:
        return
    try:
        run.close()
    # rules-allow: swallowed-error — no recorded event covers this close, so its error is not a replay result.
    except Exception:
        logging.exception('The policy failed to close outside the recorded part of the journal')


@dataclass(frozen=True)
class _Departure:
    """A scope whose outcome leaves the recorded history.

    ``error`` is a policy failure that the journal does not record. ``None`` means that the policy went on where
    the journal records its failure, so no recorded history follows.
    """

    error: Exception | None


def _start(
    policy: Policy, runtime: ReplayRuntime, startup: tuple[Startup, Primed | StartFailed]
) -> tuple[ProcessorRun[Obs, Step] | None, _Departure | None]:
    """Prime ``policy`` at the recorded startup; return its run, or ``None`` if it fails to start, and how its
    startup leaves the recorded history, or ``None`` if it ends as recorded."""
    started, recorded = startup
    runtime.log.begin_startup(started.time_ns)
    try:
        run = runtime.start(policy)
    except ReplayError:
        raise
    # rules-allow: swallowed-error — the failure is the startup's outcome, which the journal records.
    except Exception as exc:
        return None, (None if runtime.log.start_failed(exc) == recorded else _Departure(exc))
    try:
        runtime.log.primed()
    except BaseException:
        _close_unrecorded(run)
        raise
    return run, (_Departure(None) if isinstance(recorded, StartFailed) else None)


# What a turn publishes for a submission: its encoded result, its failure or its cancellation.
_Publication = bytes | Raised | Cancelled


def _recorded(recording: Recording, event: Published) -> _Publication:
    match event.outcome:
        case Returned(result=result):
            return recording.payload(result)
        case Raised() | Cancelled():
            return event.outcome


def _publish(log: TurnLog, answer: JournalAnswer[Any], publication: _Publication) -> None:
    match publication:
        case bytes():
            log.publish_result(answer, publication)
        case Raised():
            log.publish_failure(answer, publication)
        case Cancelled():
            log.publish_cancelled(answer)


def _play_turns(
    run: ProcessorRun[Obs, Step],
    runtime: ReplayRuntime,
    recording: Recording,
    turns: Sequence[_Turn],
    resolve: Callable[[Published, JournalAnswer[Any]], _Publication],
) -> tuple[int, _Departure | None]:
    """Play the recorded turns to ``run`` until one fails; return how many it played, and how the last played turn
    leaves the recorded history, or ``None`` if the turns end as recorded.

    As live, an error while a turn publishes or steps is the turn's failure. The recorded commands of a turn
    count as emitted when the replayed step matches the recorded one.
    """
    played = 0
    for turn in turns:
        played += 1
        obs = runtime.begin_turn(turn.started, recording.payload(turn.started.observation))
        publications = [
            (answer, resolve(event, answer))
            for event in turn.published
            if (answer := runtime.log.pending.get(event.submission)) is not None
        ]
        try:
            for answer, publication in publications:
                _publish(runtime.log, answer, publication)
            step = run.send(obs)
            assert step is not None, 'a policy must yield a Step for each observation'
            returned = runtime.end_turn(step)
        except ReplayError:
            raise
        # rules-allow: swallowed-error — the failure is the turn's outcome, which the journal records.
        except Exception as exc:
            if runtime.log.fail(exc) != turn.ended:
                return played, _Departure(exc)
            break
        if returned == turn.ended:
            for emitted in turn.emitted:
                runtime.log.emitted(emitted.command)
        elif isinstance(turn.ended, TurnFailed):
            return played, _Departure(None)
    return played, None


def _replay(
    policy: Policy,
    runtime: ReplayRuntime,
    recording: Recording,
    plan: _Plan,
    resolve: Callable[[Published, JournalAnswer[Any]], _Publication],
) -> int:
    """Play the closed scopes of ``plan`` to ``policy``; return how many turns it played.

    ``resolve`` gives what to publish for each recorded outcome whose submission the policy made again. The
    policy closes in the recorded close, with the recorded termination. A startup or turn failure that the
    journal does not record ends the episode with its error, closed at the time of that scope, as live. A
    policy that goes on where the journal records its failure has no recorded history left, so nothing records
    its close or termination. Nor is anything recorded after an error, or in a journal that ends early. Such
    a close is at the time of the last played scope.
    """
    if plan.startup is None:
        return 0
    started, _ = plan.startup
    run, departure = _start(policy, runtime, plan.startup)
    termination: Finished | Stopped | Raised | None = None
    try:
        played = 0
        if run is not None and departure is None:
            played, departure = _play_turns(run, runtime, recording, plan.turns, resolve)
        if departure is None and plan.end is not None:
            closing, ended = plan.end
            closing_ns, termination = closing.time_ns, ended.termination
        else:
            closing_ns = plan.turns[played - 1].started.time_ns if played else started.time_ns
            if departure is not None and departure.error is not None:
                termination = Raised.of(departure.error)
        runtime.log.begin_closing(closing_ns)
    except BaseException:
        _close_unrecorded(run)
        raise
    if termination is None:
        _close_unrecorded(run)
        return played
    if run is not None:
        run.close()
    if isinstance(termination, Finished):
        runtime.log.retain(recording.payload(termination.payload))
    runtime.log.finish(termination, runtime.metadata)
    return played


class _Expected(JournalSink):
    """Compare each event with the journal's next one, and raise at the first that differs.

    Events past ``covered`` belong to no closed scope of the journal, and are not compared. Nor are the
    events after a divergence, which the policy's finalizers can add while the divergence propagates.
    """

    def __init__(self, recording: Recording, covered: int) -> None:
        self._recording = recording
        self._covered = covered
        self._next = 1
        self._payloads: dict[str, bytes] = {}
        self._diverged = False

    def retain(self, payload_digest: str, payload: bytes) -> None:
        self._payloads[payload_digest] = payload

    def append(self, event: Event) -> None:
        if self._diverged or self._next >= self._covered:
            return
        expected = self._recording.events[self._next]
        if event != expected:
            self._diverged = True
            raise ReplayDivergence(self._next, expected, event, self._commands_detail(expected, event))
        self._next += 1
        self._payloads.clear()

    def _commands_detail(self, expected: Event, actual: Event) -> str:
        if not (isinstance(expected, StepReturned) and isinstance(actual, StepReturned)):
            return ''
        if expected.commands == actual.commands:
            return ''
        codec = self._recording.journal.commands
        recorded = codec.decode(self._recording.payload(expected.commands))
        replayed = codec.decode(self._payloads[actual.commands])
        return f'; recorded commands {recorded!r}, replayed {replayed!r}'

    def close(self) -> None:
        pass

    def check_end(self) -> None:
        if self._next != self._covered:
            raise ReplayDivergence(self._next, self._recording.events[self._next], None)


@dataclass(frozen=True)
class Verified:
    """A replay that matched ``turns`` turns of a journal.

    ``termination`` is how the recorded episode ended, matched with its close; it is ``None`` when the
    journal ends early and the replay matched it through its last closed scope.
    """

    turns: int
    termination: Finished | Stopped | Raised | None

    @property
    def complete(self) -> bool:
        return self.termination is not None


def verify(policy: Policy, journal: Journal) -> Verified:
    """Rerun a fresh ``policy`` against ``journal``; raise ``ReplayDivergence`` at the first difference."""
    recording = journal.read()
    _check_replayable(policy, recording)
    plan = _plan(recording)
    expected = _Expected(recording, plan.covered)
    runtime = ReplayRuntime(journal, expected)
    turns = _replay(policy, runtime, recording, plan, lambda event, answer: _recorded(recording, event))
    expected.check_end()
    if plan.end is None:
        return Verified(turns=turns, termination=None)
    _, ended = plan.end
    return Verified(turns=turns, termination=ended.termination)


@dataclass(frozen=True)
class ReplaceResult:
    """Publish ``result`` in place of the recorded outcome of ``submission``, at the turn that published it."""

    submission: int
    result: Any


@dataclass(frozen=True)
class RerunActivity:
    """Run ``function`` on the retained input of ``submission`` and publish its outcome at the recorded turn.

    ``version`` names the implementation. ``allow_execution`` must be true, because the function can use
    a GPU, a network or a paid service. Its wall time does not move the branch's clock. If it raises, the
    turn publishes the failure, as a live turn does. ``function`` replaces the function of the submitted
    ``Activity``: for one that a ``Codec`` wraps, it takes the observation and returns the decoded result.
    """

    submission: int
    function: Callable[..., Any]
    version: int
    allow_execution: bool = False


@dataclass(frozen=True)
class Difference:
    """A scope whose events differ between the source and the branch.

    ``invocation`` is the turn's invocation, -1 for the startup, or ``None`` for the close.
    """

    invocation: int | None
    source: tuple[Event, ...]
    branch: tuple[Event, ...]


@dataclass(frozen=True)
class Branch:
    """The branch journal, and the scopes at which it differs from its source."""

    journal: Journal
    differences: tuple[Difference, ...]


class _BranchSink(JournalSink):
    """Write the branch journal; raise at a submission that the source records differently or not at all.

    The journal ends before that submission, and the events of the policy's finalizers after it are dropped.
    """

    def __init__(self, writer: JournalWriter, recording: Recording) -> None:
        self._writer = writer
        self._submitted = {event.submission: event for event in recording.submissions()}
        self.events: list[Event] = []
        self._missing = False

    def retain(self, payload_digest: str, payload: bytes) -> None:
        self._writer.retain(payload_digest, payload)

    def append(self, event: Event) -> None:
        if self._missing:
            return
        if isinstance(event, Submitted) and (source := self._submitted.get(event.submission)) != event:
            self._missing = True
            raise MissingResult(
                f'The branch submits {event!r}, and the source journal records {source!r}. '
                'No recorded outcome matches this operation, version, turn and input'
            )
        self._writer.append(event)
        self.events.append(event)

    def close(self) -> None:
        self._writer.close()


def _changes_by_submission(
    recording: Recording, changes: Sequence[ReplaceResult | RerunActivity]
) -> dict[int, ReplaceResult | RerunActivity]:
    submitted = {event.submission: event for event in recording.submissions()}
    published = {event.submission for event in recording.events if isinstance(event, Published)}
    by_submission: dict[int, ReplaceResult | RerunActivity] = {}
    for change in changes:
        source = submitted.get(change.submission)
        if source is None:
            raise ReplayError(f'{recording.journal.path} has no submission {change.submission}')
        if change.submission in by_submission:
            raise ReplayError(f'Submission {change.submission} has more than one change')
        if change.submission not in published:
            raise ReplayError(f'Submission {change.submission} was never published, so no turn can publish a change')
        if isinstance(change, RerunActivity):
            if not change.allow_execution:
                raise ExecutionRefused(f'Rerunning submission {change.submission} needs allow_execution=True')
            if source.capture is not Capture.INPUT_AND_RESULT:
                raise MissingInput(f'Submission {change.submission} retained no input: it captured {source.capture}')
        by_submission[change.submission] = change
    return by_submission


def _encoded(answer: JournalAnswer[Any], value: Any) -> bytes:
    try:
        return answer.codec.encode(value)
    except Exception as exc:
        raise ReplayError(
            f'The changed result of submission {answer.submission} does not encode with {answer.codec.NAME}'
        ) from exc


def _differences(source: Sequence[Event], branch: Sequence[Event]) -> tuple[Difference, ...]:
    def by_scope(events: Sequence[Event]) -> dict[int | None, tuple[Event, ...]]:
        scopes: dict[int | None, list[Event]] = {}
        invocation: int | None = -1
        for event in events:
            if isinstance(event, Started):
                continue
            if isinstance(event, TurnStarted):
                invocation = event.invocation
            elif isinstance(event, Closing):
                invocation = None
            scopes.setdefault(invocation, []).append(event)
        return {invocation: tuple(scope) for invocation, scope in scopes.items()}

    recorded, replayed = by_scope(source), by_scope(branch)
    return tuple(
        Difference(invocation, recorded.get(invocation, ()), replayed.get(invocation, ()))
        for invocation in dict.fromkeys([*recorded, *replayed])
        if recorded.get(invocation) != replayed.get(invocation)
    )


def branch(policy: Policy, source: Journal, target: Path, changes: Sequence[ReplaceResult | RerunActivity]) -> Branch:
    """Replay ``source`` with ``changes`` into a new journal at ``target``, and report the scopes that differ.

    Each change keeps the turn at which the source published its submission. Every other submission must
    match the source's submission: operation, version, turn, input, payload codec and capture. Otherwise the
    branch stops with ``MissingResult``. The branch keeps the source's emitted commands only for a step equal
    to the source's step. A startup or turn failure that the source does not record ends the branch with that
    error. A branch that goes on where the source failed has no recorded history left, so its journal ends
    without an ``Ended`` event. The source journal does not change.
    """
    recording = source.read()
    _check_replayable(policy, recording)
    plan = _plan(recording)
    by_submission = _changes_by_submission(recording, changes)
    journal = Journal(target, source.observations, source.commands)
    records = tuple(
        ReplacedResult(submission=change.submission)
        if isinstance(change, ReplaceResult)
        else Reran(submission=change.submission, version=change.version)
        for change in changes
    )
    started = Started.create(
        journal,
        policy,
        simulated=recording.started.simulated,
        charge_inference_time=recording.started.charge_inference_time,
        parent=Parent(journal=recording.started.journal, path=source.path, changes=records),
    )
    sink = _BranchSink(JournalWriter(journal, started), recording)
    runtime = ReplayRuntime(journal, sink)
    inputs = {event.submission: event.input for event in recording.submissions()}

    def resolve(event: Published, answer: JournalAnswer[Any]) -> _Publication:
        match by_submission.get(event.submission):
            case None:
                return _recorded(recording, event)
            case ReplaceResult(result=result):
                return _encoded(answer, result)
            case RerunActivity(function=function):
                args, kwargs = answer.codec.decode(recording.payload(inputs[event.submission]))
                try:
                    result = function(*args, **kwargs)
                # rules-allow: swallowed-error — the failure is the activity's outcome, which the turn publishes.
                except Exception as exc:
                    return Raised.of(exc)
                return _encoded(answer, result)

    try:
        _replay(policy, runtime, recording, plan, resolve)
    finally:
        sink.close()
    return Branch(journal, _differences(recording.events, sink.events))
