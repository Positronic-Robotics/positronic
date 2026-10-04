"""Journals of policy episodes, which an offline replay reruns a policy against.

A journaled episode records the startup of the policy, each turn and the close. A turn records the
observation the policy receives, the activity outcomes published to it, the activities it submits, the
results it reads and the step it returns. Inside the startup, a turn or the close, the runtime's time and
the published outcomes do not change. ``positronic.policy.replay`` reruns a fresh policy against the
record and does not run its activities.

Declare the work a policy submits where the policy gets the dependency::

    infer = Activity('step_plan', 1, model.infer)
    rollout = Rollout(task, Move(infer), output_path=None, journal=Journal(path))
"""

import hashlib
import itertools
import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from concurrent.futures import CancelledError
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from traceback import format_exception
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Generic, Literal, ParamSpec, TypeVar, cast

import msgpack
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from positronic.policy.base import Answer, NotAnswered, Obs, Processor, Step
from positronic.utils.git import get_package_git_state
from positronic.utils.serialization import deserialise, serialise, unpack

# The record schema, and the turn rules that decide what a policy sees within a turn.
FORMAT_VERSION = 1
SEMANTICS_VERSION = 1

_EVENTS = 'events.jsonl'
_PAYLOADS = 'payloads'

P = ParamSpec('P')
T = TypeVar('T')


class PayloadCodec(ABC):
    """Bytes for the values that cross one journaled boundary.

    The live run and its replay decode the same bytes. ``decode`` gives an activity function a copy that
    it owns. ``decode_frozen`` gives the policy a value that no code can change, so every read of it is
    the same. A domain type needs a codec of its own: ``PlainData`` refuses one.
    """

    NAME: ClassVar[str]
    VERSION: ClassVar[int] = 1

    @abstractmethod
    def encode(self, value: Any) -> bytes: ...

    @abstractmethod
    def decode(self, payload: bytes) -> Any: ...

    @abstractmethod
    def decode_frozen(self, payload: bytes) -> Any: ...


class PlainData(PayloadCodec):
    """Scalars, strings, containers with string keys, and numeric NumPy values.

    ``decode`` gives dicts and lists, and ``decode_frozen`` gives read-only mappings and tuples. Arrays
    decode read-only.
    """

    NAME = 'plain_data'

    def encode(self, value: Any) -> bytes:
        return serialise(value)

    def decode(self, payload: bytes) -> Any:
        return deserialise(payload)

    def decode_frozen(self, payload: bytes) -> Any:
        return msgpack.unpackb(payload, object_hook=self._frozen, use_list=False)

    @staticmethod
    def _frozen(obj: dict) -> Any:
        value = unpack(obj)
        return MappingProxyType(value) if isinstance(value, dict) else value


PLAIN_DATA = PlainData()


class Capture(Enum):
    """What a journal retains of an activity besides its result. The digest of the input is always recorded."""

    RESULT = 'result'
    INPUT_AND_RESULT = 'input_and_result'


@dataclass(frozen=True)
class Activity(Generic[P, T]):
    """Work that a policy submits, with the identity a journal records it under.

    ``operation`` and ``version`` name what ``function`` computes; increase ``version`` when the same input
    can give a different result. ``codec`` carries the input and the result. Calling the activity
    calls ``function``, so a runtime without a journal runs it as an ordinary function.
    """

    operation: str
    version: int
    function: Callable[P, T]
    capture: Capture = Capture.INPUT_AND_RESULT
    codec: PayloadCodec = PLAIN_DATA

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.function(*args, **kwargs)


class ActivityFailed(RuntimeError):
    """A journaled activity raised. Live and in replay, it holds the recorded message and has no cause.

    The journal keeps the traceback of the original error.
    """


class UnrecordableResult(TypeError):
    """An activity returned a value that its codec cannot encode."""


class Wake(Enum):
    """Why the harness calls the policy."""

    FIRST = 'first'
    DUE = 'due'
    COMPLETION = 'completion'


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')


class CodecId(_Record):
    name: str
    version: int

    @classmethod
    def of(cls, codec: PayloadCodec) -> 'CodecId':
        return cls(name=codec.NAME, version=codec.VERSION)


def describe_error(error: BaseException) -> str:
    return f'{type(error).__name__}: {error}'


class Returned(_Record):
    kind: Literal['returned'] = 'returned'
    result: str


class Raised(_Record):
    """An error: its type and message, and the traceback of the live run."""

    kind: Literal['raised'] = 'raised'
    error: str
    traceback: str

    @classmethod
    def of(cls, error: BaseException) -> 'Raised':
        return cls(error=describe_error(error), traceback=''.join(format_exception(error)))


class Cancelled(_Record):
    kind: Literal['cancelled'] = 'cancelled'


Outcome = Annotated[Returned | Raised | Cancelled, Field(discriminator='kind')]


class ReplacedResult(_Record):
    kind: Literal['replaced_result'] = 'replaced_result'
    submission: int


class Reran(_Record):
    kind: Literal['reran'] = 'reran'
    submission: int
    version: int


Change = Annotated[ReplacedResult | Reran, Field(discriminator='kind')]


class Parent(_Record):
    """The journal that a branch replays, and the changes the branch makes to it."""

    journal: str
    path: Path
    changes: tuple[Change, ...]


def qualified_name(value: object) -> str:
    return f'{type(value).__module__}.{type(value).__qualname__}'


class Started(_Record):
    """The first event: the journal's identity, the policy class, the code revision and the timing mode."""

    kind: Literal['started'] = 'started'
    journal: str
    format: int
    semantics: int
    policy: str
    provenance: dict[str, str | bool] | None
    simulated: bool
    charge_inference_time: bool
    observations: CodecId
    commands: CodecId
    parent: Parent | None = None

    @classmethod
    def create(
        cls,
        journal: 'Journal',
        policy: Processor,
        *,
        simulated: bool,
        charge_inference_time: bool,
        parent: Parent | None = None,
    ) -> 'Started':
        return cls(
            journal=uuid.uuid4().hex,
            format=FORMAT_VERSION,
            semantics=SEMANTICS_VERSION,
            policy=qualified_name(policy),
            provenance=get_package_git_state(),
            simulated=simulated,
            charge_inference_time=charge_inference_time,
            observations=CodecId.of(journal.observations),
            commands=CodecId.of(journal.commands),
            parent=parent,
        )


class Startup(_Record):
    """The start of the policy, with the time its runtime shows until the policy is primed.

    Events at startup carry invocation -1.
    """

    kind: Literal['startup'] = 'startup'
    time_ns: int


class Primed(_Record):
    kind: Literal['primed'] = 'primed'


class StartFailed(_Record):
    kind: Literal['start_failed'] = 'start_failed'
    error: str


class TurnStarted(_Record):
    kind: Literal['turn'] = 'turn'
    invocation: int
    time_ns: int
    tick: int
    wake: Wake
    observation: str


class Published(_Record):
    kind: Literal['published'] = 'published'
    invocation: int
    submission: int
    outcome: Outcome


class Submitted(_Record):
    kind: Literal['submitted'] = 'submitted'
    invocation: int
    submission: int
    operation: str
    version: int
    codec: CodecId
    capture: Capture
    input: str


class ResultRead(_Record):
    kind: Literal['read'] = 'read'
    invocation: int
    submission: int


class CancelRequested(_Record):
    kind: Literal['cancel'] = 'cancel'
    invocation: int
    submission: int


class StepReturned(_Record):
    """The step the policy returned, and the wake-up time the harness takes from it."""

    kind: Literal['step'] = 'step'
    invocation: int
    commands: str
    resume_at_ns: int
    wake_at_ns: int


class CommandEmitted(_Record):
    """The harness emitted ``command`` of the step that turn ``invocation`` returned."""

    kind: Literal['emitted'] = 'emitted'
    invocation: int
    command: str


class TurnFailed(_Record):
    kind: Literal['failed'] = 'failed'
    invocation: int
    error: str


class Closing(_Record):
    """The close of the policy after its work drained, with the time its runtime shows until the end.

    Events at close carry the invocation of the last turn.
    """

    kind: Literal['closing'] = 'closing'
    time_ns: int


class Finished(_Record):
    """The episode ended with the terminal payload of the harness: a done signal or the timeout."""

    kind: Literal['finished'] = 'finished'
    payload: str


class Stopped(_Record):
    """The episode closed without a terminal payload or an error, as at a stop request."""

    kind: Literal['stopped'] = 'stopped'


Termination = Annotated[Finished | Stopped | Raised, Field(discriminator='kind')]


class Ended(_Record):
    """The last event: how the episode ended, and the episode metadata after the policy closed."""

    kind: Literal['ended'] = 'ended'
    termination: Termination
    metadata: str


Event = Annotated[
    Started
    | Startup
    | Primed
    | StartFailed
    | TurnStarted
    | Published
    | Submitted
    | ResultRead
    | CancelRequested
    | StepReturned
    | CommandEmitted
    | TurnFailed
    | Closing
    | Ended,
    Field(discriminator='kind'),
]
_EVENT = TypeAdapter[Event](Event)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Journal:
    """The directory of one episode's journal, and the codecs of its observations and commands.

    The directory holds ``events.jsonl``, one event per line, and ``payloads/``, one file per payload
    named by its SHA-256 digest.
    """

    path: Path
    observations: PayloadCodec = PLAIN_DATA
    commands: PayloadCodec = PLAIN_DATA

    def read(self) -> 'Recording':
        return Recording(self)


class Recording:
    """A journal read back: its events in order and the payloads they name.

    The writer ends each event with a newline. A last line without one is a write that a crash cut
    short, and the reader drops it.
    """

    def __init__(self, journal: Journal) -> None:
        self.journal = journal
        *lines, torn = (journal.path / _EVENTS).read_bytes().split(b'\n')
        if torn:
            logging.warning('%s ends with a torn event; the reader drops it', journal.path)
        self.events: tuple[Event, ...] = tuple(_EVENT.validate_json(line) for line in lines)
        started = self.events[0] if self.events else None
        if not isinstance(started, Started):
            raise ValueError(f'{journal.path} does not start with a {Started.__name__} event')
        if (started.format, started.semantics) != (FORMAT_VERSION, SEMANTICS_VERSION):
            raise ValueError(
                f'{journal.path} has format {started.format} and turn semantics {started.semantics}; '
                f'this build reads format {FORMAT_VERSION} and turn semantics {SEMANTICS_VERSION}'
            )
        for name, codec, recorded in (
            ('observation', journal.observations, started.observations),
            ('command', journal.commands, started.commands),
        ):
            if CodecId.of(codec) != recorded:
                raise ValueError(f'{journal.path} records {name}s with {recorded}, not {CodecId.of(codec)}')
        self.started = started

    def payload(self, payload_digest: str) -> bytes:
        payload = (self.journal.path / _PAYLOADS / payload_digest).read_bytes()
        if digest(payload) != payload_digest:
            raise ValueError(f'Payload {payload_digest} in {self.journal.path} does not match its digest')
        return payload

    def submissions(self) -> tuple[Submitted, ...]:
        return tuple(event for event in self.events if isinstance(event, Submitted))


class JournalSink(ABC):
    """Where a ``TurnLog`` puts its events and the payloads they name."""

    @abstractmethod
    def retain(self, payload_digest: str, payload: bytes) -> None: ...

    @abstractmethod
    def append(self, event: Event) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class JournalWriter(JournalSink):
    """Write a new journal directory.

    Each event goes to the operating system when it is appended, so a process crash keeps the events
    before it and can tear the last line. The writer does not sync to disk: a power loss can lose more.
    """

    def __init__(self, journal: Journal, started: Started) -> None:
        self._payloads = journal.path / _PAYLOADS
        self._payloads.mkdir(parents=True, exist_ok=False)
        self._events = (journal.path / _EVENTS).open('x')
        self.append(started)

    def retain(self, payload_digest: str, payload: bytes) -> None:
        target = self._payloads / payload_digest
        if not target.exists():
            partial = target.with_suffix('.partial')
            partial.write_bytes(payload)
            partial.replace(target)

    def append(self, event: Event) -> None:
        self._events.write(event.model_dump_json() + '\n')
        self._events.flush()

    def close(self) -> None:
        self._events.close()


class JournalAnswer(Answer[T]):
    """An answer whose outcome becomes visible only when a turn publishes it, and stays the same after."""

    def __init__(self, log: 'TurnLog', submission: int, codec: PayloadCodec) -> None:
        self.submission = submission
        self.codec = codec
        self._log = log
        self._outcome: Returned | Raised | Cancelled | None = None
        self._value: Any = None
        self._read = False

    def done(self) -> bool:
        return self._outcome is not None

    def result(self) -> T:
        outcome = self._outcome
        if outcome is None:
            raise NotAnswered('No turn has published this answer')
        self._log.read(self)
        self._read = True
        match outcome:
            case Returned():
                return cast(T, self._value)
            case Raised():
                raise ActivityFailed(outcome.error) from None
            case Cancelled():
                raise CancelledError()

    def cancel(self) -> None:
        """Ask for the work to stop. The outcome arrives at a later turn; work already running may finish."""
        self._log.cancel(self)

    def unread_error(self) -> str | None:
        """The published failure, if the policy never read it."""
        return self._outcome.error if isinstance(self._outcome, Raised) and not self._read else None

    def _settle(self, outcome: Returned | Raised | Cancelled, value: Any) -> None:
        assert self._outcome is None, f'submission {self.submission} is published twice'
        self._outcome, self._value = outcome, value


class TurnLog:
    """One episode's journal events, written by a live runtime and compared by its replay.

    The policy runs in three kinds of scope: its startup, a turn and its close. A turn begins with the
    observation and the outcomes published to it, and ends with the policy's step or failure. The policy
    submits at startup and in a turn, and reads and cancels in every scope.

    The log holds an answer until a turn publishes it, and a published failure until the end. A published
    result lives as long as the policy holds its answer.
    """

    def __init__(self, journal: Journal, sink: JournalSink, cancel_work: Callable[[int], None]) -> None:
        self.journal = journal
        self.pending: dict[int, JournalAnswer[Any]] = {}
        self._failures: list[JournalAnswer[Any]] = []
        self._submissions = itertools.count()
        self.scope: Startup | TurnStarted | Closing | None = None
        self._startup: Startup | None = None
        self._turn: TurnStarted | None = None
        self._sink = sink
        self._cancel_work = cancel_work

    @property
    def time_ns(self) -> int:
        if self.scope is None:
            raise RuntimeError('A journaled policy has no time outside its startup, its turns and its close')
        return self.scope.time_ns

    @property
    def invocation(self) -> int:
        """The invocation of the last turn, or -1 before the first."""
        return -1 if self._turn is None else self._turn.invocation

    @property
    def tick(self) -> int:
        return -1 if self._turn is None else self._turn.tick

    def retain(self, payload: bytes) -> str:
        payload_digest = digest(payload)
        self._sink.retain(payload_digest, payload)
        return payload_digest

    def _current_turn(self) -> TurnStarted:
        if not isinstance(self.scope, TurnStarted):
            raise RuntimeError('No turn is open')
        return self.scope

    def begin_startup(self, time_ns: int) -> None:
        if self._startup is not None or self._turn is not None:
            raise RuntimeError('A journaled episode starts one policy, before its first turn')
        self.scope = self._startup = Startup(time_ns=time_ns)
        self._sink.append(self._startup)

    def primed(self) -> None:
        assert isinstance(self.scope, Startup), 'no startup is open'
        self._sink.append(Primed())
        self.scope = None

    def start_failed(self, error: BaseException) -> StartFailed:
        assert isinstance(self.scope, Startup), 'no startup is open'
        failed = StartFailed(error=describe_error(error))
        self._sink.append(failed)
        self.scope = None
        return failed

    def begin(self, invocation: int, time_ns: int, tick: int, wake: Wake, observation: bytes) -> Obs:
        """Start a turn with an encoded observation; return the observation that the policy receives.

        A failure leaves no turn open.
        """
        assert self.scope is None, f'{self.scope!r} is still open'
        owned = self.journal.observations.decode_frozen(observation)
        turn = TurnStarted(
            invocation=invocation, time_ns=time_ns, tick=tick, wake=wake, observation=self.retain(observation)
        )
        self._sink.append(turn)
        self.scope = self._turn = turn
        return owned

    def submit(
        self, activity: Activity[..., Any], args: tuple, kwargs: Mapping[str, Any]
    ) -> tuple[JournalAnswer[Any], tuple, dict[str, Any]]:
        """Record a submission; return its answer and the decoded arguments the work receives."""
        if not isinstance(self.scope, Startup | TurnStarted):
            raise RuntimeError('A journaled policy submits only at startup or inside a turn')
        payload = activity.codec.encode([list(args), dict(kwargs)])
        decoded_args, decoded_kwargs = activity.codec.decode(payload)
        payload_digest = digest(payload)
        if activity.capture is Capture.INPUT_AND_RESULT:
            self._sink.retain(payload_digest, payload)
        answer = JournalAnswer[Any](self, next(self._submissions), activity.codec)
        self.pending[answer.submission] = answer
        self._sink.append(
            Submitted(
                invocation=self.invocation,
                submission=answer.submission,
                operation=activity.operation,
                version=activity.version,
                codec=CodecId.of(activity.codec),
                capture=activity.capture,
                input=payload_digest,
            )
        )
        return answer, tuple(decoded_args), decoded_kwargs

    def _publish(
        self, answer: JournalAnswer[Any], outcome: Returned | Raised | Cancelled, value: Callable[[], Any]
    ) -> None:
        invocation = self._current_turn().invocation
        # The record comes before the decode, so a replay publishes the same outcome and meets the same failure.
        self._sink.append(Published(invocation=invocation, submission=answer.submission, outcome=outcome))
        del self.pending[answer.submission]
        answer._settle(outcome, value())
        if isinstance(outcome, Raised):
            self._failures.append(answer)

    def publish_result(self, answer: JournalAnswer[Any], payload: bytes) -> None:
        self._publish(answer, Returned(result=self.retain(payload)), partial(answer.codec.decode_frozen, payload))

    def publish_failure(self, answer: JournalAnswer[Any], raised: Raised) -> None:
        self._publish(answer, raised, lambda: None)

    def publish_cancelled(self, answer: JournalAnswer[Any]) -> None:
        self._publish(answer, Cancelled(), lambda: None)

    def _acting_invocation(self) -> int:
        if self.scope is None:
            raise RuntimeError('A journaled policy acts on answers only at startup, inside a turn or at close')
        return self.invocation

    def read(self, answer: JournalAnswer[Any]) -> None:
        self._sink.append(ResultRead(invocation=self._acting_invocation(), submission=answer.submission))

    def cancel(self, answer: JournalAnswer[Any]) -> None:
        self._sink.append(CancelRequested(invocation=self._acting_invocation(), submission=answer.submission))
        self._cancel_work(answer.submission)

    def end(self, step: Step, wake_at_ns: int) -> StepReturned:
        turn = self._current_turn()
        commands = self.retain(self.journal.commands.encode(step.commands))
        returned = StepReturned(
            invocation=turn.invocation, commands=commands, resume_at_ns=step.resume_at_ns, wake_at_ns=wake_at_ns
        )
        self._sink.append(returned)
        self.scope = None
        return returned

    def emitted(self, command: str) -> None:
        assert self.scope is None and self._turn is not None, 'commands are emitted after a turn'
        self._sink.append(CommandEmitted(invocation=self._turn.invocation, command=command))

    def fail(self, error: BaseException) -> TurnFailed:
        failed = TurnFailed(invocation=self._current_turn().invocation, error=describe_error(error))
        self._sink.append(failed)
        self.scope = None
        return failed

    def begin_closing(self, time_ns: int) -> None:
        assert self.scope is None, f'{self.scope!r} is still open'
        self.scope = Closing(time_ns=time_ns)
        self._sink.append(self.scope)

    def finish(self, termination: Finished | Stopped | Raised, metadata: Mapping[str, Any]) -> None:
        """Record how the episode ended and its metadata, and close the sink. Report failures never read."""
        assert isinstance(self.scope, Closing), 'the close is not open'
        for answer in self._failures:
            if (error := answer.unread_error()) is not None:
                logging.error('A submitted function failed without its result being read: %s', error)
        self._sink.append(Ended(termination=termination, metadata=self.retain(PLAIN_DATA.encode(metadata))))
        self.scope = None
        self._sink.close()
