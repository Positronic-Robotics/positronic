"""Reusable processors, their episode generators, and the runtime that serves them."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Mapping
from functools import partial
from typing import Any, ClassVar, Generic, ParamSpec, Self

from attr import dataclass, field
from eval_vocabulary.episode import PolicyEnd
from typing_extensions import TypeAliasType, TypeVar

from positronic import telemetry


class NotAnswered(RuntimeError):
    """The call has not answered yet. A read of an ``Answer`` raises this rather than waiting."""


T = TypeVar('T')


class Answer(ABC, Generic[T]):
    """The caller's handle on one call.

    pimm has an ``Answer`` of the same shape. The two are not interchangeable.
    """

    @abstractmethod
    def done(self) -> bool: ...

    @property
    @abstractmethod
    def elapsed_sec(self) -> float:
        """Submission to completion, including queueing; zero in uncharged simulation."""

    @property
    @abstractmethod
    def execution_sec(self) -> float:
        """Callable start to completion, excluding runtime queueing; zero in uncharged simulation."""

    @abstractmethod
    def result(self) -> T:
        """What the function returned. Raises what the function raised, or ``NotAnswered`` before it answers."""

    def cancel(self):
        """Cancel the call if anything can be cancelled."""
        pass


Obs = Mapping[str, Any]
Commands = Mapping[str, Any]


@dataclass
class Step:
    """Commands to emit now, auxiliary outputs, and the next requested call time.

    Any submitted call becoming ready may resume the policy sooner, including at the same clock time.
    """

    # TODO: Allow selecting which answers can wake the policy early, alongside its time deadline.
    commands: Commands
    resume_at_ns: int
    outputs: Mapping[str, Any] = field(factory=dict)


P = ParamSpec('P')
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')
ReturnT = TypeVar('ReturnT', default=Any)

ProcessorRun = TypeAliasType(
    'ProcessorRun', Generator[OutputT | None, InputT, ReturnT], type_params=(InputT, OutputT, ReturnT)
)


class Runtime(ABC):
    """What the framework offers one episode. Every episode gets its own.

    At episode shutdown, drain submitted work before closing live generators whose resources it may use.
    """

    @abstractmethod
    def set_metadata(self, name: str, value: Any) -> None:
        """Write a deep copy of an episode value on the control thread, replacing any value at this name."""

    @property
    @abstractmethod
    def time_ns(self) -> int:
        """The current time in nanoseconds."""
        pass

    @property
    @abstractmethod
    def tick(self) -> int:
        """The current tick number."""
        pass

    @staticmethod
    def _timed_run(run: ProcessorRun[InputT, OutputT, ReturnT], name: str) -> ProcessorRun[InputT, OutputT, ReturnT]:
        """Time each resumption, leaving no span active while the generator is suspended."""
        result = None
        try:
            while True:
                try:
                    value = yield result
                except GeneratorExit:
                    raise
                except BaseException as exc:
                    resume = partial(run.throw, exc)
                else:
                    resume = partial(run.send, value)
                with telemetry.span(name):
                    try:
                        result = resume()
                    except StopIteration as end:
                        return end.value
        finally:
            run.close()

    def start(
        self, processor: Processor[InputT, OutputT, ReturnT], /, *args: Any, **kwargs: Any
    ) -> ProcessorRun[InputT, OutputT, ReturnT]:
        """Create and prime an episode generator. The caller owns its closure.

        Bind telemetry before starting processors; timing wrappers are selected at startup.
        """
        run = processor.run(self, *args, **kwargs)
        initial = next(run)
        if initial is not None:
            run.close()
            raise AssertionError('a processor must yield None before receiving its first input')
        if not telemetry.enabled():
            return run
        timed = self._timed_run(run, telemetry.component_name(processor))
        next(timed)
        return timed

    @abstractmethod
    def submit(self, function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Answer[T]: ...

    @abstractmethod
    def has_ready_answers(self) -> bool:
        """Whether any non-cancelled call is ready and its result has not been read."""


class Processor(ABC, Generic[InputT, OutputT, ReturnT]):
    """A reusable computation definition with typed inputs and outputs.

    Constructors hold configuration. Each ``run`` creates one episode's generator, keeping mutable
    episode state in its locals. ``Runtime.start`` advances it to its first yield, which must be
    ``None``, after which ``send(input)`` returns one output. Dependencies are supplied live and already
    primed when they are generators. Whoever starts a generator owns its closure. A run ends when it returns. A policy
    run returns a ``PolicyEnd``, and a run around another run returns what that run returned.
    """

    # A receiver resolves this name through its registry of installed processor classes.
    WIRE_NAME: ClassVar[str]
    WIRE_VERSION: ClassVar[int] = 1

    @abstractmethod
    def run(self, runtime: Runtime, *args: Any, **kwargs: Any) -> ProcessorRun[InputT, OutputT, ReturnT]: ...

    def meta(self) -> dict[str, Any]:
        """Model and configuration metadata shared across episodes.

        The harness reads one entry: ``ee_frames``, nested and keyed by pose signal, states the frame of each pose
        signal the policy reads. A run that learns its frames only as it starts, such as a remote run, writes the
        same entry with ``runtime.set_metadata`` before its first yield. A policy that states entries observes only
        those pose signals. A policy that states none observes each pose at the rig's legacy point
        (``Harness._pose_handshake``).
        """
        return {}

    def to_spec(self) -> dict[str, Any]:
        """A versioned component declaration with plain-data arguments."""
        raise NotImplementedError(f'{type(self).__name__} has no wire spec')

    @classmethod
    def from_spec_args(cls, **args: Any) -> Self:
        """Build a component from its wire-spec arguments."""
        return cls(**args)


Policy = Processor[Obs, Step, PolicyEnd]
PolicyRun = ProcessorRun[Obs, Step, PolicyEnd]
