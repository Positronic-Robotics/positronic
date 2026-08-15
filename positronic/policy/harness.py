import concurrent.futures
import logging
import time
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias

import numpy as np
from opentelemetry.trace import Span

import pimm
from positronic import keys, telemetry, telemetry_keys
from positronic.dataset.ds_writer_agent import DsWriterCommand
from positronic.dataset.serializers import expand_suffixed
from positronic.drivers import roboarm
from positronic.drivers.roboarm.ik import assert_default_frame
from positronic.eval import Embodiment, Observation, Task
from positronic.policy.base import Policy, Session
from positronic.utils import flatten_dict, frozen_view

# How far from now an action may be scheduled. A chunk spans seconds, so this is loose enough that no real
# trajectory approaches it, and tight enough to catch a rig-side stack that left timestamps relative to the
# chunk (decades behind) or anchored them twice (decades ahead).
MAX_ACTION_SKEW_SEC = 60.0

# How long a real-time round may last when no waypoint is due sooner. It bounds how late a directive is
# noticed, and with it the granularity every command timestamp is quantized to.
POLL_PERIOD_SEC = 0.01

# How long a submitted session call may take to answer and still resolve within its round. A wrapper that
# skips inference answers in microseconds; a real model call runs far past this and is then paced by
# ``_take`` across rounds.
SKIP_REPLY_SEC = 0.001

# One channel's schedule: waypoints stamped with absolute clock ns, ascending.
Trajectory: TypeAlias = list[tuple[int, Any]]


def _last(due: Sequence[Any]) -> Any:
    """The trailing value wins — the right collapse for absolute setpoints and gripper targets."""
    return due[-1]


class TrajectoryPlayer:
    """Plays one channel's schedule: ``set()`` a trajectory, then ``advance(now)`` each round for the value
    to emit.

    ``reduce`` collapses the waypoints that came due together in one round. Keeping the last is right for a
    value that states where to be; a channel carrying deltas passes ``roboarm.command.reduce`` so their
    motion is summed rather than dropped.
    """

    def __init__(self, reduce: Callable[[Sequence[Any]], Any] = _last):
        self._pending: deque[tuple[int, Any]] = deque()
        self._reduce = reduce

    def set(self, trajectory: Trajectory):
        self._pending = deque(trajectory)

    def next_due(self) -> int | None:
        """Timestamp of the earliest waypoint not yet played, or ``None`` once the schedule is exhausted."""
        return self._pending[0][0] if self._pending else None

    def advance(self, current_time: int):
        """The single value due at ``current_time``, collapsed by ``reduce`` when several came due since the
        previous call, or ``None`` when none did."""
        due = []
        while self._pending and self._pending[0][0] <= current_time:
            due.append(self._pending.popleft()[1])
        return self._reduce(due) if due else None


def _assert_anchored(actions: list[dict[str, Any]], now: float) -> None:
    """Reject a chunk whose timestamps are not times on the harness clock."""
    skew = max((abs(action[keys.ACTION_TIMESTAMP] - now) for action in actions), default=0.0)
    if skew > MAX_ACTION_SKEW_SEC:
        raise ValueError(
            f'Action scheduled {skew:.0f}s from now, over the {MAX_ACTION_SKEW_SEC:.0f}s bound: the rig-side '
            f'stack is not anchoring chunks to the harness clock'
        )


class DirectiveType(Enum):
    RUN = 'run'
    FINISH = 'finish'
    ABORT = 'abort'


@dataclass
class Directive:
    """Directive from the orchestrator to the harness."""

    type: DirectiveType
    payload: Any | None = None

    @classmethod
    def RUN(cls, **kwargs) -> 'Directive':
        """Begin running the policy with the given context."""
        return cls(DirectiveType.RUN, kwargs)

    @classmethod
    def FINISH(cls, **kwargs) -> 'Directive':
        """Finalize the recording with optional eval data, then home devices."""
        return cls(DirectiveType.FINISH, kwargs)

    @classmethod
    def ABORT(cls) -> 'Directive':
        """Discard the live recording and home the devices."""
        return cls(DirectiveType.ABORT)


class _EpisodeTelemetry:
    """The live rollout's wall-clock telemetry: the episode span, its index, its step count and the virtual
    instant it began. Inert while telemetry is unbound, so the harness calls it unconditionally.

    The span stays anchored while open, so the rollout's phase spans (reset, env.step, policy.infer,
    record.io) parent to it rather than to the pass.
    """

    def __init__(self) -> None:
        self._span: Span | None = None
        self._index = -1
        self._steps = 0
        # ``None`` until the rollout starts, so the reset is excluded and a failed reset leaves it unstamped.
        self._virtual_start: float | None = None

    def begin(self, context: dict[str, Any]) -> None:
        """Open the episode span, stamped with the index and the flat trial-context keys. Called before the
        scene reset, so the reset is timed inside the rollout it belongs to."""
        self._index += 1
        self._steps = 0
        self._virtual_start = None
        attrs: dict[str, Any] = {telemetry_keys.ATTR_EPISODE_INDEX: self._index}
        attrs.update({k: v for k, v in context.items() if isinstance(v, (bool, int, float, str))})
        self._span = telemetry.start_span(telemetry_keys.SPAN_EPISODE, **attrs)
        telemetry.push_anchor(self._span)

    def start_rollout(self, virtual_now: float) -> None:
        """Anchor the rollout's virtual duration at the instant its first observation landed."""
        self._virtual_start = virtual_now

    def step(self) -> None:
        self._steps += 1

    def end(self, virtual_now: float) -> None:
        """Close a finished rollout, stamped with its step count and its virtual duration up to
        ``virtual_now`` — captured when the rollout ended, before the flush round advances the sim clock."""
        if self._span is None:
            return
        self._close(virtual_now)
        telemetry.force_flush()

    def abort(self) -> None:
        """Close an aborted rollout, marked ``episode.aborted`` so the reduce drops it."""
        if self._span is None:
            return
        telemetry.set_attrs(self._span, **{telemetry_keys.ATTR_EPISODE_ABORTED: True})
        self._end_span()

    def seal(self, virtual_now: float) -> None:
        """Close a rollout abandoned mid-flight by a raising ``reset`` / ``new_session`` / session call, stamped
        like a clean end and marked ``episode.partial``. Ending it is what exports it — the batch processor drops
        an unended span, orphaning the finished children and losing their phases. Partial rather than aborted so
        the reduce keeps it. Inert when no span is open."""
        if self._span is None:
            return
        telemetry.set_attrs(self._span, **{telemetry_keys.ATTR_EPISODE_PARTIAL: True})
        self._close(virtual_now)
        telemetry.force_flush()

    def _close(self, virtual_now: float) -> None:
        # A rollout whose first observation never landed — a reset that raised, or a task already done before
        # it arrived — has zero virtual duration.
        virtual_s = max(virtual_now - self._virtual_start, 0.0) if self._virtual_start is not None else 0.0
        assert self._span is not None
        attrs = {telemetry_keys.ATTR_EPISODE_STEPS: self._steps, telemetry_keys.ATTR_EPISODE_VIRTUAL_S: virtual_s}
        telemetry.set_attrs(self._span, **attrs)
        self._end_span()

    def _end_span(self) -> None:
        assert self._span is not None
        self._span.end()
        telemetry.pop_anchor(self._span)
        self._span = None


class Harness(pimm.ControlSystem):
    """Control system that runs the episode lifecycle and plays the policy's trajectory to the drivers.

    Handles directives (RUN/FINISH/ABORT) and dataset recording. Inference intelligence — scheduling,
    error recovery, blending, absolute time stamping — lives in the policy/session layer: the wrapper owns
    the plan, the harness plays it, one command per channel per round. The session call runs on a worker
    thread so playing continues while the model does; the harness withholds the trajectory, and the world
    clock, until the trial's inference charge (``inference_latency``) is paid, and the ``now`` it hands
    ``new_session`` reads time the same way — so wrappers stamp chunks for the paid instant without
    knowing the mode. The RUN context is handed whole to the task's scene reset, which reads the
    per-trial keys it needs (e.g. ``eval.seed``).

    A ``trials`` plan (a sequence of RUN contexts) makes the harness self-driving: it starts the next trial
    whenever idle and returns once the plan is exhausted, so the unattended path needs no driver. A task's
    ``timeout`` bounds every trial, self-driven or operator-driven, so an attended episode still terminates
    at the deadline if the operator never sends FINISH. A bounded trial also ends early on a truthy
    privileged ``done``, recording ``eval.terminated`` True and the delivered payload in its static data; a
    timeout records False. A task-less session has neither deadline nor budget and ends only on directives.

    The ``Embodiment`` supplies the observation serializers (which own the canonical key names), the command
    channels and the home action. The policy owns its wrapper stack; the harness runs what it is given.
    """

    def __init__(
        self,
        policy: Policy,
        embodiment: Embodiment,
        *,
        task: Task | None = None,
        trials: Iterable[dict[str, Any]] | None = None,
        static_meta: dict[str, Any] | None = None,
    ):
        assert trials is None or task is not None, 'A trial plan needs a task: its timeout bounds each trial'
        self._embodiment = embodiment
        self._task = task
        # Each entry is a RUN context; when None, directives are the only lifecycle source.
        self._trials = iter(trials) if trials is not None else None
        self.policy: Policy = policy
        self.context: dict[str, Any] = {}
        self._static_meta = static_meta or {}
        self._policy_session: Session | None = None
        # True between RUN and FINISH/ABORT: the trial is live — stepping and recording happen together.
        self._running = False
        # One session call at a time, on a worker so the harness keeps playing while the model runs. The
        # worker belongs to the episode: ending one abandons the call in flight rather than waiting for it,
        # so the next episode must not queue behind it.
        self._executor: ThreadPoolExecutor | None = None
        self._retiring: ThreadPoolExecutor | None = None
        self._future: Future[list[dict[str, Any]] | None] | None = None
        # The in-flight call's start: the world instant its observation was built, and the wall instant it
        # was submitted.
        self._t0_ns = 0
        self._wall_t0 = 0.0
        # Seconds each model call costs the world clock this episode, or ``None`` to charge the call's own
        # wall duration (hardware pace, and the sim's ``inference_latency=True``).
        self._charge: float | None = None
        # ``task.timeout``, set per episode; a task-less session has no deadline and ends on directives.
        self._deadline: float | None = None
        # Whether this episode's first observation has landed. Until it does the deadline stands where the
        # reset put it, which bounds an episode whose first observation never arrives.
        self._rollout_started = False
        # Wall-clock telemetry for the live rollout, opened under ``--timing`` and inert otherwise.
        self._telemetry = _EpisodeTelemetry()
        # Channels that have not delivered since this episode's reset. A receiver latches its last value, so
        # emptying this set is what keeps the first inference off the previous episode's final frame.
        self._awaiting_obs: set[str] = set()

        self._descriptor = embodiment.descriptor
        self.observations = pimm.ReceiverDict(self)
        self.commands = pimm.EmitterDict(self)
        for name in embodiment.observations:
            self.observations[name]  # touch to allocate the port
        for name in embodiment.commands:
            self.commands[name]
        self._players = {
            name: TrajectoryPlayer(roboarm.command.reduce if keys.is_robot_command(name) else _last)
            for name in embodiment.commands
        }

        self.directive = pimm.ControlSystemReceiver[Directive](self, default=None, maxsize=3)
        self.manual_command = pimm.ControlSystemReceiver(self, default=None)
        self.ds_command = pimm.ControlSystemEmitter[DsWriterCommand](self)
        self.robot_meta_in = pimm.ControlSystemReceiver(self, default={})
        # Privileged stop-signal: a truthy value within the trial's budget ends it.
        self.done = pimm.ControlSystemReceiver[dict](self, default={})

    def _statics(self) -> dict[str, Any]:
        """What is known about the rig before the episode runs, live values winning."""
        return self._embodiment.static_meta | self._static_meta | self.robot_meta_in.value

    def _build_episode_meta(self, context: dict[str, Any]) -> dict[str, Any]:
        meta = self._statics()
        if self._task is not None:
            # TODO: also stamp the eval's catalog name and its resolved config — both need configuronic
            # introspection that does not exist yet.
            meta['eval.universe'] = 'sim' if self._embodiment.simulated else 'real'
            meta['eval.embodiment'] = self._embodiment.descriptor
            meta['eval.timeout'] = self._task.timeout
        # ``policy.meta`` is the static baseline; the session overlays per-episode specifics (e.g. the
        # sampled sub-policy) and wins on conflict.
        session_meta = self.policy.meta | (self._policy_session.meta if self._policy_session else {})
        for k, v in flatten_dict(session_meta).items():
            meta[f'{keys.POLICY_META}.{k}'] = v
        meta.update(context)
        return meta

    def _emit(self, action: dict[str, Any]) -> None:
        for name, value in action.items():
            self.commands[name].emit(value)

    def _home(self) -> None:
        self._emit(self._embodiment.home)

    def _pace(self, clock: pimm.Clock) -> pimm.Command:
        """Sim: yield, so the simulator's control-period sleep is the sole time-master and the policy reads
        each observation instantly. Real: sleep to the next waypoint, capped at the poll period, so a
        waypoint is emitted at its own time and a round rarely finds more than one due."""
        if self._embodiment.simulated:
            return pimm.Yield()
        due = [ts for player in self._players.values() if (ts := player.next_due()) is not None]
        if not due:
            return pimm.Sleep(POLL_PERIOD_SEC)
        return pimm.Sleep(min(POLL_PERIOD_SEC, max(min(due) - clock.now_ns(), 1) / 1e9))

    def _cancel_session(self) -> None:
        """Drop everything the episode has going: the schedule being played, and the call on the worker.

        The call is let go of rather than waited for — a model that hangs must not hold up the recording's
        stop or the home — so its worker is retired with it and whatever it eventually answers, or raises,
        lands nowhere. Devices hold their last commanded position; nothing is buffered downstream to clear.
        """
        for player in self._players.values():
            player.set([])
        self._retire_worker()

    @staticmethod
    def _report_abandoned(future: Future[list[dict[str, Any]] | None]) -> None:
        """Report the failure of a call nobody is waiting for any more."""
        # rules-allow: swallowed-error — the call outlived the episode that asked for it, so there is no
        # caller left to raise to; the log is the only place its failure can go.
        if not future.cancelled() and (exc := future.exception()) is not None:
            logging.error(f'Inference failed after the episode that asked for it ended: {exc}')

    def _retire_worker(self) -> None:
        """Let go of this episode's worker and the call it is running: the answer lands nowhere and the
        failure only reaches the log. The worker is kept for ``_reap_worker`` to join at the next episode's
        start, since ending an episode must not wait for a model that hangs."""
        if self._future is not None:
            self._future.add_done_callback(self._report_abandoned)
            self._future = None
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._retiring, self._executor = self._executor, None

    def _reap_worker(self) -> None:
        """Wait out the previous episode's abandoned call before this one opens a session.

        An in-process policy is a single model across episodes, so ``new_session`` resets the very object an
        abandoned call may still be inside — a running thread survives ``shutdown(cancel_futures=True)``,
        which cancels only what is still queued. Waiting here rather than at the end of the episode it
        belongs to keeps a hung model from holding up that episode's recording and home, and the task reset
        this follows usually covers the wait.
        """
        if self._retiring is not None:
            self._retiring.shutdown(wait=True)
            self._retiring = None

    def _finalize_recording(
        self, clock: pimm.Clock, payload: dict[str, Any] | None = None
    ) -> Generator[pimm.Command, None, None]:
        """Commit the live episode: cancel the in-flight chunk, stop the recorder — stamping the
        episode's full static meta (plus any terminal payload) — then close its span."""
        self._cancel_session()
        self.ds_command.emit(DsWriterCommand.STOP({**self._build_episode_meta(self.context), **(payload or {})}))
        virtual_now = clock.now()  # before the round below, whose sim-clock advance belongs to no rollout
        # Give the recorder a round to commit the STOP before the next START (they share ``ds_command``, where
        # last-value-wins would drop one) and before the home command, so homing stays out of the recording.
        yield self._pace(clock)
        # After that round, so the recorder's STOP-time record.io span is still in flight and parents to the
        # episode. Accepted skew: a producer stepping in that shared round charges one span (≤ one control
        # period) to the closing episode — the cooperative scheduler cannot give the recorder a turn alone.
        self._telemetry.end(virtual_now)

    def _begin_episode(self, context: dict[str, Any], clock: pimm.Clock) -> None:
        """Open a fresh episode: reset the scene, fix the task context and session, and open the recording.

        A resettable task's ``reset`` only arms the producer, which publishes the first observation on a later
        round. The recorder drains its channels the turn it opens, so the pre-reset frame and the
        inter-episode home command drop out and its first sample is the post-reset scene. The deadline is
        armed here and moved to that first observation once it lands, so an episode that never gets one is
        still bounded.
        """
        self.context = dict(context)
        if self._embodiment.simulated:
            # A sim trial that doesn't ask for latency simulation runs free of it: the world holds still for
            # every model call.
            latency = self.context.setdefault(keys.INFERENCE_LATENCY, False)
        else:
            # The charge is a device for simulating a trial, so a real rig ignores it and pays the wall time
            # its calls really take.
            latency = True
        self._charge = None if latency is True else float(latency)
        self._awaiting_obs = set(self._embodiment.observations)
        self._rollout_started = False
        # Before the reset, so the reset and the rollout's other phase spans parent to the episode span.
        self._telemetry.begin(context)
        # Reset before opening the session: a resettable task only learns its instruction on reset (a remote
        # env reports it then), so the session context — and the sampling it drives — must read it here.
        if self._task is not None and self._task.reset is not None:
            with telemetry.span(telemetry_keys.SPAN_RESET):
                self._task.reset(self.context)
        if self._task is not None:
            self.context = {**self.context, keys.TASK: self._task.instruction}
        self._reap_worker()
        # Arm the clock before handing it out: a session reading it before its first call must see this
        # episode's start, not the release time of the last episode's final call.
        self._t0_ns = clock.now_ns()
        self._wall_t0 = time.monotonic()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='harness-session')
        self._policy_session = self.policy.new_session(self.context, self._effect_time)
        self._running = True
        self._deadline = clock.now() + self._task.timeout if self._task is not None else None
        self.ds_command.emit(DsWriterCommand.START())

    def _end_episode(
        self, clock: pimm.Clock, payload: dict[str, Any] | None = None, *, abort: bool = False
    ) -> Generator[pimm.Command, None, None]:
        """Close the live episode: finalize (or abort) the recording, release the session, home devices.

        Releasing the session here, not only at shutdown, closes a ``RemoteSession``'s websocket promptly, so
        the offboard server's per-session cleanup (active-session decrement, idle watchdog) runs now.
        """
        if self._running:
            if abort:
                self._cancel_session()  # abort has no finalize to do it — stop the episode before the home
                self.ds_command.emit(DsWriterCommand.ABORT())
                yield self._pace(clock)  # the settling round a finalize also takes, before the home command
                self._telemetry.abort()
            else:
                yield from self._finalize_recording(clock, payload)
        if self._policy_session:
            self._policy_session.close()
            self._policy_session = None
        self._home()
        self._running = False

    def _handle_directive(self, directive: Directive, clock: pimm.Clock) -> Generator[pimm.Command, None, None]:
        """Dispatch a directive to the episode lifecycle; updates ``_running``."""
        match directive.type:
            case DirectiveType.RUN:
                if not self._running:  # a RUN mid-trial is ignored — the operator finishes before starting anew
                    self._begin_episode(directive.payload or {}, clock)
            case DirectiveType.FINISH:
                if self._running:  # a FINISH while idle is ignored — nothing to finalize
                    yield from self._end_episode(clock, directive.payload)
            case DirectiveType.ABORT:
                yield from self._end_episode(clock, abort=True)
            case _:
                raise ValueError(f'Unknown directive type: {directive.type}')

    @staticmethod
    def _is_faulted(value: Any) -> bool:
        """Whether a raw observation is an arm reporting a fault. Every other not-ready sample is simply absent."""
        return isinstance(value, roboarm.State) and value.status is roboarm.RobotStatus.ERROR

    def _read_channel(self, name: str, obs: Observation) -> tuple[dict[str, Any] | None, bool]:
        """This channel's entries under their full names, and whether the arm behind it is faulted.

        The entries are ``None`` when the channel has no sample to give — a resetting or faulted arm alike.
        Raises ``NoValueException`` before the channel has produced anything at all.
        """
        message = self.observations[name].read()
        if message is None:
            raise pimm.NoValueException
        if message.updated:
            self._awaiting_obs.discard(name)
        value = message.data
        if obs.serializer is not None:
            value = obs.serializer(value)
            if value is None:
                # HACK(#619): a serializer answers `None` for a resetting arm and a faulted one alike, so the
                # fault is recovered from the raw sample and stapled on by the caller as `keys.ROBOT_FAULT` — a
                # name already claiming to be part of `robot_state`. Emit it from the serializer and this
                # branch, the raw-type check and the flag all go, and the fault reaches the recording as well.
                return None, self._is_faulted(message.data)
        return {full: v for full, v in expand_suffixed(name, value) if v is not None}, False

    def _build_obs(self, clock: pimm.Clock) -> dict[str, Any] | None:
        """Read every observation channel and assemble the policy input dict.

        Raises ``NoValueException`` if any channel has no value yet. Returns ``None`` while a serializer
        reports a sample is not ready (``robot_state`` during a ``RESETTING`` arm) or a channel still holds a
        pre-reset value — either way the harness skips inference rather than feed a partial or stale obs.

        A faulted arm is the exception: it has no sample either, but the plan being played was made for an
        arm that is now somewhere else, so the observation goes to the policy stack carrying
        ``keys.ROBOT_FAULT`` and without the arm's own entries.
        """
        # Against the live model, not the one known at episode start: a remote env publishes its ``robot_meta``
        # a turn after the reset that produced it, so at episode start there is no model to check.
        assert_default_frame(self._statics())
        inputs: dict[str, Any] = {}
        faulted = False
        not_ready = False
        for name, obs in self._embodiment.observations.items():
            entries, channel_faulted = self._read_channel(name, obs)
            faulted = faulted or channel_faulted
            not_ready = not_ready or entries is None
            inputs.update(entries or {})
        # Every channel is read before this decision, so a bimanual rig cannot hide one arm's fault behind
        # another arm's not-ready sample: whichever channel comes first, the fault still reaches the stack.
        if not_ready and not faulted:
            return None
        if self._awaiting_obs:
            return None
        inputs[keys.ROBOT_FAULT] = faulted
        inputs[keys.WALL_TIME_NS] = time.time_ns()
        inputs[keys.OBS_TIME_NS] = clock.now_ns()
        inputs.update(self.context)
        inputs['descriptor'] = self._descriptor  # last, so a context key can't shadow it
        return inputs

    def _effect_time(self) -> float:
        """The trial instant the in-flight call's output takes effect: its observation instant plus the
        charge — the declared constant whole, or the wall time elapsed so far. Read on the worker thread;
        the loop thread writes the call's start fields before submitting it.
        """
        charge = time.monotonic() - self._wall_t0 if self._charge is None else self._charge
        return self._t0_ns / 1e9 + charge

    @staticmethod
    def _owned(obs: dict[str, Any]) -> dict[str, Any]:
        """The observation with its arrays copied, so nothing rewrites what the worker is still reading.

        A producer may reuse one buffer for every sample it emits — a camera renders into the array behind
        the adapter it re-emits each frame — and the loop thread yields while a call charged in wall time
        runs, so that producer advances alongside the worker. Copying at dispatch pays once per call rather
        than per round.
        """
        return {name: value.copy() if isinstance(value, np.ndarray) else value for name, value in obs.items()}

    class _Answer(Enum):
        """What became of the call ``_take`` was handed: its trajectory is installed and the future spent, or
        the world has not yet paid the charge and the same future comes back next round."""

        CONSUMED = 'consumed'
        PENDING = 'pending'

    def _step(self, clock: pimm.Clock) -> None:
        """Keep one session call in flight and install the trajectory it returns.

        The call goes to the worker so the harness keeps playing while the model runs; a wrapper that
        answers without inference still resolves in the round it was asked.
        """
        session, executor = self._policy_session, self._executor
        assert session is not None and executor is not None  # only a live episode steps
        if self._future is not None and self._take(self._future, clock) is Harness._Answer.PENDING:
            return
        obs = self._build_obs(clock)
        if obs is None:
            return
        if not self._rollout_started:
            # The rollout begins at its first observation, not when the reset returned: a reset only asks
            # the producer for a scene, and the turns spent delivering it are neither the trial's budget
            # nor its duration.
            self._rollout_started = True
            self._telemetry.start_rollout(clock.now())
            if self._task is not None:
                self._deadline = clock.now() + self._task.timeout
        self._t0_ns = clock.now_ns()
        self._wall_t0 = time.monotonic()
        self._future = executor.submit(session, frozen_view(self._owned(obs)))
        if self._charge is None:
            # Sleeping zero hands the worker the GIL without adding a wake-up granularity to the handshake.
            while not self._future.done() and time.monotonic() - self._wall_t0 < SKIP_REPLY_SEC:
                time.sleep(0)
        self._take(self._future, clock)

    def _take(self, future: Future[list[dict[str, Any]] | None], clock: pimm.Clock) -> _Answer:
        """Install the call's trajectory once the world has paid for it.

        Under a constant charge the world holds still until the call answers — blocking here blocks the loop
        thread, which is what advances a virtual clock. Until a call answers there is no telling a skip from
        a model call, so letting the world run meanwhile would spend trial time on whichever the machine
        turned out to be slow at. What the charge then buys is the instant a trajectory takes effect: it is
        stamped for ``t0`` plus the charge and withheld until the world reaches it, playing what is already
        scheduled on the way. An answer with no waypoints to place — a skip, or the empty trajectory that
        stops what is executing — has no such instant and lands at once. A charge measured in wall time can
        hold nothing still, so there the world runs no further ahead of the call's start than wall time has.
        """
        if self._charge is not None:
            concurrent.futures.wait([future])
        elif not future.done():
            ahead = clock.now() - (self._t0_ns / 1e9 + time.monotonic() - self._wall_t0)
            if ahead <= 0.0:
                return Harness._Answer.PENDING
            concurrent.futures.wait([future], timeout=ahead)
            if not future.done():
                return Harness._Answer.PENDING
        actions = future.result()  # taken on the loop thread, so a failing call still seals the episode
        if actions and self._charge is not None:
            # Integer ns, the world's own timeline: a float compare misses the release instant by one ULP
            # and slips the install a full round.
            if clock.now_ns() < self._t0_ns + round(self._charge * 1e9):
                return Harness._Answer.PENDING  # the schedule already playing carries the world to the release instant
        self._future = None
        if actions is not None:
            self._install(actions, clock)
        return Harness._Answer.CONSUMED

    def _install(self, actions: list[dict[str, Any]], clock: pimm.Clock) -> None:
        """Replace the schedule being played with the session's trajectory. Every channel it names gets that
        channel's waypoints; one it omits is cleared and holds. The timestamps are already absolute, stamped
        by the scheduling wrapper for the instant its charge is paid.
        """
        _assert_anchored(actions, clock.now())
        self._telemetry.step()
        for name, player in self._players.items():
            # Wrappers do action-timing math in float seconds; the schedule and every pimm channel are in ns.
            # This is the single explicit seconds->ns seam.
            player.set([(int(a[keys.ACTION_TIMESTAMP] * 1e9), a[name]) for a in actions if name in a])

    def _play(self, clock: pimm.Clock) -> None:
        """Emit each channel's command due this round, and nothing on a channel with none."""
        now_ns = clock.now_ns()
        for name, player in self._players.items():
            value = player.advance(now_ns)
            if value is not None:
                self.commands[name].emit(value)

    def _trial_terminal(self, clock: pimm.Clock) -> dict[str, Any] | None:
        """The terminal static payload if a self-driven trial has ended this round, else ``None``.

        The deadline is hard: a truthy ``done`` delivered within budget records ``eval.terminated`` True plus
        its payload, the budget passing records False, and a terminal past the deadline is a timeout rather
        than a late success. Only a freshly delivered ``done`` counts — the receiver latches its last value,
        so a prior trial's terminal would otherwise re-fire; gating on delivery clears it without asking the
        producer to republish. Reached only for a task with a deadline.
        """
        done_msg = self.done.read()
        if done_msg.updated and done_msg.data and done_msg.ts <= self._deadline * 1e9:
            return {**done_msg.data, keys.EVAL_TERMINATED: True}
        if clock.now() >= self._deadline:
            return {keys.EVAL_TERMINATED: False}
        return None

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        # Home the embodiment before the first episode; each ``_end_episode`` re-homes for the next one, so
        # every episode begins from the home pose (a real arm gets the inter-episode gap to reach it).
        self._home()

        try:
            yield from self._run(should_stop, clock)
        except BaseException:
            # A failure mid-rollout unwinds past the normal span close. Seal the open span before the
            # exception reaches ``bind``'s exit flush, or it never exports and its finished children orphan,
            # losing their phases and charging the episode's wall to between_episodes.
            self._telemetry.seal(clock.now())
            raise
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        """Release the worker and the session. A call still in flight runs to completion and its result is
        dropped: the run is over and nothing is left to install it.

        The harness does not own the policy's lifetime: the caller may run several harnesses over one policy
        (a multi-eval sweep), so it closes the policy once, after the last run.
        """
        self._retire_worker()
        if self._policy_session is not None:
            self._policy_session.close()
            self._policy_session = None

    def _run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        while not should_stop.value:
            # One action per round, mutually exclusive: handle a directive, start the next trial (or exit
            # when the plan is done), finish a self-driven trial that is out of budget or done, or step the
            # policy. Starting takes its own round so a begin never shares one with a step — inference waits
            # for the producer's post-reset observation, which the recorder logs once its open-turn drain has
            # cleared the channels.
            directive_msg = self.directive.read()
            # Read every round so the flag clears mid-episode; a press during a trial is consumed, not replayed.
            manual_msg = self.manual_command.read()
            # Both receivers carry a default, so ``read`` always yields a message.
            assert directive_msg is not None and manual_msg is not None
            if directive_msg.updated:
                yield from self._handle_directive(directive_msg.data, clock)
            elif not self._running:
                if manual_msg.updated and manual_msg.data is not None:
                    self._emit(manual_msg.data)
                elif self._trials is not None:
                    trial = next(self._trials, None)
                    if trial is None:  # plan exhausted — let the recorder commit the final episode, then exit
                        yield pimm.Sleep(0.5)
                        break
                    self._begin_episode(trial, clock)
            elif self._deadline is not None and (terminal := self._trial_terminal(clock)) is not None:
                yield from self._end_episode(clock, terminal)
            else:
                try:
                    self._step(clock)
                except pimm.NoValueException:
                    pass
            self._play(clock)
            yield self._pace(clock)

        if self._running:
            yield from self._finalize_recording(clock)
