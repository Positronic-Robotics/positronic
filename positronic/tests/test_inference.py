import io
import logging
import multiprocessing
import os
import signal
import socket
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Any

import av
import httpx
import numpy as np
import pos3
import pytest
from websockets.sync.client import connect

import pimm
from pimm.shared_memory import NumpySMAdapter
from pimm.time import EMITTED_WALL, EMITTED_WORLD, RECEIVED_WALL, RECEIVED_WORLD
from positronic import keys, wire
from positronic.dataset.local_dataset import LocalDataset
from positronic.dataset.serializers import Serializers
from positronic.dataset.video import VideoSignal
from positronic.drivers import keyboard
from positronic.eval import Embodiment, Observation, Task
from positronic.eval import keys as eval_keys
from positronic.gui.station import Episode, Outcome, Phase, Verdict
from positronic.gui.web import EndBody, EndTrial, InstructionBody, StationConsole, Status
from positronic.inference import KeyboardOperator, TrialForwarder, real, web
from positronic.policy import Policy
from positronic.policy.base import Step
from positronic.policy.harness import Harness, Rollout
from positronic.tests.testing_coutils import drive_scheduler, scripted_driver


class _IdlePolicy(Policy):
    """Enough policy for the attended path to run an episode and close; it commands nothing."""

    def __init__(self):
        self.started = False
        self.closed = False
        self.observations: list[dict] = []

    def run(self, runtime):
        self.started = True
        try:
            obs = yield
            while True:
                self.observations.append(obs)
                obs = yield Step({}, runtime.time_ns + 100_000_000)
        finally:
            self.closed = True


ARM_STOPPED_SHORT = 'the arm stopped short of its target'


class _ReadyDevices(pimm.ControlSystem):
    """The arm and fingers of a rig that reaches wherever it is asked to go ``move_s`` after it is asked. The arm's
    first ``arm_failures`` moves stop short. Its teardown creates ``parked``, when one is given."""

    def __init__(self, move_s: float = 0.0, arm_failures: int = 0, parked: Path | None = None):
        self._move_s = move_s
        self._arm_failures = arm_failures
        self._parked = parked
        self.arm = pimm.calls.ControlSystemHandler[Any, None](self)
        self.gripper = pimm.calls.ControlSystemHandler[Any, None](self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        moving: list[tuple[pimm.calls.Call[Any, None], float]] = []
        while not should_stop.value:
            now = clock.now()
            for call in self.arm.incoming():
                if self._arm_failures > 0:
                    self._arm_failures -= 1
                    call.set_exception(RuntimeError(ARM_STOPPED_SHORT))
                else:
                    moving.append((call, now + self._move_s))
            moving.extend((call, now + self._move_s) for call in self.gripper.incoming())
            for call, _ in [(call, at) for call, at in moving if at <= now]:
                call.set_result(None)
            moving = [(call, at) for call, at in moving if at > now]
            yield pimm.Sleep(0.01)
        if self._parked is not None:
            self._parked.touch()


def _embodiment(simulated: bool = False, move_s: float = 0.0, arm_failures: int = 0) -> Embodiment:
    devices = _ReadyDevices(move_s, arm_failures)
    return Embodiment(
        descriptor='stub',
        observations={},
        commands={},
        prepare_handlers={eval_keys.ARM: devices.arm, eval_keys.GRIPPER: devices.gripper},
        static_meta={},
        meta_source=None,
        control_systems=(devices,),
        simulated=simulated,
    )


def _trial(instruction: str = 'stub') -> Callable[[], Task]:
    prepare_args = {eval_keys.ARM: 'start-pose', eval_keys.GRIPPER: 0.0}
    return partial(Task, instruction_source=instruction, timeout_sec=None, prepare_args=prepare_args)


@pytest.mark.timeout(30.0)
def test_the_keyboard_path_ends_when_the_keyboard_returns(monkeypatch):
    """How an attended run finishes: the operator returns, the world stops, no policy run is started.

    A stdin that is not a terminal is the return the test can force; ``q`` is the other one.
    """
    monkeypatch.setattr(sys, 'stdin', io.StringIO())
    policy = _IdlePolicy()

    real(policy=policy, embodiment=_embodiment(), next_task=_trial())

    assert not policy.closed
    assert not policy.started


def test_the_keyboard_path_refuses_a_simulated_embodiment():
    """It composes a real-time world and records against the wall clock, which a simulated embodiment needs
    neither of."""
    with pytest.raises(ValueError, match='sim'):
        real(policy=_IdlePolicy(), embodiment=_embodiment(simulated=True), next_task=_trial())


@pytest.mark.timeout(30.0)
def test_a_keypress_opens_an_episode_and_another_ends_it(monkeypatch, caplog):
    """The press is the whole start signal: the rig's devices ready, the episode opens on the instruction it
    was given, and it runs until the operator stops it."""
    policy = _IdlePolicy()

    def presses():
        yield 's'
        while not policy.observations:
            yield None
        yield 'p'
        while 'Episode ended:' not in caplog.text:
            yield None
        yield 'q'

    script = presses()
    monkeypatch.setattr(keyboard, 'key_reader', partial(nullcontext, lambda: next(script, None)))

    with caplog.at_level(logging.INFO):
        real(policy=policy, embodiment=_embodiment(), next_task=_trial('pick up the cube'))

    assert policy.observations, 'the episode never opened'
    assert policy.observations[0][keys.TASK] == 'pick up the cube'
    assert eval_keys.ENDED_BY_OPERATOR in caplog.text
    assert policy.closed


def test_a_task_failure_is_reported_without_stopping_the_operator(monkeypatch, caplog):
    presses = iter(['s'])
    monkeypatch.setattr(keyboard, 'key_reader', partial(nullcontext, lambda: next(presses, None)))
    operator = KeyboardOperator(lambda: Task(instruction_source='pick', timeout_sec=None), _IdlePolicy(), None)
    with pimm.World(virtual_time=True) as world:
        handler = world.pair(operator.perform_task)

        def refuse():
            for call in handler.incoming():
                call.set_exception(RuntimeError('endpoint down'))

        with caplog.at_level(logging.ERROR):
            drive_scheduler(world.start([operator, scripted_driver((refuse, 0.1), (None, 0.1))]))
    assert 'endpoint down' in caplog.text


def test_the_operator_declines_a_press_while_an_episode_runs(monkeypatch, caplog):
    """One episode at a time is the operator's own rule: the second press never reaches the harness."""
    task = Task(instruction_source='pick', timeout_sec=None)
    presses = iter(['s', 's'])
    monkeypatch.setattr(keyboard, 'key_reader', partial(nullcontext, lambda: next(presses, None)))
    operator = KeyboardOperator(lambda: task, _IdlePolicy(), None)
    with pimm.World(virtual_time=True) as world:
        harness = world.pair(operator.perform_task)
        received = []

        def hold_the_ask():
            """Stand in for a harness running the episode it was asked for: take the call, answer nothing."""
            received.extend(call.request.task for call in harness.incoming())

        driver = scripted_driver((hold_the_ask, 0.05), (hold_the_ask, 0.05))
        drive_scheduler(world.start([operator, driver]))

    assert received == [task]
    assert 'already running' in caplog.text


def test_the_web_console_refuses_a_simulated_embodiment():
    with pytest.raises(ValueError, match='sim'):
        web(policy=_IdlePolicy(), embodiment=_embodiment(simulated=True), next_task=_trial())


class _ForwarderBench:
    """A ``TrialForwarder`` whose trial calls, rollout calls and verdicts a script drives in virtual time."""

    PAYLOAD = {eval_keys.ENDED_BY: eval_keys.ENDED_BY_OPERATOR, eval_keys.SUCCESS: True}

    def __init__(self):
        self.policy = _IdlePolicy()
        self.forwarder = TrialForwarder(partial(Rollout, policy=self.policy, output_path=None))
        self.task = Task(instruction_source='pick', timeout_sec=None)
        self.rollouts: list[Rollout] = []
        self.harness_calls: list[pimm.calls.Call] = []
        self.answers: list[pimm.calls.Answer] = []
        self.done: list[dict] = []

    def run(self, *steps: Callable[[], None]) -> None:
        with pimm.World(virtual_time=True) as world:
            self._trials = world.pair(self.forwarder.trials)
            self._harness = world.pair(self.forwarder.perform_task)
            self._done = world.pair(self.forwarder.done)
            script = [(step, 0.05) for step in (*steps, self.watch, self.watch, self.watch)]
            drive_scheduler(world.start([self.forwarder, scripted_driver(*script)]))

    def watch(self) -> None:
        """Take what reached the harness: the calls it is asked, and each ``done``."""
        for call in self._harness.incoming():
            self.rollouts.append(call.request)
            self.harness_calls.append(call)
        if (message := pimm.read_updated(self._done)) is not None:
            self.done.append(message.data)

    def ask(self) -> None:
        self.answers.append(self._trials(self.task))

    def end(self) -> None:
        self.answers.append(self._trials(EndTrial(self.PAYLOAD)))

    def answer(self, result: dict) -> None:
        for call in self.harness_calls:
            call.set_result(result)
        self.harness_calls.clear()


def test_the_forwarder_runs_each_trial_as_its_rollout_and_returns_the_harness_answer():
    bench = _ForwarderBench()
    bench.run(bench.ask, bench.watch, lambda: bench.answer({eval_keys.TERMINATED: True}))
    assert bench.answers[0].result() == {eval_keys.TERMINATED: True}
    assert bench.rollouts == [Rollout(bench.task, bench.policy, None)]


def test_the_forwarder_hands_a_failed_episode_back_to_its_caller():
    bench = _ForwarderBench()

    def refuse():
        for call in bench.harness_calls:
            call.set_exception(RuntimeError('endpoint down'))

    bench.run(bench.ask, bench.watch, refuse)
    with pytest.raises(RuntimeError, match='endpoint down'):
        bench.answers[0].result()


def test_a_verdict_is_forwarded_once_without_waiting_for_a_start_notification():
    bench = _ForwarderBench()
    bench.run(bench.ask, bench.watch, bench.end, bench.watch, bench.end, bench.watch)
    assert bench.done == [_ForwarderBench.PAYLOAD]


def test_a_verdict_for_an_answered_trial_returns_its_result_without_sending_done():
    bench = _ForwarderBench()
    timeout = {eval_keys.TERMINATED: False}

    def finish_and_end():
        bench.answer(timeout)
        bench.end()

    bench.run(bench.ask, bench.watch, finish_and_end)
    assert bench.done == []
    assert [answer.result() for answer in bench.answers] == [timeout, timeout]


def test_a_verdict_after_its_trial_was_answered_is_refused():
    bench = _ForwarderBench()
    finished = {eval_keys.TERMINATED: True}
    bench.run(bench.ask, bench.watch, lambda: bench.answer(finished), bench.watch, bench.end)
    assert bench.done == []
    with pytest.raises(RuntimeError, match='ended before its verdict'):
        bench.answers[1].result()


_CAMERA_FRAME = 'camera.frame'


class _Camera(pimm.ControlSystem):
    """A camera that sends a new frame 15 times a second."""

    def __init__(self):
        self.frame = pimm.ControlSystemEmitter[NumpySMAdapter](self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        adapter = None
        count = 0
        while not should_stop.value:
            adapter = NumpySMAdapter.lazy_init(np.full((120, 160, 3), (count * 37) % 256, dtype=np.uint8), adapter)
            self.frame.emit(adapter, time=pimm.Time(**{_CAMERA_FRAME: count}))
            count += 1
            yield pimm.Sleep(1 / 15)


class _MarkingPolicy(Policy):
    """Checks array snapshots and marks each episode after five policy steps."""

    def __init__(self, marks: Path):
        self._marks = marks
        self._episodes = 0

    def run(self, runtime):
        self._episodes += 1
        number = self._episodes
        obs = yield
        steps = 0
        while True:
            snapshots = [(value, value.copy()) for value in obs.values() if isinstance(value, np.ndarray)]
            obs = yield Step({}, runtime.time_ns + 100_000_000)
            for value, snapshot in snapshots:
                np.testing.assert_array_equal(value, snapshot)
            steps += 1
            if steps == 5:
                (self._marks / f'episode-{number}').touch()


def _serve_station(port: int, output_dir: Path, marks: Path) -> None:
    camera, devices = _Camera(), _ReadyDevices(parked=marks / 'parked')
    embodiment = Embodiment(
        descriptor='stub',
        observations={keys.WRIST_IMAGE: Observation(camera.frame, Serializers.camera_images)},
        commands={},
        prepare_handlers={eval_keys.ARM: devices.arm, eval_keys.GRIPPER: devices.gripper},
        static_meta={},
        meta_source=None,
        control_systems=(devices, camera),
    )
    with pos3.mirror():
        web(_MarkingPolicy(marks), embodiment, _trial('pick up the cube'), output_dir=str(output_dir), port=port)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def _wait_for(condition: Callable[[], Any], what: str, timeout: float = 90.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if value := condition():
            return value
        time.sleep(0.1)
    raise AssertionError(f'timed out waiting for {what}')


def _status(base: str) -> Status | None:
    try:
        return Status.model_validate(httpx.get(f'{base}/status', timeout=2.0).json())
    except httpx.TransportError:
        return None


def _run_episode(base: str, marks: Path, number: int, verdict: Verdict) -> Episode:
    httpx.post(f'{base}/episode/start').raise_for_status()
    _wait_for((marks / f'episode-{number}').exists, f'episode {number} to open')
    httpx.post(f'{base}/episode/end', json=EndBody(verdict=verdict).model_dump(mode='json')).raise_for_status()

    def closed() -> Status | None:
        status = _status(base)
        return status if status and status.run.phase is Phase.READY else None

    return _wait_for(closed, f'episode {number} to close').run.episodes[number - 1]


@pytest.mark.timeout(240.0)
def test_the_web_console_records_each_episode_with_its_instruction_and_verdict(tmp_path):
    """The whole command, in a process of its own: the page streams the camera, and each episode it starts
    records the instruction it sent, the override flag, its number, and the operator's verdict."""
    port = _free_port()
    base = f'http://127.0.0.1:{port}'
    marks = tmp_path / 'marks'
    marks.mkdir()
    output_dir = tmp_path / 'episodes'
    run = multiprocessing.get_context('spawn').Process(target=_serve_station, args=(port, output_dir, marks))
    run.start()
    try:
        status = _wait_for(lambda: (s := _status(base)) and s.cameras[0].live and s, 'the camera to be live')
        assert status.run.configured == 'pick up the cube'
        with connect(f'ws://127.0.0.1:{port}/video/{keys.WRIST_IMAGE}', open_timeout=10) as tile:
            codec, _init, fragment = (tile.recv(timeout=10) for _ in range(3))
            assert isinstance(codec, str) and codec.startswith('avc1.')
            assert isinstance(fragment, bytes) and fragment[4:8] == b'moof'

        override = InstructionBody(override='pick up the red cube').model_dump()
        httpx.post(f'{base}/instruction', json=override).raise_for_status()
        last_frame = -1
        for number, verdict in enumerate((Outcome.PASS, Outcome.FAIL), start=1):
            assert _run_episode(base, marks, number, verdict).outcome is verdict
            dataset = LocalDataset(output_dir)
            assert len(dataset) == number
            camera = dataset[number - 1][keys.WRIST_IMAGE]
            assert isinstance(camera, VideoSignal)
            assert len(camera) >= 5
            assert set(camera.timelines) == {EMITTED_WALL, EMITTED_WORLD, RECEIVED_WALL, RECEIVED_WORLD, _CAMERA_FRAME}
            times = camera.timestamps(camera.timelines)
            assert times[0][_CAMERA_FRAME] >= last_frame
            with av.open(str(camera.video_path)) as video:
                for frame, timestamp in zip(video.decode(video=0), times, strict=True):
                    pixels = frame.to_ndarray(format='rgb24')
                    assert pixels.shape == (120, 160, 3)
                    np.testing.assert_allclose(pixels, (timestamp[_CAMERA_FRAME] * 37) % 256, atol=3)
                    assert timestamp[EMITTED_WALL] == timestamp[EMITTED_WORLD]
                    assert timestamp[RECEIVED_WALL] == timestamp[RECEIVED_WORLD]
                    assert timestamp[EMITTED_WALL] <= timestamp[RECEIVED_WALL]
            assert all(before < after for before, after in pairwise(times))
            assert np.all(np.diff(camera.timestamps(_CAMERA_FRAME)) > 0)
            last_frame = times[-1][_CAMERA_FRAME]
    finally:
        if run.pid is not None and run.is_alive():
            os.kill(run.pid, signal.SIGINT)
        run.join(timeout=120)
        if run.is_alive():
            run.kill()

    first, second = (episode.static for episode in LocalDataset(output_dir))
    assert first[keys.TASK] == 'pick up the red cube'
    assert first[eval_keys.INSTRUCTION_OVERRIDDEN] is True
    assert first[eval_keys.TRIAL_INDEX] == 0
    assert first[eval_keys.SUCCESS] is True
    assert first[eval_keys.ENDED_BY] == eval_keys.ENDED_BY_OPERATOR
    assert second[eval_keys.TRIAL_INDEX] == 1
    assert second[eval_keys.SUCCESS] is False


class _PageOperator(pimm.ControlSystem):
    """Presses the station page's buttons over HTTP from inside the world, as ``script`` says. Its return ends
    the world."""

    def __init__(self, script: Callable[[pimm.Clock], Iterator[pimm.Command]]):
        self._script = script

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        yield from self._script(clock)


def _wait(condition: Callable[[], Any], clock: pimm.Clock, seconds: float = 5.0) -> Iterator[pimm.Command]:
    """Yield until ``condition`` holds or ``seconds`` pass on ``clock``, and return whether it held."""
    deadline = clock.now() + seconds
    while not condition():
        if clock.now() > deadline:
            return False
        yield pimm.Sleep(0.02)
    return True


def _run_attended(
    marks: Path, script: Callable[[str, pimm.Clock], Iterator[pimm.Command]], move_s: float = 0.0, arm_failures: int = 0
) -> None:
    """The world ``web`` composes, with the station console in a process of its own and ``script`` pressing the
    page's buttons from beside the harness. The rig takes ``move_s`` for each move, and its arm's first
    ``arm_failures`` moves stop short."""
    port = _free_port()
    console = StationConsole(_trial('pick up the cube'), policy='stub', host='127.0.0.1', port=port)
    forwarder = TrialForwarder(partial(Rollout, policy=_MarkingPolicy(marks), output_path=None))
    embodiment = _embodiment(move_s=move_s, arm_failures=arm_failures)
    harness = Harness(embodiment)
    operator = _PageOperator(partial(script, f'http://127.0.0.1:{port}'))
    with pimm.World() as world:
        wire.wire_embodiment(world, harness, embodiment, done=forwarder.done)
        world.connect(console.trials, forwarder.trials)
        world.connect(forwarder.perform_task, harness.perform_task)
        world.run([operator, forwarder, harness, *embodiment.control_systems], [console])


def _post(base: str, path: str, body: EndBody | None = None) -> int:
    """The HTTP status of the press. A press the console refuses must not raise inside the world."""
    return httpx.post(f'{base}{path}', json=None if body is None else body.model_dump(mode='json')).status_code


def _phase(base: str) -> Phase | None:
    status = _status(base)
    return status.run.phase if status else None


def test_a_verdict_sent_with_start_is_discarded_before_the_rollout_begins():
    policy = _IdlePolicy()
    embodiment = _embodiment(simulated=True)
    harness = Harness(embodiment)
    forwarder = TrialForwarder(partial(Rollout, policy=policy, output_path=None))
    answers = []
    with pimm.World(virtual_time=True) as world:
        caller = world.pair(forwarder.trials)
        wire.wire_embodiment(world, harness, embodiment, done=forwarder.done)
        world.connect(forwarder.perform_task, harness.perform_task)

        def start_and_finish():
            answers.append(caller(Task('pick', 0.05)))
            answers.append(caller(EndTrial(_ForwarderBench.PAYLOAD)))

        script = scripted_driver((start_and_finish, 0.2))
        drive_scheduler(world.start([forwarder, harness, *embodiment.control_systems, script]))
    assert policy.observations
    assert [answer.result() for answer in answers] == [{eval_keys.TERMINATED: False}] * 2


@pytest.mark.timeout(120.0)
def test_a_verdict_ends_its_own_trial_and_not_the_next(tmp_path):
    phases: list[Phase | None] = []
    statuses: list[Status | None] = []

    verdicts: list[tuple[int, Verdict]] = [(1, Outcome.PASS), (2, Outcome.FAIL)]

    def script(base: str, clock: pimm.Clock):
        yield from _wait(lambda: _status(base) is not None, clock, seconds=60.0)
        for number, verdict in verdicts:
            _post(base, '/episode/start')
            yield from _wait((tmp_path / f'episode-{number}').exists, clock)
            yield pimm.Sleep(0.5)
            phases.append(_phase(base))
            _post(base, '/episode/end', EndBody(verdict=verdict))
            yield from _wait(lambda: _phase(base) is Phase.READY, clock)
        statuses.append(_status(base))

    _run_attended(tmp_path, script)

    assert phases == [Phase.RUNNING, Phase.RUNNING]
    [status] = statuses
    assert status is not None
    assert [episode.outcome for episode in status.run.episodes] == [Outcome.PASS, Outcome.FAIL]


@pytest.mark.timeout(120.0)
def test_a_verdict_does_not_end_the_next_trial_after_a_slow_move_back(tmp_path):
    """The rig takes a second for each move, so the harness answers each trial a second after its verdict. The next
    trial still runs until its own verdict."""
    phases: list[Phase | None] = []
    statuses: list[Status | None] = []
    verdicts: list[tuple[int, Verdict]] = [(1, Outcome.PASS), (2, Outcome.FAIL)]

    def script(base: str, clock: pimm.Clock):
        yield from _wait(lambda: _status(base) is not None, clock, seconds=60.0)
        for number, verdict in verdicts:
            _post(base, '/episode/start')
            yield from _wait((tmp_path / f'episode-{number}').exists, clock, seconds=10.0)
            yield pimm.Sleep(0.5)
            phases.append(_phase(base))
            _post(base, '/episode/end', EndBody(verdict=verdict))
            yield from _wait(lambda: _phase(base) is Phase.READY, clock, seconds=10.0)
        statuses.append(_status(base))

    _run_attended(tmp_path, script, move_s=1.0)

    assert phases == [Phase.RUNNING, Phase.RUNNING]
    [status] = statuses
    assert status is not None
    assert [episode.outcome for episode in status.run.episodes] == [Outcome.PASS, Outcome.FAIL]


@pytest.mark.timeout(120.0)
def test_a_home_that_stops_short_shows_its_error_and_the_next_start_runs(tmp_path):
    presses: list[int] = []
    statuses: list[Status | None] = []

    def ended(base: str) -> bool:
        status = _status(base)
        return status is not None and bool(status.run.episodes) and status.run.episodes[-1].outcome is not None

    def script(base: str, clock: pimm.Clock):
        yield from _wait(lambda: _status(base) is not None, clock, seconds=60.0)
        presses.append(_post(base, '/episode/start'))
        yield from _wait(lambda: ended(base), clock)
        statuses.append(_status(base))
        presses.append(_post(base, '/episode/start'))
        yield from _wait((tmp_path / 'episode-1').exists, clock)
        presses.append(_post(base, '/episode/end', EndBody(verdict=Outcome.PASS)))
        yield from _wait(lambda: _phase(base) is Phase.READY, clock)
        statuses.append(_status(base))

    _run_attended(tmp_path, script, arm_failures=1)

    assert presses == [200, 200, 200]
    failed, retried = statuses
    assert failed is not None and retried is not None
    assert failed.run.phase is Phase.READY
    [episode] = failed.run.episodes
    assert (episode.outcome, episode.error) == (Outcome.ERROR, f'RuntimeError: {ARM_STOPPED_SHORT}')
    assert [episode.outcome for episode in retried.run.episodes] == [Outcome.ERROR, Outcome.PASS]


@pytest.mark.timeout(240.0)
def test_end_run_waits_for_the_open_episode_and_then_stops_the_whole_command(tmp_path):
    """End run stops the World, so each device runs its teardown and the command returns without a signal. A camera
    tile stays open on the page through it."""
    port = _free_port()
    base = f'http://127.0.0.1:{port}'
    marks = tmp_path / 'marks'
    marks.mkdir()
    run = multiprocessing.get_context('spawn').Process(target=_serve_station, args=(port, tmp_path / 'episodes', marks))
    run.start()
    try:
        _wait_for(lambda: _status(base), 'the console to answer')
        httpx.post(f'{base}/episode/start').raise_for_status()
        _wait_for((marks / 'episode-1').exists, 'episode 1 to open')
        refused = httpx.post(f'{base}/run/end')
        assert refused.status_code == 409
        assert 'finish the episode first' in refused.json()['detail']

        verdict = EndBody(verdict=Outcome.PASS).model_dump(mode='json')
        httpx.post(f'{base}/episode/end', json=verdict).raise_for_status()
        _wait_for(lambda: (s := _status(base)) and s.run.phase is Phase.READY, 'episode 1 to close')
        with connect(f'ws://127.0.0.1:{port}/video/{keys.WRIST_IMAGE}', open_timeout=10) as tile:
            assert isinstance(tile.recv(timeout=10), str)
            ended = httpx.post(f'{base}/run/end')
            ended.raise_for_status()
            assert Status.model_validate(ended.json()).run.phase is Phase.RUN_ENDED
            run.join(timeout=60)
        assert run.exitcode == 0
        assert (marks / 'parked').exists()
    finally:
        if run.pid is not None and run.is_alive():
            os.kill(run.pid, signal.SIGINT)
        run.join(timeout=120)
        if run.is_alive():
            run.kill()
