import io
import logging
import multiprocessing
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
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
from positronic.drivers.recording import HARNESS_WALL, HARNESS_WORLD
from positronic.eval import Embodiment, Observation, Task
from positronic.eval import keys as eval_keys
from positronic.gui.station import Episode, Outcome, Phase, Verdict
from positronic.gui.web import EndBody, InstructionBody, StationConsole, Status
from positronic.inference import KeyboardOperator, real, web
from positronic.policy import Policy
from positronic.policy.base import Step
from positronic.policy.harness import EpisodeNotStarted, Harness, Rollout
from positronic.tests.testing_coutils import drive_scheduler, scripted_driver
from positronic.tests.testing_inference import (
    ARM_STOPPED_SHORT,
    PageOperator,
    ReadyDevices,
    free_port,
    make_embodiment,
    make_trial,
    post,
    read_status,
    run_attended,
    wait,
)


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


@pytest.mark.timeout(30.0)
def test_the_keyboard_path_ends_when_the_keyboard_returns(monkeypatch):
    """How an attended run finishes: the operator returns, the world stops, no policy run is started.

    A stdin that is not a terminal is the return the test can force; ``q`` is the other one.
    """
    monkeypatch.setattr(sys, 'stdin', io.StringIO())
    policy = _IdlePolicy()

    real(policy=policy, embodiment=make_embodiment(), next_task=make_trial())

    assert not policy.closed
    assert not policy.started


def test_the_keyboard_path_refuses_a_simulated_embodiment():
    """It composes a real-time world and records against the wall clock, which a simulated embodiment needs
    neither of."""
    with pytest.raises(ValueError, match='sim'):
        real(policy=_IdlePolicy(), embodiment=make_embodiment(simulated=True), next_task=make_trial())


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
        real(policy=policy, embodiment=make_embodiment(), next_task=make_trial('pick up the cube'))

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


def test_the_next_press_after_an_episode_that_did_not_start_runs_the_same_trial(monkeypatch):
    drawn = iter([Task(instruction_source=f'trial {n}', timeout_sec=None) for n in range(2)])
    presses = iter(['s', *[None] * 9, 's'])
    monkeypatch.setattr(keyboard, 'key_reader', partial(nullcontext, lambda: next(presses, None)))
    operator = KeyboardOperator(lambda: next(drawn), _IdlePolicy(), None)
    with pimm.World(virtual_time=True) as world:
        harness = world.pair(operator.perform_task)
        received = []

        def not_started():
            for call in harness.incoming():
                received.append(call.request.task.instruction)
                call.set_exception(EpisodeNotStarted('RuntimeError: the teleoperator cancelled'))

        def hold_the_ask():
            received.extend(call.request.task.instruction for call in harness.incoming())

        steps = [(not_started, 0.05), *[(hold_the_ask, 0.02)] * 10]
        drive_scheduler(world.start([operator, scripted_driver(*steps)]))

    assert received == ['trial 0', 'trial 0']


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
        web(policy=_IdlePolicy(), embodiment=make_embodiment(simulated=True), next_task=make_trial())


class _LockedPolicy(_IdlePolicy):
    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()


def test_a_policy_with_live_resources_fails_when_the_console_process_starts():
    console = StationConsole(make_trial(), _LockedPolicy(), None, host='127.0.0.1', port=0)
    with pimm.World() as world:
        with pytest.raises(RuntimeError, match='Background control systems must be picklable') as failed:
            world.run([], [console])
    assert isinstance(failed.value.__cause__, TypeError)
    assert 'lock' in str(failed.value.__cause__)


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
    """Checks array snapshots and marks each episode after five policy steps. With a ``camera_key``, the mark also
    waits for five different frames under that observation key."""

    def __init__(self, marks: Path, camera_key: str | None = None):
        self._marks = marks
        self._camera_key = camera_key

    def run(self, runtime):
        number = len(list(self._marks.glob('episode-*'))) + 1
        obs = yield
        steps = 0
        frames_seen: set[bytes] = set()
        while True:
            snapshots = [(value, value.copy()) for value in obs.values() if isinstance(value, np.ndarray)]
            obs = yield Step({}, runtime.time_ns + 100_000_000)
            for value, snapshot in snapshots:
                np.testing.assert_array_equal(value, snapshot)
            steps += 1
            if self._camera_key is not None:
                frames_seen.add(obs[self._camera_key].tobytes())
            if steps >= 5 and (self._camera_key is None or len(frames_seen) >= 5):
                (self._marks / f'episode-{number}').touch()


def _serve_station(port: int, output_dir: Path, marks: Path) -> None:
    camera, devices = _Camera(), ReadyDevices(parked=marks / 'parked')
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
        web(
            _MarkingPolicy(marks, camera_key=keys.WRIST_IMAGE),
            embodiment,
            make_trial('pick up the cube'),
            output_dir=str(output_dir),
            port=port,
        )


def _wait_for(condition: Callable[[], Any], what: str, timeout: float = 90.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if value := condition():
            return value
        time.sleep(0.1)
    raise AssertionError(f'timed out waiting for {what}')


def _run_episode(base: str, marks: Path, number: int, verdict: Verdict) -> Episode:
    httpx.post(f'{base}/episode/start').raise_for_status()
    _wait_for((marks / f'episode-{number}').exists, f'episode {number} to open')
    httpx.post(f'{base}/episode/end', json=EndBody(verdict=verdict).model_dump(mode='json')).raise_for_status()

    def closed() -> Status | None:
        status = read_status(base)
        return status if status and status.run.phase is Phase.READY else None

    return _wait_for(closed, f'episode {number} to close').run.episodes[number - 1]


@pytest.mark.timeout(240.0)
def test_the_web_console_records_each_episode_with_its_instruction_and_verdict(tmp_path):
    """The whole command, in a process of its own: the page streams the camera, and each episode it starts
    records the instruction it sent, the override flag, its number, and the operator's verdict."""
    port = free_port()
    base = f'http://127.0.0.1:{port}'
    marks = tmp_path / 'marks'
    marks.mkdir()
    output_dir = tmp_path / 'episodes'
    run = multiprocessing.get_context('spawn').Process(target=_serve_station, args=(port, output_dir, marks))
    run.start()
    try:
        status = _wait_for(lambda: (s := read_status(base)) and s.cameras[0].live and s, 'the camera to be live')
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
            assert set(camera.timelines) == {
                EMITTED_WALL,
                EMITTED_WORLD,
                RECEIVED_WALL,
                RECEIVED_WORLD,
                HARNESS_WALL,
                HARNESS_WORLD,
                _CAMERA_FRAME,
            }
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
                    assert timestamp[HARNESS_WORLD] == timestamp[RECEIVED_WORLD]
                    assert timestamp[HARNESS_WALL] == timestamp[RECEIVED_WALL]
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


def _phase(base: str) -> Phase | None:
    status = read_status(base)
    return status.run.phase if status else None


def test_a_verdict_sent_with_the_ask_keeps_the_episode_from_starting_and_the_next_ask_runs():
    policy = _IdlePolicy()
    embodiment = make_embodiment(simulated=True)
    harness = Harness(embodiment)
    answers = []
    verdict = {eval_keys.ENDED_BY: eval_keys.ENDED_BY_OPERATOR, eval_keys.SUCCESS: True}
    with pimm.World(virtual_time=True) as world:
        caller = world.pair(harness.perform_task)
        done = world.pair(harness.done)
        wire.wire_embodiment(world, harness, embodiment)

        def script(clock):
            answers.append(caller(Rollout(Task('pick', 0.05), policy, None)))
            done.emit(verdict)
            yield from wait(answers[0].done, 'the answer to the first ask', clock)
            answers.append(caller(Rollout(Task('pick', 0.05), policy, None)))
            yield from wait(lambda: policy.observations, 'an observation', clock)
            done.emit(verdict)
            yield from wait(answers[1].done, 'the answer to the second ask', clock)

        drive_scheduler(world.start([harness, *embodiment.control_systems, PageOperator(script)]))
    with pytest.raises(EpisodeNotStarted, match='a done signal arrived before the episode started'):
        answers[0].result()
    assert answers[1].result() == {**verdict, eval_keys.TERMINATED: True}


@pytest.mark.timeout(120.0)
def test_a_verdict_ends_its_own_trial_and_not_the_next(tmp_path):
    phases: list[Phase | None] = []
    statuses: list[Status | None] = []

    verdicts: list[tuple[int, Verdict]] = [(1, Outcome.PASS), (2, Outcome.FAIL)]

    def script(base: str, clock: pimm.Clock):
        yield from wait(lambda: read_status(base) is not None, 'the console to answer', clock, seconds=60.0)
        for number, verdict in verdicts:
            post(base, '/episode/start')
            yield from wait((tmp_path / f'episode-{number}').exists, f'episode {number} to open', clock)
            yield pimm.Sleep(0.5)
            phases.append(_phase(base))
            post(base, '/episode/end', EndBody(verdict=verdict))
            yield from wait(lambda: _phase(base) is Phase.READY, 'the console to be ready', clock)
        statuses.append(read_status(base))

    run_attended(_MarkingPolicy(tmp_path), script)

    assert phases == [Phase.RUNNING, Phase.RUNNING]
    [status] = statuses
    assert status is not None
    assert [episode.outcome for episode in status.run.episodes] == [Outcome.PASS, Outcome.FAIL]


@pytest.mark.timeout(120.0)
def test_repeated_verdicts_during_a_slow_move_back_do_not_end_the_next_trial(tmp_path):
    """The rig takes a second for each move, so the harness answers each trial a second after its verdict. The next
    trial still runs until its own verdict."""
    phases: list[Phase | None] = []
    statuses: list[Status | None] = []
    verdicts: list[tuple[int, Verdict]] = [(1, Outcome.PASS), (2, Outcome.FAIL)]

    def script(base: str, clock: pimm.Clock):
        yield from wait(lambda: read_status(base) is not None, 'the console to answer', clock, seconds=60.0)
        for number, verdict in verdicts:
            post(base, '/episode/start')
            yield from wait((tmp_path / f'episode-{number}').exists, f'episode {number} to open', clock)
            yield pimm.Sleep(0.5)
            phases.append(_phase(base))
            post(base, '/episode/end', EndBody(verdict=verdict))
            yield pimm.Sleep(0.2)
            assert post(base, '/episode/end', EndBody(verdict=verdict)) == 200
            yield from wait(lambda: _phase(base) is Phase.READY, 'the console to be ready', clock)
        statuses.append(read_status(base))

    run_attended(_MarkingPolicy(tmp_path), script, move_s=1.0)

    assert phases == [Phase.RUNNING, Phase.RUNNING]
    [status] = statuses
    assert status is not None
    assert [episode.outcome for episode in status.run.episodes] == [Outcome.PASS, Outcome.FAIL]


@pytest.mark.timeout(120.0)
def test_a_home_that_stops_short_shows_its_error_and_the_next_start_runs(tmp_path):
    presses: list[int] = []
    statuses: list[Status | None] = []

    def ended(base: str) -> bool:
        status = read_status(base)
        return status is not None and bool(status.run.episodes) and status.run.episodes[-1].outcome is not None

    def script(base: str, clock: pimm.Clock):
        yield from wait(lambda: read_status(base) is not None, 'the console to answer', clock, seconds=60.0)
        presses.append(post(base, '/episode/start'))
        yield from wait(lambda: ended(base), 'the episode to end', clock)
        statuses.append(read_status(base))
        presses.append(post(base, '/episode/start'))
        yield from wait((tmp_path / 'episode-1').exists, 'episode 1 to open', clock)
        presses.append(post(base, '/episode/end', EndBody(verdict=Outcome.PASS)))
        yield from wait(lambda: _phase(base) is Phase.READY, 'the console to be ready', clock)
        statuses.append(read_status(base))

    run_attended(_MarkingPolicy(tmp_path), script, arm_failures=1)

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
    port = free_port()
    base = f'http://127.0.0.1:{port}'
    marks = tmp_path / 'marks'
    marks.mkdir()
    run = multiprocessing.get_context('spawn').Process(target=_serve_station, args=(port, tmp_path / 'episodes', marks))
    run.start()
    try:
        _wait_for(lambda: read_status(base), 'the console to answer')
        httpx.post(f'{base}/episode/start').raise_for_status()
        _wait_for((marks / 'episode-1').exists, 'episode 1 to open')
        refused = httpx.post(f'{base}/run/end')
        assert refused.status_code == 409
        assert 'finish the episode first' in refused.json()['detail']

        verdict = EndBody(verdict=Outcome.PASS).model_dump(mode='json')
        httpx.post(f'{base}/episode/end', json=verdict).raise_for_status()
        _wait_for(lambda: (s := read_status(base)) and s.run.phase is Phase.READY, 'episode 1 to close')
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
