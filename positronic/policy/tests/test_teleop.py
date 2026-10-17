import queue
import time

import numpy as np
import pytest

from positronic import geom, keys
from positronic.drivers.roboarm.command import CartesianPosition
from positronic.policy import keys as policy_keys
from positronic.policy.base import PolicyEnd, Step
from positronic.policy.executor import Executor
from positronic.policy.spec import from_spec
from positronic.policy.teleop import TeleopClient, TrialNotStarted
from positronic.teleop import protocol
from positronic.teleop.protocol import Reply, Request

ROBOT_POSE = geom.Transform3D(np.array([0.4, 0.0, 0.3]), geom.Rotation.from_quat([0.0, 1.0, 0.0, 0.0]))
TARGET = geom.Transform3D(np.array([0.45, 0.05, 0.35]), geom.Rotation.identity)
OBS = {
    keys.EE_POSE: ROBOT_POSE.as_vector(geom.Rotation.Representation.QUAT),
    keys.GRIP: 0.25,
    keys.TASK: 'pick up the cube',
    keys.EXTERIOR_IMAGE: np.full((48, 64, 3), 128, dtype=np.uint8),
}


class _Service:
    """The teleop service as the client sees it through ``infer``: it records each request, and answers from
    ``replies``."""

    def __init__(self):
        self.requests: list[dict] = []
        self.replies: queue.Queue[dict] = queue.Queue()

    def __call__(self, request: dict) -> dict:
        self.requests.append(request)
        try:
            return self.replies.get(timeout=5)
        except queue.Empty:
            raise TimeoutError('no reply') from None


@pytest.fixture
def service():
    return _Service()


class _Clock:
    """An episode clock that moves 1 ns at each read, from a time that a test can set."""

    def __init__(self):
        self.ns = 0

    def __call__(self) -> int:
        self.ns += 1
        return self.ns


@pytest.fixture
def clock():
    return _Clock()


@pytest.fixture
def runtime(clock):
    executor = Executor(clock, simulated=False, charge_inference_time=False)
    yield executor
    executor.close()


def _frame(target: geom.Transform3D | None = TARGET, grip: float = 0.5, start_task=False, end=None) -> dict:
    reply = {protocol.KIND: Reply.FRAME, protocol.START_TASK: start_task}
    if target is not None:
        reply[protocol.TARGET] = target.as_vector(geom.Rotation.Representation.QUAT)
        reply[protocol.GRIP] = grip
    if end is not None:
        reply[protocol.END] = end
    return reply


def _holds_within(condition, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.001)
    return True


def _client(**args) -> TeleopClient:
    defaults = protocol.SessionParams()
    return TeleopClient(**{
        'ready_backstop_sec': 5.0,
        'cameras': defaults.cameras,
        'image_fps': defaults.image_fps,
        'still_clock_timeout': defaults.still_clock_timeout,
        **args,
    })


def _started(runtime: Executor, service: _Service, **client_args):
    """A run of a fresh client that the teleoperator readied, after its stay command."""
    service.replies.put({protocol.KIND: Reply.READY})
    run = runtime.start(_client(**client_args), service)
    stay = run.send(OBS)
    assert stay is not None
    assert _holds_within(lambda: len(service.requests) == 2, seconds=2.0), 'the run sent no frame request'
    return run, stay


def _answer(run, service: _Service, reply: dict) -> Step | PolicyEnd:
    """Send ``reply`` to the pending frame request. Return the step that the run makes from it, or its end."""
    service.replies.put(reply)
    sent = len(service.requests)
    for _ in range(2000):
        try:
            step = run.send(OBS)
        except StopIteration as end:
            return end.value
        if _holds_within(lambda: len(service.requests) > sent, seconds=0.005):
            assert step is not None
            return step
    raise AssertionError('the run made no step from the frame')


def test_the_run_asks_for_ready_until_the_service_answers_it_before_its_first_step(runtime, service):
    for reply in (Reply.WAITING, Reply.WAITING, Reply.READY):
        service.replies.put({protocol.KIND: reply})
    runtime.start(_client(), service)
    assert service.requests == [{protocol.KIND: Request.READY}] * 3
    assert not runtime.has_pending


@pytest.mark.parametrize(('reply', 'match'), [(Reply.CANCELLED, 'cancelled'), (Reply.EXPIRED, 'ready limit')])
def test_a_cancel_or_the_ready_limit_fails_the_run_before_its_first_step(runtime, service, reply, match):
    service.replies.put({protocol.KIND: reply})
    with pytest.raises(TrialNotStarted, match=match):
        runtime.start(_client(), service)


def test_a_service_that_waits_past_the_backstop_fails_the_run_before_its_first_step(runtime):
    def waiting(_request: dict) -> dict:
        time.sleep(0.01)
        return {protocol.KIND: Reply.WAITING}

    with pytest.raises(TimeoutError, match='no Ready within 0.05 s'):
        runtime.start(_client(ready_backstop_sec=0.05), waiting)


def test_the_first_step_holds_the_robot_where_it_is(runtime, service):
    _, stay = _started(runtime, service, cameras=[keys.EXTERIOR_IMAGE])
    command = stay.commands[keys.ROBOT_COMMAND]
    assert isinstance(command, CartesianPosition)
    np.testing.assert_allclose(command.pose.as_vector('quat'), OBS[keys.EE_POSE])
    assert stay.commands[keys.TARGET_GRIP] == OBS[keys.GRIP]
    request = service.requests[1]
    images = request.pop(protocol.IMAGES)
    assert {name: image.shape for name, image in images.items()} == {keys.EXTERIOR_IMAGE: (48, 64, 3)}
    np.testing.assert_allclose(request.pop(protocol.EE_POSE), OBS[keys.EE_POSE])
    assert request == {protocol.KIND: Request.FRAME, protocol.TASK: OBS[keys.TASK]}


def test_a_reply_with_no_target_sends_no_command(runtime, service):
    run, _ = _started(runtime, service)
    step = _answer(run, service, _frame(None))
    assert isinstance(step, Step) and step.commands == {}


def test_a_reply_with_a_target_moves_the_robot_there_with_its_grip(runtime, service):
    run, _ = _started(runtime, service)
    step = _answer(run, service, _frame(TARGET, 0.8))
    assert isinstance(step, Step)
    np.testing.assert_allclose(step.commands[keys.ROBOT_COMMAND].pose.as_vector('quat'), TARGET.as_vector('quat'))
    assert step.commands[keys.TARGET_GRIP] == 0.8


def test_start_task_marks_the_attempt_start_once(runtime, service, clock):
    run, _ = _started(runtime, service)
    clock.ns = 3_000_000_000
    _answer(run, service, _frame(start_task=True))
    clock.ns = 5_000_000_000
    _answer(run, service, _frame(start_task=True))
    [(name, start_ns)] = runtime.metadata_snapshot().items()
    assert name == policy_keys.ATTEMPT_START_NS and 3_000_000_000 < start_ns < 5_000_000_000


@pytest.mark.parametrize('end', list(PolicyEnd))
def test_an_end_in_the_reply_ends_the_run_with_it(runtime, service, end):
    run, _ = _started(runtime, service)
    assert _answer(run, service, _frame(end=end.value)) is end


def test_a_start_task_and_an_end_in_one_reply_mark_the_start_before_the_end(runtime, service):
    run, _ = _started(runtime, service)
    assert _answer(run, service, _frame(start_task=True, end=PolicyEnd.COMPLETE.value)) is PolicyEnd.COMPLETE
    assert policy_keys.ATTEMPT_START_NS in runtime.metadata_snapshot()


def _sent_cameras(runtime, service, obs, **client_args) -> list[str]:
    """The cameras of the first frame request of a readied run, in the order that the request holds them."""
    service.replies.put({protocol.KIND: Reply.READY})
    run = runtime.start(_client(**client_args), service)
    run.send(obs)
    assert _holds_within(lambda: len(service.requests) == 2, seconds=2.0)
    return list(service.requests[1][protocol.IMAGES])


MORE_CAMERAS = {
    **OBS,
    keys.WRIST_IMAGE: np.zeros((24, 32, 3), dtype=np.uint8),
    keys.HANDCAM_RIGHT_IMAGE: OBS[keys.EXTERIOR_IMAGE],
}


def test_the_run_sends_no_camera_by_default(runtime, service):
    assert _sent_cameras(runtime, service, MORE_CAMERAS) == []


def test_the_run_sends_the_cameras_that_it_names_in_their_order(runtime, service):
    cameras = [keys.HANDCAM_RIGHT_IMAGE, keys.WRIST_IMAGE]
    assert _sent_cameras(runtime, service, MORE_CAMERAS, cameras=cameras) == cameras


def test_the_run_sends_images_at_most_at_its_image_rate(runtime, service):
    run, _ = _started(runtime, service, cameras=[keys.EXTERIOR_IMAGE], image_fps=1.0)
    _answer(run, service, _frame())
    _answer(run, service, _frame())
    assert protocol.IMAGES in service.requests[1]
    assert all(protocol.IMAGES not in request for request in service.requests[2:])


@pytest.mark.parametrize(('limit', 'fails'), [(0.1, True), (30.0, False)])
def test_an_episode_clock_that_stands_still_for_the_limit_fails_the_run(service, limit, fails):
    runtime = Executor(lambda: 0, simulated=True, charge_inference_time=False)
    try:
        run, _ = _started(runtime, service, still_clock_timeout=limit)
        deadline = time.monotonic() + 0.5
        failure = None
        while time.monotonic() < deadline and failure is None:
            try:
                run.send(OBS)
            except RuntimeError as raised:
                failure = raised
            time.sleep(0.01)
        assert (failure is not None) is fails
        if failure is not None:
            assert 'stands still' in str(failure)
    finally:
        service.replies.put(_frame())
        runtime.close()


@pytest.mark.parametrize('cameras', [(), [keys.WRIST_IMAGE]])
def test_a_declared_client_builds_from_its_spec(cameras):
    client = TeleopClient(10.0, cameras=cameras, image_fps=5.0, still_clock_timeout=2.0)
    built = from_spec(client.to_spec())
    assert isinstance(built, TeleopClient) and built.to_spec() == client.to_spec()
