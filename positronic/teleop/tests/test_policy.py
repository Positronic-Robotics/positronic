import pickle
import queue
import time
from pathlib import Path

import numpy as np
import pytest
from positronic_model_server.serialization import deserialise, serialise
from positronic_wire import registry
from positronic_wire.wire import ClientConnection, ClientWire, ConnectRefused, Refusal, UnixSocketAddress

from positronic import geom, keys
from positronic.drivers.roboarm.command import CartesianPosition
from positronic.policy import keys as policy_keys
from positronic.policy.base import PolicyEnd, Step
from positronic.policy.executor import Executor
from positronic.teleop import TeleopPolicy, session
from positronic.teleop.clutch import OperatorPosition

WIRE = 'fake_teleop'
ADDRESS = UnixSocketAddress(path='/session', query='', uds=Path('/run/teleop.sock'))
ROBOT_POSE = geom.Transform3D(np.array([0.4, 0.0, 0.3]), geom.Rotation.from_quat([0.0, 1.0, 0.0, 0.0]))
OBS = {
    keys.EE_POSE: ROBOT_POSE.as_vector(geom.Rotation.Representation.QUAT),
    keys.GRIP: 0.25,
    keys.TASK: 'pick up the cube',
}
HAND = geom.Transform3D(np.array([0.1, 0.2, 1.0]), geom.Rotation.identity)


class _Service(ClientConnection):
    """The teleop service as the policy sees it: it records each request, and answers from ``replies``."""

    def __init__(self):
        self.requests: list[dict] = []
        self.replies: queue.Queue[dict] = queue.Queue()
        self.closed = False

    def send(self, message: bytes) -> None:
        self.requests.append(deserialise(message))

    def recv(self, timeout: float | None = None) -> bytes:
        try:
            return serialise(self.replies.get(timeout=timeout))
        except queue.Empty:
            raise TimeoutError('no reply') from None

    def close(self) -> str:
        self.closed = True
        return 'closed'


class _Wire(ClientWire[UnixSocketAddress]):
    NAME = WIRE
    ADDRESS = UnixSocketAddress

    def __init__(self, service: _Service, refusal: Refusal | None = None):
        self.service = service
        self.refusal = refusal
        self.probes = 0

    def session_url(self, address: UnixSocketAddress) -> str:
        return f'fake+unix://{address.uds}'

    def dial(self, address, headers, open_timeout) -> ClientConnection:
        return self.service

    def keepalive(self, address, headers, timeout) -> int | None:
        return None

    def probe(self, address, headers, open_timeout) -> Refusal | None:
        self.probes += 1
        return self.refusal


@pytest.fixture
def service(monkeypatch):
    service = _Service()
    monkeypatch.setitem(registry.CLIENT_WIRES, WIRE, _Wire(service))
    return service


@pytest.fixture
def clock():
    """The episode time in ns, as a list that a test can set."""
    return [0]


@pytest.fixture
def runtime(clock):
    executor = Executor(lambda: clock[0], simulated=False, charge_inference_time=False)
    yield executor
    executor.close()


def _frame(pose: geom.Transform3D | None = HAND, grip: float = 0.5, *events: session.Event) -> dict:
    reply = {session.KIND: session.Reply.FRAME, session.GRIP: grip, session.EVENTS: [str(e) for e in events]}
    if pose is not None:
        reply[session.POSE] = pose.as_vector(geom.Rotation.Representation.QUAT)
    return reply


def _wait(condition, seconds: float = 2.0) -> bool:
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.001)
    return True


def _started(runtime: Executor, service: _Service):
    """A run of a fresh policy that the teleoperator readied, after its stay command."""
    run = runtime.start(TeleopPolicy(WIRE, ADDRESS, OperatorPosition.FRONT))
    service.replies.put({session.KIND: session.Reply.READY})
    assert _wait(lambda: all(call.done() for call in runtime.ready_calls))
    stay = run.send(OBS)
    assert stay is not None
    assert _wait(lambda: len(service.requests) == 2)
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
        if _wait(lambda: len(service.requests) > sent, seconds=0.005):
            assert step is not None
            return step
    raise AssertionError('the run made no step from the frame')


def test_the_run_asks_the_service_for_ready_before_its_first_step(runtime, service):
    runtime.start(TeleopPolicy(WIRE, ADDRESS))
    [ready] = runtime.ready_calls
    assert _wait(lambda: service.requests == [{session.KIND: session.Request.READY}])
    assert not ready.done()
    service.replies.put({session.KIND: session.Reply.READY})
    assert _wait(ready.done)
    ready.result()


def test_a_cancel_fails_the_ready_call(runtime, service):
    runtime.start(TeleopPolicy(WIRE, ADDRESS))
    service.replies.put({session.KIND: session.Reply.CANCELLED})
    [ready] = runtime.ready_calls
    assert _wait(ready.done)
    with pytest.raises(session.Cancelled):
        ready.result()


def test_a_service_error_fails_the_ready_call(runtime, service):
    runtime.start(TeleopPolicy(WIRE, ADDRESS))
    service.replies.put({session.KIND: session.Reply.ERROR, session.MESSAGE: 'the ready limit passed'})
    [ready] = runtime.ready_calls
    assert _wait(ready.done)
    with pytest.raises(session.ServiceError, match='the ready limit passed'):
        ready.result()


def test_the_first_step_holds_the_robot_where_it_is(runtime, service):
    _, stay = _started(runtime, service)
    command = stay.commands[keys.ROBOT_COMMAND]
    assert isinstance(command, CartesianPosition)
    np.testing.assert_allclose(command.pose.as_vector('quat'), OBS[keys.EE_POSE])
    assert stay.commands[keys.TARGET_GRIP] == OBS[keys.GRIP]
    assert service.requests[1] == {
        session.KIND: session.Request.FRAME,
        session.DISPLAY: {session.TASK: OBS[keys.TASK], session.CONTROL: False, session.ATTEMPT_STARTED: False},
    }


def test_no_command_goes_out_while_control_is_off(runtime, service):
    run, _ = _started(runtime, service)
    step = _answer(run, service, _frame())
    assert isinstance(step, Step) and step.commands == {}


def test_control_starts_from_the_robot_pose_and_then_follows_the_hand(runtime, service):
    run, _ = _started(runtime, service)
    on = _answer(run, service, _frame(HAND, 0.5, session.Event.CLUTCH))
    assert isinstance(on, Step)
    np.testing.assert_allclose(on.commands[keys.ROBOT_COMMAND].pose.as_vector('quat'), OBS[keys.EE_POSE], atol=1e-9)
    assert on.commands[keys.TARGET_GRIP] == 0.5
    assert service.requests[-1][session.DISPLAY][session.CONTROL] is True

    moved = geom.Transform3D(HAND.translation + np.array([0.0, 0.0, 0.05]), HAND.rotation)
    follow = _answer(run, service, _frame(moved, 0.8))
    assert isinstance(follow, Step)
    shift = follow.commands[keys.ROBOT_COMMAND].pose.translation - ROBOT_POSE.translation
    np.testing.assert_allclose(np.linalg.norm(shift), 0.05, atol=1e-9)
    assert follow.commands[keys.TARGET_GRIP] == 0.8


def test_a_second_clutch_press_turns_control_off(runtime, service):
    run, _ = _started(runtime, service)
    _answer(run, service, _frame(HAND, 0.5, session.Event.CLUTCH))
    off = _answer(run, service, _frame(HAND, 0.5, session.Event.CLUTCH))
    assert isinstance(off, Step) and off.commands == {}


def test_a_hand_that_the_headset_loses_turns_control_off(runtime, service):
    run, _ = _started(runtime, service)
    _answer(run, service, _frame(HAND, 0.5, session.Event.CLUTCH))
    lost = _answer(run, service, _frame(None))
    found = _answer(run, service, _frame(HAND))
    assert isinstance(lost, Step) and lost.commands == {}
    assert isinstance(found, Step) and found.commands == {}


def test_a_clutch_press_while_the_hand_is_lost_does_nothing(runtime, service):
    run, _ = _started(runtime, service)
    pressed = _answer(run, service, _frame(None, 0.5, session.Event.CLUTCH))
    after = _answer(run, service, _frame(HAND))
    assert isinstance(pressed, Step) and pressed.commands == {}
    assert isinstance(after, Step) and after.commands == {}


def test_start_task_marks_the_attempt_start_once(runtime, service, clock):
    run, _ = _started(runtime, service)
    clock[0] = 3_000_000_000
    _answer(run, service, _frame(HAND, 0.5, session.Event.START_TASK))
    clock[0] = 5_000_000_000
    _answer(run, service, _frame(HAND, 0.5, session.Event.START_TASK))
    assert runtime.metadata_snapshot() == {policy_keys.ATTEMPT_START_NS: 3_000_000_000}
    assert service.requests[-1][session.DISPLAY][session.ATTEMPT_STARTED] is True


@pytest.mark.parametrize(
    ('event', 'end'), [(session.Event.COMPLETE, PolicyEnd.COMPLETE), (session.Event.GIVE_UP, PolicyEnd.GIVE_UP)]
)
def test_complete_and_give_up_end_the_run(runtime, service, event, end):
    run, _ = _started(runtime, service)
    assert _answer(run, service, _frame(HAND, 0.5, event)) is end


def test_the_run_closes_its_session(runtime, service):
    run, _ = _started(runtime, service)
    _answer(run, service, _frame(HAND, 0.5, session.Event.GIVE_UP))
    assert service.closed


def test_meta_probes_the_service_on_the_first_call_only(monkeypatch):
    wire = _Wire(_Service())
    monkeypatch.setitem(registry.CLIENT_WIRES, WIRE, wire)
    policy = TeleopPolicy(WIRE, ADDRESS, OperatorPosition.BACK)
    assert policy.meta() == {policy_keys.TYPE: 'teleop', 'operator_position': 'BACK'}
    policy.meta()
    assert wire.probes == 1


def test_meta_raises_when_the_service_does_not_answer(monkeypatch):
    monkeypatch.setitem(registry.CLIENT_WIRES, WIRE, _Wire(_Service(), Refusal.SILENT))
    with pytest.raises(ConnectRefused):
        TeleopPolicy(WIRE, ADDRESS).meta()


def test_a_probed_policy_pickles_with_its_probe(monkeypatch):
    wire = _Wire(_Service())
    monkeypatch.setitem(registry.CLIENT_WIRES, WIRE, wire)
    policy = TeleopPolicy(WIRE, ADDRESS)
    policy.meta()
    pickle.loads(pickle.dumps(policy)).meta()
    assert wire.probes == 1
