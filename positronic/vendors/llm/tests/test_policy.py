import io
import json
import threading
from contextlib import contextmanager

import numpy as np
import pytest
from PIL import Image
from pydantic_ai.messages import BinaryContent, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart

from positronic import keys
from positronic.drivers.roboarm import RobotStatus
from positronic.policy.executor import Executor
from positronic.policy.layers import ChunkedSchedule, StopOnFault
from positronic.vendors.llm.client import Endpoint
from positronic.vendors.llm.motion import Motion
from positronic.vendors.llm.policy import Images, LLMPolicy, llm


def observation(time_ns=0, x=0.0):
    return {
        keys.EE_POSE: np.array([x, 0, 0, 1, 0, 0, 0]),
        keys.GRIP: 0.0,
        keys.TASK: 'Move the cube.',
        keys.OBS_TIME_NS: time_ns,
        keys.WRIST_IMAGE: np.zeros((10, 20, 3), dtype=np.uint8),
        keys.EXTERIOR_IMAGE: np.ones((10, 20, 3), dtype=np.uint8),
        keys.ROBOT_STATUS: RobotStatus.AVAILABLE,
        'sim_state': 'privileged-value-never-send',
    }


def move(x=0.01):
    return ModelResponse([
        ToolCallPart(
            'move_to',
            {'x': x, 'y': 0, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': 0, 'gripper': 0, 'note': 'Approach the cube.'},
            tool_call_id='move',
        )
    ])


def finish(tool='done'):
    return ModelResponse([
        ToolCallPart(tool, {'reason': 'Attempt ended.', 'hindsight': 'Inspect the final image.'}, tool_call_id='finish')
    ])


@pytest.fixture
def model(monkeypatch):
    requests, replies = [], []

    def request(endpoint, messages, tools, transcript, call, obs_time_ns):
        requests.append((messages, tools))
        response = replies.pop(0)
        return response() if callable(response) else response

    monkeypatch.setattr(Endpoint, 'request', request)
    return requests, replies


@contextmanager
def session(policy, directory=None):
    rt = Executor(policy.functions, artifact_dir=directory)
    active = policy.new_session(rt=rt)
    try:
        yield active, rt
    finally:
        active.cancel()
        rt.close()
        active.close()


def complete(active, rt, obs, time_ns=0):
    assert active(obs, time_ns) is None
    rt.wait(5)
    assert not rt.in_flight
    return active(obs, time_ns)


def frames(messages):
    return [
        item
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
        for item in part.content
        if isinstance(item, BinaryContent)
    ]


def test_history_keeps_two_observations_images_and_reports_actual_pose(model, tmp_path):
    requests, replies = model
    replies.extend([move(), move(), finish()])
    policy = LLMPolicy(Endpoint('test'), Motion(), image_size=8)
    with session(policy, tmp_path) as (active, rt):
        complete(active, rt, observation(1))
        complete(active, rt, observation(2))
        assert complete(active, rt, observation(3)) == []
        assert active.stop_requested
        assert active.meta['hindsight'] == 'Inspect the final image.'
    assert [len(frames(messages)) for messages, _ in requests] == [2, 4, 4]
    second = requests[1][0]
    states = [
        json.loads(part.content)
        for message in second
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    ]
    assert states[-1]['remaining_translation_m'] == [0.01, 0.0, 0.0]
    assert 'privileged-value-never-send' not in str(requests)
    assert Image.open(io.BytesIO(frames(requests[-1][0])[0].data)).size == (8, 4)
    events = [json.loads(line) for line in (tmp_path / 'transcript.jsonl').read_text().splitlines()]
    assert len([event for event in events if event['event'] == 'decision']) == 3
    with session(policy) as (fresh, rt):
        replies.append(finish('give_up'))
        complete(fresh, rt, observation())
        assert len(frames(requests[-1][0])) == 2
        assert 'Inspect the final image.' not in str(requests[-1][0])


def test_on_demand_pictures_reveal_only_requested_cameras(model):
    requests, replies = model
    picture = ModelResponse([ToolCallPart('take_pic', {'cameras': [keys.WRIST_IMAGE], 'note': 'Inspect wrist.'})])
    replies.extend([picture, picture, finish()])
    policy = LLMPolicy(Endpoint('test'), Motion(), images=Images.ON_DEMAND)
    with session(policy) as (active, rt):
        complete(active, rt, observation())
    assert [len(frames(messages)) for messages, _ in requests] == [0, 1, 1]
    assert 'already revealed' in str(requests[-1][0])
    assert 'take_pic' in [tool.name for tool in requests[0][1]]


@pytest.mark.parametrize(
    'bad',
    [
        move(0.2),
        ModelResponse([TextPart('I will move.')]),
        ModelResponse([ToolCallPart('move_to', '{broken')]),
        ModelResponse([ToolCallPart('done', {}), ToolCallPart('give_up', {})]),
        ModelResponse([ToolCallPart('run_code', {'code': 'print(1)'})]),
        ModelResponse([ToolCallPart('done', {'reason': 'x', 'hindsight': 'x'})], finish_reason='length'),
    ],
)
def test_invalid_reply_gets_correction_and_has_finite_retry_budget(model, bad):
    requests, replies = model
    replies.extend([bad, bad, bad])
    with session(LLMPolicy(Endpoint('test'), Motion())) as (active, rt):
        with pytest.raises(RuntimeError, match='3 consecutive invalid'):
            complete(active, rt, observation())
    assert len(requests) == 3
    assert 'Rejected' in str(requests[-1][0])


def test_call_budget_includes_picture_requests(model):
    requests, replies = model
    replies.append(ModelResponse([ToolCallPart('take_pic', {'cameras': [], 'note': 'Look.'})]))
    policy = LLMPolicy(Endpoint('test'), Motion(), images=Images.ON_DEMAND, max_calls=1)
    with session(policy) as (active, rt):
        assert complete(active, rt, observation()) == []
        assert active.stop_requested and active.meta['stop_reason'] == 'call_budget'
    assert len(requests) == 1


@pytest.mark.parametrize('cancel_before_answer', [True, False])
def test_fault_discards_delayed_answer_and_keeps_one_request_in_flight(model, tmp_path, cancel_before_answer):
    requests, replies = model
    entered, release = threading.Event(), threading.Event()

    def delayed():
        entered.set()
        assert release.wait(5)
        return move()

    replies.extend([delayed, finish()])
    policy = (StopOnFault() | ChunkedSchedule()).wrap(LLMPolicy(Endpoint('test'), Motion()))
    with session(policy, tmp_path) as (active, rt):
        assert active(observation(), 0) is None
        assert entered.wait(5)
        for tick in range(10):
            assert active(observation(), tick) is None
        if not cancel_before_answer:
            release.set()
            rt.wait(5)
        fault = observation() | {keys.ROBOT_STATUS: RobotStatus.ERROR}
        assert active(fault, 10) == []
        assert active(observation(), 11) is None
        release.set()
        rt.wait(5)
        if cancel_before_answer:
            assert active(observation(), 12) is None
        complete(active, rt, observation(20), 20)
        assert active.stop_requested
    assert len(requests) == 2
    assert 'Reassess' in str(requests[-1][0])
    events = [json.loads(line)['event'] for line in (tmp_path / 'transcript.jsonl').read_text().splitlines()]
    assert 'discarded' in events


@pytest.mark.parametrize('current_x,accepted', [(0.02, True), (-0.1, False)])
def test_delayed_motion_starts_at_delivery_pose_and_is_anchored_at_delivery(model, current_x, accepted):
    _, replies = model
    replies.append(move(0.04))
    policy = ChunkedSchedule().wrap(LLMPolicy(Endpoint('test'), Motion()))
    with session(policy) as (active, rt):
        assert active(observation(), 0) is None
        rt.wait(5)
        trajectory = active(observation(x=current_x), 5_000_000_000)
        if accepted:
            assert trajectory
            assert 5 < trajectory[0][keys.ACTION_TIMESTAMP] < trajectory[-1][keys.ACTION_TIMESTAMP]
            assert current_x < trajectory[0][keys.ROBOT_COMMAND].pose.translation[0] < 0.04
            assert trajectory[-1][keys.ROBOT_COMMAND].pose.translation[0] == pytest.approx(0.04)
            assert active(observation(), 5_100_000_000) is None
        else:
            assert trajectory == []


def test_config_builds_a_local_policy_with_scheduling(model):
    _, replies = model
    replies.append(finish())
    policy = llm(model='test')
    with session(policy) as (active, rt):
        complete(active, rt, observation())
        assert active.stop_requested
        assert active.meta['api'] == 'openai-responses'
