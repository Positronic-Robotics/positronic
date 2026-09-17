import io
import json
import threading
from contextlib import contextmanager

import numpy as np
import pytest
from PIL import Image
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)

from positronic import keys
from positronic.drivers.roboarm import RobotStatus
from positronic.eval import Task
from positronic.policy.executor import Executor
from positronic.policy.harness import Rollout
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

    def request(endpoint, messages, tools):
        requests.append((messages, tools))
        response = replies.pop(0)
        return response() if callable(response) else response

    monkeypatch.setattr(Endpoint, 'request', request)
    return requests, replies


@contextmanager
def session(policy):
    rt = Executor(policy.functions)
    active = policy.new_session(rt=rt)
    try:
        yield active, rt
    finally:
        active.cancel()
        rt.close()
        active.close()


def complete(active, rt, obs, time_ns=0):
    result = active(obs, time_ns)
    while result is None:
        assert rt.owes_an_answer
        rt.wait(5)
        assert not rt.in_flight
        result = active(obs, time_ns)
    return result


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


def test_history_keeps_two_observations_images_and_reports_actual_pose(model):
    requests, replies = model
    replies.extend([move(), move(), finish()])
    policy = LLMPolicy(Endpoint('test'), Motion(), image_size=8)
    with session(policy) as (active, rt):
        complete(active, rt, observation(1))
        first_meta = active.meta
        first_serialized = json.dumps(first_meta)
        complete(active, rt, observation(2))
        assert complete(active, rt, observation(3)) == []
        assert active.meta['stop_reason'] == 'done'
        assert active.meta['hindsight'] == 'Inspect the final image.'
        events = active.meta['transcript']
        assert json.dumps(first_meta) == first_serialized
        first_meta['transcript'][1]['position_m'][0] = 123
        assert active.meta['transcript'][1]['position_m'][0] == 0
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
    assert [event['obs_time_ns'] for event in events if event['event'] == 'observation'] == [1, 2, 3]
    assert [event['call'] for event in events if event['event'] == 'accepted'] == [1, 2, 3]
    assert len([event for event in events if event['event'] == 'instructions']) == 1
    assert 'privileged-value-never-send' not in json.dumps(events)
    with session(policy) as (fresh, rt):
        assert fresh.meta['transcript'] == []
        replies.append(finish('give_up'))
        complete(fresh, rt, observation())
        assert len(frames(requests[-1][0])) == 2
        assert 'Inspect the final image.' not in str(requests[-1][0])
        assert [e['call'] for e in fresh.meta['transcript'] if e['event'] == 'request'] == [1]


def test_on_demand_pictures_reveal_only_requested_cameras(model):
    requests, replies = model
    picture = ModelResponse([ToolCallPart('take_pic', {'cameras': [keys.WRIST_IMAGE], 'note': 'Inspect wrist.'})])
    replies.extend([picture, picture, finish()])
    policy = LLMPolicy(Endpoint('test'), Motion(), images=Images.ON_DEMAND)
    with session(policy) as (active, rt):
        complete(active, rt, observation())
        events = active.meta['transcript']
    assert [len(frames(messages)) for messages, _ in requests] == [0, 1, 1]
    assert 'already revealed' in str(requests[-1][0])
    assert 'take_pic' in [tool.name for tool in requests[0][1]]
    assert [e['cameras'] for e in events if e['event'] == 'request'] == [[], [keys.WRIST_IMAGE], [keys.WRIST_IMAGE]]
    assert len([e for e in events if e['event'] == 'observation']) == 1
    assert [e['call'] for e in events if e['event'] == 'rejected'] == [2]


@pytest.mark.parametrize(
    'reply',
    [
        ModelResponse([TextPart('I will move.')]),
        ModelResponse([ToolCallPart('take_pic', {'cameras': [], 'note': 'Look.'})]),
    ],
)
def test_follow_up_needs_another_session_call_and_uses_the_frozen_observation(model, reply):
    requests, replies = model
    replies.extend([reply, finish()])
    policy = LLMPolicy(Endpoint('test'), Motion(), images=Images.ON_DEMAND)
    with session(policy) as (active, rt):
        assert active(observation(1), 1) is None
        rt.wait(5)
        assert not rt.in_flight
        assert len(requests) == 1
        later = observation(2, x=0.1)
        later[keys.WRIST_IMAGE][:] = 255
        assert active(later, 2) is None
        rt.wait(5)
        assert not rt.in_flight
        assert len(requests) == 2
        assert active(later, 3) == []
        events = active.meta['transcript']
    assert [e['obs_time_ns'] for e in events if e['event'] == 'request'] == [1, 1]
    assert len([e for e in events if e['event'] == 'observation']) == 1
    if reply.tool_calls:
        pictures = frames(requests[1][0])
        assert len(pictures) == 2
        np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(pictures[0].data))), 0)


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
        assert active.meta['stop_reason'] == 'call_budget'
    assert len(requests) == 1


@pytest.mark.parametrize('ending', ['done', 'give_up', 'call_budget'])
def test_finished_session_stays_idle_after_cancellation_and_new_session_starts_fresh(model, ending):
    requests, replies = model
    replies.append(
        ModelResponse([ToolCallPart('take_pic', {'cameras': [], 'note': 'Look.'})])
        if ending == 'call_budget'
        else finish(ending)
    )
    policy = (StopOnFault() | ChunkedSchedule()).wrap(
        LLMPolicy(Endpoint('test'), Motion(), images=Images.ON_DEMAND, max_calls=1)
    )
    with session(policy) as (active, rt):
        assert complete(active, rt, observation()) == []
        meta = active.meta
        assert meta['stop_reason'] == ending
        for tick in range(1, 4):
            assert active(observation(tick), tick) == []
        active.cancel()
        assert active(observation(4) | {keys.ROBOT_STATUS: RobotStatus.ERROR}, 4) == []
        assert active(observation(5), 5) == []
        assert not rt.in_flight
        assert active.meta == meta
    assert len(requests) == 1
    replies.append(finish())
    with session(policy) as (fresh, rt):
        assert 'stop_reason' not in fresh.meta
        assert fresh.meta['transcript'] == []
        assert complete(fresh, rt, observation(6)) == []
        assert [e['call'] for e in fresh.meta['transcript'] if e['event'] == 'request'] == [1]
    assert len(requests) == 2


@pytest.mark.parametrize('cancel_before_answer', [True, False])
@pytest.mark.parametrize('follow_up', [False, True])
def test_fault_discards_delayed_answer_and_keeps_one_request_in_flight(model, cancel_before_answer, follow_up):
    requests, replies = model
    entered, release = threading.Event(), threading.Event()

    def delayed():
        entered.set()
        assert release.wait(5)
        return move()

    if follow_up:
        replies.append(ModelResponse([ToolCallPart('take_pic', {'cameras': [], 'note': 'Look.'})]))
    replies.extend([delayed, finish()])
    policy = (StopOnFault() | ChunkedSchedule()).wrap(LLMPolicy(Endpoint('test'), Motion(), images=Images.ON_DEMAND))
    with session(policy) as (active, rt):
        assert active(observation(), 0) is None
        if follow_up:
            rt.wait(5)
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
        assert active.meta['stop_reason'] == 'done'
        events = [event['event'] for event in active.meta['transcript']]
    assert len(requests) == 2 + int(follow_up)
    assert 'Reassess' in str(requests[-1][0])
    assert any(isinstance(part, SystemPromptPart) for message in requests[-1][0] for part in message.parts)
    assert 'discarded' in events


@pytest.mark.parametrize(
    'late_response',
    [
        move(),
        ModelResponse([TextPart('I will move.')]),
        ModelResponse([ToolCallPart('take_pic', {'cameras': [], 'note': 'Look.'})]),
    ],
)
def test_rollout_close_discards_late_reply_without_follow_up_requests(model, monkeypatch, late_response):
    requests, replies = model
    entered, release = threading.Event(), threading.Event()

    def delayed():
        entered.set()
        assert release.wait(5)
        return late_response

    replies.extend([delayed, finish()])
    policy = LLMPolicy(Endpoint('test'), Motion(), images=Images.ON_DEMAND)
    rollout = Rollout(Task(instruction_source='test', timeout_sec=None), policy, None)
    close_runtime = rollout.rt.close

    def release_then_close():
        release.set()
        close_runtime()

    monkeypatch.setattr(rollout.rt, 'close', release_then_close)
    try:
        assert rollout.session(observation(), 0) is None
        assert entered.wait(5)
    finally:
        rollout.close()
    assert len(requests) == 1
    events = [e['event'] for e in rollout.session.meta['transcript']]
    assert events.index('response') < events.index('discarded')
    assert 'accepted' not in events


def test_metadata_snapshot_excludes_response_arriving_after_cancellation(model):
    _, replies = model
    entered, release = threading.Event(), threading.Event()

    def delayed():
        entered.set()
        assert release.wait(5)
        return move()

    replies.append(delayed)
    with session(LLMPolicy(Endpoint('test'), Motion())) as (active, rt):
        assert active(observation(), 0) is None
        assert entered.wait(5)
        active.cancel()
        recorded_meta = active.meta
        serialized = json.dumps(recorded_meta)
        release.set()
        rt.wait(5)
        assert active(observation(), 1) is None
        assert json.dumps(recorded_meta) == serialized
        assert not any(e['event'] == 'response' for e in recorded_meta['transcript'])
        assert any(e['event'] == 'discarded' for e in active.meta['transcript'])


@pytest.mark.parametrize('bad', [move(0.2), ModelResponse([TextPart('I will move.')])])
def test_corrected_replies_are_preserved_in_static_transcript(model, bad):
    _, replies = model
    replies.extend([bad, finish()])
    with session(LLMPolicy(Endpoint('test'), Motion())) as (active, rt):
        complete(active, rt, observation(123))
        events = active.meta['transcript']
    assert [e['call'] for e in events if e['event'] == 'response'] == [1, 2]
    assert [e['call'] for e in events if e['event'] == 'rejected'] == [1]
    assert [e['call'] for e in events if e['event'] == 'accepted'] == [2]
    assert [e['obs_time_ns'] for e in events if e['event'] == 'request'] == [123, 123]
    response = next(e for e in events if e['event'] == 'response')
    if bad.tool_calls:
        assert response['tools'][0]['arguments']['x'] == 0.2
    else:
        assert response['text'] == ['I will move.']


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
        assert active.meta['stop_reason'] == 'done'
        assert active.meta['model'] == 'test:test'
