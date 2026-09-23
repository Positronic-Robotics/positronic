import io
import json
import threading
from collections.abc import Generator
from contextlib import contextmanager
from copy import deepcopy
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from pimm.world import VirtualClock
from positronic import keys
from positronic.drivers.roboarm import RobotStatus
from positronic.policy import keys as policy_keys
from positronic.policy.base import Obs, Step
from positronic.policy.executor import Executor, WaitStatus
from positronic.vendors.llm.client import Endpoint
from positronic.vendors.llm.motion import Motion
from positronic.vendors.llm.policy import Images, LLMPolicy, llm


def observation(x=0.0):
    return {
        keys.EE_POSE: np.array([x, 0, 0, 1, 0, 0, 0]),
        keys.GRIP: 0.0,
        keys.TASK: 'Move the cube.',
        keys.WRIST_IMAGE: np.zeros((10, 20, 3), dtype=np.uint8),
        keys.EXTERIOR_IMAGE: np.ones((10, 20, 3), dtype=np.uint8),
        keys.ROBOT_STATUS: RobotStatus.AVAILABLE,
        'sim_state': 'privileged-value-never-send',
    }


def move(x: float | str = 0.01):
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
def execution(policy):
    clock = VirtualClock()
    runtime = Executor(clock.now_ns, simulated=True, charge_inference_time=False)
    run = cast(Generator[Step, Obs, None], runtime.start(policy))
    try:
        yield run, runtime, clock
    finally:
        runtime.close()
        run.close()


def complete(run, runtime, clock, obs) -> Step:
    accepted = sum(e['event'] == 'accepted' for e in runtime.metadata['transcript'])
    for _ in range(1000):
        step = run.send(obs)
        if (
            'stop_reason' in runtime.metadata
            or sum(e['event'] == 'accepted' for e in runtime.metadata['transcript']) > accepted
        ):
            return step
        result = runtime.wait(5)
        assert result.status is not WaitStatus.TIMED_OUT
        if result.status is WaitStatus.CAN_ADVANCE:
            clock.advance_to_ns(max(clock.now_ns(), step.resume_at_ns))
    pytest.fail('The model did not reach a decision')


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
    with execution(policy) as (run, runtime, clock):
        for second in range(3):
            clock.advance_to_ns(second * 1_000_000_000)
            complete(run, runtime, clock, observation())
        assert runtime.metadata['stop_reason'] == 'done'
        assert runtime.metadata['hindsight'] == 'Inspect the final image.'
        events = runtime.metadata['transcript']
    assert [len(frames(messages)) for messages, _ in requests] == [2, 4, 4]
    states = [e for e in events if e['event'] == 'observation']
    assert states[1]['remaining_translation_m'] == [0.01, 0.0, 0.0]
    assert 'privileged-value-never-send' not in str(requests)
    assert Image.open(io.BytesIO(frames(requests[-1][0])[0].data)).size == (8, 4)
    assert [e[policy_keys.OBS_TIME_NS] for e in states] == [0, 1_000_000_000, 2_000_000_000]
    assert [e['call'] for e in events if e['event'] == 'accepted'] == [1, 2, 3]
    assert len([e for e in events if e['event'] == 'instructions']) == 1
    assert 'privileged-value-never-send' not in json.dumps(events)
    with execution(policy) as (fresh, runtime, clock):
        assert runtime.metadata['transcript'] == []
        replies.append(finish('give_up'))
        complete(fresh, runtime, clock, observation())
        assert len(frames(requests[-1][0])) == 2
        assert 'Inspect the final image.' not in str(requests[-1][0])
        assert [e['call'] for e in runtime.metadata['transcript'] if e['event'] == 'request'] == [1]


def test_on_demand_pictures_reveal_only_requested_cameras(model):
    requests, replies = model
    picture = ModelResponse([ToolCallPart('take_pic', {'cameras': [keys.WRIST_IMAGE], 'note': 'Inspect wrist.'})])
    replies.extend([picture, picture, finish()])
    with execution(LLMPolicy(Endpoint('test'), Motion(), images=Images.ON_DEMAND)) as (run, runtime, clock):
        complete(run, runtime, clock, observation())
        events = runtime.metadata['transcript']
    assert [len(frames(messages)) for messages, _ in requests] == [0, 1, 1]
    assert 'already revealed' in str(requests[-1][0])
    assert 'take_pic' in [tool.name for tool in requests[0][1]]
    assert [e['cameras'] for e in events if e['event'] == 'request'] == [[], [keys.WRIST_IMAGE], [keys.WRIST_IMAGE]]
    assert len([e for e in events if e['event'] == 'observation']) == 1
    assert [e['call'] for e in events if e['event'] == 'rejected'] == [2]


@pytest.mark.parametrize('images', list(Images))
@pytest.mark.parametrize('image_horizon', [1, 2])
def test_retained_history_prunes_images_by_observation(model, images, image_horizon):
    requests, replies = model
    policy = LLMPolicy(Endpoint('test'), Motion(), images=images, image_horizon=image_horizon)
    pictured, motion_replies = [], []
    with execution(policy) as (run, runtime, clock):
        for second in range(1, 5):
            time_ns = second * 1_000_000_000
            clock.advance_to_ns(time_ns)
            if images is Images.ALWAYS or second != 3:
                pictured.append(time_ns)
            if images is Images.ON_DEMAND and second != 3:
                replies.extend(
                    ModelResponse([ToolCallPart('take_pic', {'cameras': [camera], 'note': 'Inspect camera.'})])
                    for camera in policy.camera_keys
                )
            reply = move()
            reply.parts = [ThinkingPart('Check the scene.', signature='native-signature'), *reply.parts]
            reply.provider_response_id = f'reply-{second}'
            motion_replies.append(reply)
            replies.append(reply)
            complete(run, runtime, clock, observation())
            messages = requests[-1][0]
            assert len(frames(messages)) == 2 * min(len(pictured), image_horizon)
        messages = requests[-1][0]
        retained = [
            message.metadata[policy_keys.OBS_TIME_NS]
            for message in messages
            if isinstance(message, ModelRequest) and message.metadata is not None and frames([message])
        ]
        assert set(retained) == set(pictured[-image_horizon:])
        assert '[older camera frame omitted]' in str(messages)
        assert all(reply in messages for reply in motion_replies[:-1])
    first_pictures = next(messages for messages, _ in requests if frames(messages))
    assert len(frames(first_pictures)) == (2 if images is Images.ALWAYS else 1)


@pytest.mark.parametrize(
    'reply',
    [
        ModelResponse([TextPart('I will move.')]),
        ModelResponse([ToolCallPart('take_pic', {'cameras': [], 'note': 'Look.'})]),
    ],
)
def test_follow_up_needs_another_tick_and_uses_the_frozen_observation(model, reply):
    requests, replies = model
    replies.extend([reply, finish()])
    with execution(LLMPolicy(Endpoint('test'), Motion(), images=Images.ON_DEMAND)) as (run, runtime, clock):
        clock.advance_to_ns(1)
        assert run.send(observation()) == Step({}, 1)
        assert runtime.wait(5).status is WaitStatus.ANSWERS_READY
        assert len(requests) == 1
        later = observation(x=0.1)
        later[keys.WRIST_IMAGE][:] = 255
        clock.advance_to_ns(2)
        assert run.send(later) == Step({}, 2)
        assert runtime.wait(5).status is WaitStatus.ANSWERS_READY
        assert len(requests) == 2
        assert not run.send(later).commands
        events = runtime.metadata['transcript']
    assert [e[policy_keys.OBS_TIME_NS] for e in events if e['event'] == 'request'] == [1, 1]
    assert len([e for e in events if e['event'] == 'observation']) == 1
    if reply.tool_calls:
        pictures = frames(requests[1][0])
        assert len(pictures) == 2
        np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(pictures[0].data))), 0)


@pytest.mark.parametrize(
    'bad',
    [
        move('invalid'),
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
    with execution(LLMPolicy(Endpoint('test'), Motion())) as (run, runtime, clock):
        with pytest.raises(RuntimeError, match='3 consecutive invalid'):
            complete(run, runtime, clock, observation())
    assert len(requests) == 3
    assert 'Rejected' in str(requests[-1][0])


@pytest.mark.parametrize('ending', ['done', 'give_up', 'call_budget'])
def test_finished_run_stays_idle_and_new_run_starts_fresh(model, ending):
    requests, replies = model
    replies.append(
        ModelResponse([ToolCallPart('take_pic', {'cameras': [], 'note': 'Look.'})])
        if ending == 'call_budget'
        else finish(ending)
    )
    policy = llm(model='test', images='on_demand', max_calls=1)
    with execution(policy) as (run, runtime, clock):
        assert not complete(run, runtime, clock, observation()).commands
        meta = deepcopy(runtime.metadata)
        assert meta['stop_reason'] == ending
        for status in (RobotStatus.AVAILABLE, RobotStatus.ERROR, RobotStatus.AVAILABLE):
            clock.advance_to_ns(clock.now_ns() + 1_000_000_000)
            assert not run.send(observation() | {keys.ROBOT_STATUS: status}).commands
        assert runtime.metadata == meta
    assert len(requests) == 1
    replies.append(finish())
    with execution(policy) as (fresh, runtime, clock):
        assert 'stop_reason' not in runtime.metadata
        assert runtime.metadata['transcript'] == []
        complete(fresh, runtime, clock, observation())
        assert [e['call'] for e in runtime.metadata['transcript'] if e['event'] == 'request'] == [1]
    assert len(requests) == 2


@pytest.mark.parametrize('status', [RobotStatus.BUSY, RobotStatus.ERROR])
def test_fault_pauses_commands_without_discarding_pending_reply(model, status):
    requests, replies = model
    entered, release = threading.Event(), threading.Event()

    def delayed():
        entered.set()
        assert release.wait(5)
        return move(0.04)

    replies.append(delayed)
    with execution(llm(model='test')) as (run, runtime, clock):
        assert not run.send(observation()).commands
        assert entered.wait(5)
        try:
            fault = observation() | {keys.ROBOT_STATUS: status}
            assert not run.send(fault).commands
        finally:
            release.set()
        runtime.wait(5)
        assert not run.send(fault).commands
        first = run.send(observation(x=0.02))
        assert first == Step({}, 40_000_000)
        clock.advance_to_ns(first.resume_at_ns)
        command = run.send(observation(x=0.02)).commands[keys.ROBOT_COMMAND]
        assert 0.02 < command.pose.translation[0] <= 0.04
        accepted = [e for e in runtime.metadata['transcript'] if e['event'] == 'accepted']
        assert accepted[0]['target']['x'] == 0.04
    assert len(requests) == 1


@pytest.mark.parametrize('error', [TimeoutError('API timed out'), RuntimeError('API unavailable')])
def test_active_request_failure_propagates(model, error):
    requests, replies = model
    replies.append(Mock(side_effect=error))
    with execution(LLMPolicy(Endpoint('test'), Motion())) as (run, runtime, clock):
        with pytest.raises(type(error), match=str(error)):
            complete(run, runtime, clock, observation())
    assert len(requests) == 1


@pytest.mark.parametrize(
    'late_response',
    [
        move(),
        ModelResponse([TextPart('I will move.')]),
        ModelResponse([ToolCallPart('take_pic', {'cameras': [], 'note': 'Look.'})]),
    ],
)
def test_close_drains_request_without_processing_reply(model, late_response):
    requests, replies = model
    entered, release = threading.Event(), threading.Event()

    def delayed():
        entered.set()
        assert release.wait(5)
        return late_response

    replies.append(delayed)
    with execution(LLMPolicy(Endpoint('test'), Motion(), images=Images.ON_DEMAND)) as (run, runtime, clock):
        try:
            assert not run.send(observation()).commands
            assert entered.wait(5)
            snapshot = deepcopy(runtime.metadata)
        finally:
            release.set()
    assert runtime.metadata == snapshot
    assert len(requests) == 1
    assert [e['event'] for e in snapshot['transcript']] == ['instructions', 'observation', 'request']


@pytest.mark.parametrize('bad', [move('invalid'), ModelResponse([TextPart('I will move.')])])
def test_corrected_replies_are_preserved_in_transcript(model, bad):
    _, replies = model
    replies.extend([bad, finish()])
    with execution(LLMPolicy(Endpoint('test'), Motion())) as (run, runtime, clock):
        clock.advance_to_ns(123)
        complete(run, runtime, clock, observation())
        events = runtime.metadata['transcript']
    assert [e['call'] for e in events if e['event'] == 'response'] == [1, 2]
    assert [e['call'] for e in events if e['event'] == 'rejected'] == [1]
    assert [e['call'] for e in events if e['event'] == 'accepted'] == [2]
    assert [e[policy_keys.OBS_TIME_NS] for e in events if e['event'] == 'request'] == [123, 123]


@pytest.mark.parametrize('requested_x,current_x,expected_x', [(0.2, 0, 0.05), (0.04, -0.1, -0.05), (0.2, 0.18, 0.2)])
def test_motion_uses_delivery_pose_and_waits_for_final_period_before_observing(
    model, requested_x, current_x, expected_x
):
    requests, replies = model
    replies.extend([move(requested_x), finish()])
    with execution(LLMPolicy(Endpoint('test'), Motion())) as (run, runtime, clock):
        assert run.send(observation()) == Step({}, 0)
        runtime.wait(5)
        clock.advance_to_ns(5_000_000_000)
        step = run.send(observation(x=current_x))
        assert step == Step({}, 5_040_000_000)
        accepted = [e for e in runtime.metadata['transcript'] if e['event'] == 'accepted'][0]
        assert accepted['target']['x'] == pytest.approx(expected_x)
        assert accepted['clamped'] is (requested_x != expected_x)
        end_ns = 5_000_000_000 + round(accepted['duration_s'] * 1e9)
        clock.advance_to_ns(end_ns - 40_000_000)
        final = run.send(observation(x=current_x))
        assert final.commands[keys.ROBOT_COMMAND].pose.translation[0] == pytest.approx(expected_x)
        assert final.resume_at_ns == end_ns
        assert len(requests) == 1
        clock.advance_to_ns(end_ns - 1)
        assert run.send(observation(x=expected_x)) == Step({}, end_ns)
        assert len(requests) == 1
        clock.advance_to_ns(end_ns)
        complete(run, runtime, clock, observation(x=expected_x))
        events = runtime.metadata['transcript']
    assert len(requests) == 2
    assert not any(e['event'] == 'rejected' for e in events)
    result = next(
        part
        for message in requests[-1][0]
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    )
    content = result.model_response_object()
    assert content['target'] == accepted['target']
    assert content['duration_s'] == accepted['duration_s']
    state = [e for e in events if e['event'] == 'observation'][-1]
    assert state['previous_target'] == accepted['target']
    assert state['remaining_translation_m'] == pytest.approx([0, 0, 0])


def test_late_resume_emits_final_command_before_requesting_next_decision(model):
    requests, replies = model
    replies.extend([move(), finish()])
    with execution(LLMPolicy(Endpoint('test'), Motion())) as (run, runtime, clock):
        complete(run, runtime, clock, observation())
        clock.advance_to_ns(10_000_000_000)
        step = run.send(observation())
        assert step.commands[keys.ROBOT_COMMAND].pose.translation[0] == pytest.approx(0.01)
        assert len(requests) == 1
        complete(run, runtime, clock, observation(x=0.01))
        assert len(requests) == 2
