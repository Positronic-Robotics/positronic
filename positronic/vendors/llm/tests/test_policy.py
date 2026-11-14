import io
import json
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from itertools import pairwise
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from pydantic_ai.messages import (
    BinaryContent,
    CachePoint,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.settings import ModelSettings

from pimm.world import VirtualClock
from positronic import geom, keys
from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm.yam.kinematics import YamKinematics
from positronic.policy import keys as policy_keys
from positronic.policy.base import Obs, PolicyEnd, Step
from positronic.policy.executor import Executor, WaitStatus
from positronic.vendors.llm.client import Endpoint
from positronic.vendors.llm.motion import Motion, MoveArmsPath, MovePath
from positronic.vendors.llm.policy import Conversation, Images, LLMPolicy, Tool, llm, llm_bimanual
from positronic.vendors.llm.reach import Box, YamReach


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


def move(*xs: float | str):
    waypoints = [{'x': x, 'y': 0, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': 0, 'gripper': 0} for x in xs or (0.01,)]
    return ModelResponse([
        ToolCallPart(Tool.MOVE_PATH.value, {'waypoints': waypoints, 'note': 'Approach the cube.'}, tool_call_id='move')
    ])


def finish():
    return ModelResponse([
        ToolCallPart(
            Tool.DONE.value,
            {'reason': 'Attempt ended.', 'hindsight': 'Inspect the final image.'},
            tool_call_id='finish',
        )
    ])


@pytest.fixture
def model(monkeypatch):
    requests, replies = [], []

    def request(endpoint, runner, model, messages, tools):
        requests.append((messages, tools))
        response = replies.pop(0)
        return response() if callable(response) else response

    monkeypatch.setattr(Endpoint, 'request', request)
    return requests, replies


@contextmanager
def execution(policy):
    clock = VirtualClock()
    runtime = Executor(clock.now_ns, simulated=True, charge_inference_time=False)
    run = cast(Generator[Step, Obs, PolicyEnd], runtime.start(policy))
    try:
        yield run, runtime, clock
    finally:
        runtime.close()
        run.close()


def complete(run, runtime, clock, obs) -> Step | PolicyEnd:
    """The step after the model's next decision, or the ``PolicyEnd`` that the run returns."""
    accepted = sum(e['event'] == 'accepted' for e in runtime.metadata_snapshot()[policy_keys.TRANSCRIPT])
    for _ in range(1000):
        try:
            step = run.send(obs)
        except StopIteration as end:
            return end.value
        metadata = runtime.metadata_snapshot()
        if (
            policy_keys.STOP_REASON in metadata
            or sum(e['event'] == 'accepted' for e in metadata[policy_keys.TRANSCRIPT]) > accepted
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


def test_history_keeps_the_frames_of_the_last_observations_and_reports_actual_pose(model):
    requests, replies = model
    replies.extend([move(), move(), move(), finish()])
    policy = LLMPolicy(Endpoint('test'), Motion(), image_size=8)
    with execution(policy) as (run, runtime, clock):
        decisions = []
        for second in range(4):
            clock.advance_to_ns(second * 1_000_000_000)
            decisions.append(complete(run, runtime, clock, observation()))
        assert decisions[-1] is PolicyEnd.COMPLETE
        assert runtime.metadata_snapshot()[policy_keys.STOP_REASON] == 'done'
        assert runtime.metadata_snapshot()[policy_keys.HINDSIGHT] == 'Inspect the final image.'
        events = runtime.metadata_snapshot()[policy_keys.TRANSCRIPT]
    assert [len(frames(messages)) for messages, _ in requests] == [2, 4, 6, 6]
    assert str(requests[-1][0]).count('[older camera frame omitted]') == 2
    states = [e for e in events if e['event'] == 'observation']
    assert states[1]['remaining_translation_m'] == [0.01, 0.0, 0.0]
    assert 'privileged-value-never-send' not in str(requests)
    assert Image.open(io.BytesIO(frames(requests[-1][0])[0].data)).size == (8, 4)
    assert [e[policy_keys.OBS_TIME_NS] for e in states] == [0, 1_000_000_000, 2_000_000_000, 3_000_000_000]
    assert [e['call'] for e in events if e['event'] == 'accepted'] == [1, 2, 3, 4]
    assert len([e for e in events if e['event'] == 'instructions']) == 1
    assert 'privileged-value-never-send' not in json.dumps(events)
    with execution(policy) as (fresh, runtime, clock):
        assert runtime.metadata_snapshot()[policy_keys.TRANSCRIPT] == []
        replies.append(finish())
        assert complete(fresh, runtime, clock, observation()) is PolicyEnd.COMPLETE
        assert len(frames(requests[-1][0])) == 2
        assert 'Inspect the final image.' not in str(requests[-1][0])
        assert [e['call'] for e in runtime.metadata_snapshot()[policy_keys.TRANSCRIPT] if e['event'] == 'request'] == [
            1
        ]


def test_image_horizon_must_be_positive():
    with pytest.raises(ValueError, match='image horizon'):
        LLMPolicy(Endpoint('test'), Motion(), image_horizon=0)


def test_cache_ttl_must_be_one_of_the_ttls_that_providers_accept():
    with pytest.raises(ValueError, match='cache_ttl'):
        LLMPolicy(Endpoint('test'), Motion(), cache_ttl='2h')


@pytest.mark.parametrize(
    'setting',
    ['anthropic_cache_tool_definitions', 'anthropic_cache', 'bedrock_cache_messages', 'openrouter_cache_instructions'],
)
def test_policy_refuses_a_provider_cache_setting(setting):
    with pytest.raises(ValueError, match=f'provider caching.*{setting}'):
        LLMPolicy(Endpoint('test', settings=cast(ModelSettings, {setting: True})), Motion())


def test_on_demand_pictures_reveal_only_requested_cameras(model):
    requests, replies = model
    picture = ModelResponse([
        ToolCallPart(Tool.REVEAL_FRAMES.value, {'cameras': [keys.WRIST_IMAGE], 'note': 'Inspect wrist.'})
    ])
    replies.extend([picture, picture, finish()])
    with execution(LLMPolicy(Endpoint('test'), Motion(), images=Images.ON_DEMAND)) as (run, runtime, clock):
        assert complete(run, runtime, clock, observation()) is PolicyEnd.COMPLETE
        events = runtime.metadata_snapshot()[policy_keys.TRANSCRIPT]
    assert [len(frames(messages)) for messages, _ in requests] == [0, 1, 1]
    assert 'already revealed' in str(requests[-1][0])
    assert Tool.REVEAL_FRAMES.value in [tool.name for tool in requests[0][1]]
    assert [e['cameras'] for e in events if e['event'] == 'request'] == [[], [keys.WRIST_IMAGE], [keys.WRIST_IMAGE]]
    assert len([e for e in events if e['event'] == 'observation']) == 1
    assert [e['call'] for e in events if e['event'] == 'rejected'] == [2]


def without_cache_points(messages):
    """The indices of the messages that end with a cache point, and the request as the stored history holds it."""
    marked, stored = [], []
    for index, message in enumerate(messages):
        *parts, part = message.parts
        if isinstance(part, UserPromptPart) and not isinstance(part.content, str) and part.content[-1] == CachePoint():
            marked.append(index)
            *content, _ = part.content
            text = content[0] if len(content) == 1 and isinstance(content[0], str) else content
            message = replace(message, parts=[*parts, replace(part, content=text)])
        stored.append(message)
    assert 'CachePoint' not in str(stored)
    return marked, stored


@pytest.mark.parametrize('images', list(Images))
def test_each_request_keeps_the_previous_request_up_to_its_newest_stub(model, images):
    requests, replies = model
    policy = LLMPolicy(Endpoint('test'), Motion(), images=images, image_horizon=2)
    pictured, motion_replies = [], []
    with execution(policy) as (run, runtime, clock):
        for second in range(1, 7):
            clock.advance_to_ns(second * 1_000_000_000)
            pictured.append(images is Images.ALWAYS or second != 3)
            if images is Images.ON_DEMAND and second != 3:
                replies.extend(
                    ModelResponse([
                        ToolCallPart(Tool.REVEAL_FRAMES.value, {'cameras': [camera], 'note': 'Inspect camera.'})
                    ])
                    for camera in policy.camera_keys
                )
            reply = move()
            reply.parts = [ThinkingPart('Check the scene.', signature='native-signature'), *reply.parts]
            reply.provider_response_id = f'reply-{second}'
            motion_replies.append(reply)
            replies.append(reply)
            complete(run, runtime, clock, observation())
            assert len(frames(requests[-1][0])) == 2 * sum(pictured[-2:])
        assert all(reply in requests[-1][0] for reply in motion_replies[:-1])
    sent = [without_cache_points(messages) for messages, _ in requests]
    assert all(len(marked) <= 2 and marked[-1] == len(stored) - 1 for marked, stored in sent)
    for (marked, earlier), (_, later) in pairwise(sent):
        anchor = marked[0] if len(marked) == 2 else -1
        pairs = zip(earlier, later, strict=False)
        changed = next((i for i, (old, new) in enumerate(pairs) if old != new), len(earlier))
        assert changed > anchor
        assert changed == len(earlier) or (frames([earlier[changed]]) and not frames([later[changed]]))
    assert any(len(marked) == 2 for marked, _ in sent)


@pytest.mark.parametrize(
    'reply',
    [
        ModelResponse([TextPart('I will move.')]),
        ModelResponse([ToolCallPart(Tool.REVEAL_FRAMES.value, {'cameras': [], 'note': 'Look.'})]),
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
        with pytest.raises(StopIteration) as end:
            run.send(later)
        assert end.value.value is PolicyEnd.COMPLETE
        events = runtime.metadata_snapshot()[policy_keys.TRANSCRIPT]
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
        ModelResponse([ToolCallPart(Tool.MOVE_PATH.value, '{broken')]),
        move(*[0.01] * 6),
        ModelResponse([ToolCallPart(Tool.DONE.value, {}), ToolCallPart(Tool.DONE.value, {})]),
        ModelResponse([ToolCallPart('give_up', {'reason': 'x', 'hindsight': 'x'})]),
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


@pytest.mark.parametrize('ending,end', [('done', PolicyEnd.COMPLETE), ('call_budget', PolicyEnd.GIVE_UP)])
def test_a_finished_run_returns_its_end_and_a_new_run_starts_fresh(model, ending, end):
    requests, replies = model
    replies.append(
        ModelResponse([ToolCallPart(Tool.REVEAL_FRAMES.value, {'cameras': [], 'note': 'Look.'})])
        if ending == 'call_budget'
        else finish()
    )
    policy = llm(model='test', images='on_demand', max_calls=1)
    with execution(policy) as (run, runtime, clock):
        assert complete(run, runtime, clock, observation()) is end
        assert runtime.metadata_snapshot()[policy_keys.STOP_REASON] == ending
    assert len(requests) == 1
    replies.append(finish())
    with execution(policy) as (fresh, runtime, clock):
        assert policy_keys.STOP_REASON not in runtime.metadata_snapshot()
        assert runtime.metadata_snapshot()[policy_keys.TRANSCRIPT] == []
        assert complete(fresh, runtime, clock, observation()) is PolicyEnd.COMPLETE
        assert [e['call'] for e in runtime.metadata_snapshot()[policy_keys.TRANSCRIPT] if e['event'] == 'request'] == [
            1
        ]
    assert len(requests) == 2


@pytest.mark.parametrize('status', [RobotStatus.BUSY, RobotStatus.ERROR])
def test_unavailable_arm_does_not_pause_inference_or_commands(model, status):
    requests, replies = model
    entered, release = threading.Event(), threading.Event()

    def delayed():
        entered.set()
        assert release.wait(5)
        return move(0.04)

    replies.append(delayed)
    with execution(llm(model='test')) as (run, runtime, clock):
        obs = observation(x=0.02) | {keys.ROBOT_STATUS: status}
        assert not run.send(obs).commands
        try:
            assert entered.wait(5)
            assert not run.send(obs).commands
        finally:
            release.set()
        runtime.wait(5)
        first = run.send(obs)
        assert first == Step({}, 40_000_000)
        clock.advance_to_ns(first.resume_at_ns)
        command = run.send(obs).commands[keys.ROBOT_COMMAND]
        assert 0.02 < command.pose.translation[0] <= 0.04
        accepted = [e for e in runtime.metadata_snapshot()[policy_keys.TRANSCRIPT] if e['event'] == 'accepted']
        assert accepted[0]['waypoints'][-1]['x'] == 0.04
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
        ModelResponse([ToolCallPart(Tool.REVEAL_FRAMES.value, {'cameras': [], 'note': 'Look.'})]),
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
            snapshot = runtime.metadata_snapshot()
        finally:
            release.set()
    assert runtime.metadata_snapshot() == snapshot
    assert len(requests) == 1
    assert [e['event'] for e in snapshot[policy_keys.TRANSCRIPT]] == ['instructions', 'observation', 'request']


@pytest.mark.parametrize('bad', [move('invalid'), ModelResponse([TextPart('I will move.')])])
def test_corrected_replies_are_preserved_in_transcript(model, bad):
    _, replies = model
    replies.extend([bad, finish()])
    with execution(LLMPolicy(Endpoint('test'), Motion())) as (run, runtime, clock):
        clock.advance_to_ns(123)
        assert complete(run, runtime, clock, observation()) is PolicyEnd.COMPLETE
        events = runtime.metadata_snapshot()[policy_keys.TRANSCRIPT]
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
    with execution(LLMPolicy(Endpoint('test'), Motion(max_translation=0.05))) as (run, runtime, clock):
        assert run.send(observation()) == Step({}, 0)
        runtime.wait(5)
        clock.advance_to_ns(5_000_000_000)
        step = run.send(observation(x=current_x))
        assert step == Step({}, 5_040_000_000)
        accepted = [e for e in runtime.metadata_snapshot()[policy_keys.TRANSCRIPT] if e['event'] == 'accepted'][0]
        assert accepted['waypoints'][-1]['x'] == pytest.approx(expected_x)
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
        assert complete(run, runtime, clock, observation(x=expected_x)) is PolicyEnd.COMPLETE
        events = runtime.metadata_snapshot()[policy_keys.TRANSCRIPT]
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
    assert content['waypoints'] == accepted['waypoints']
    assert content['duration_s'] == accepted['duration_s']
    state = [e for e in events if e['event'] == 'observation'][-1]
    assert state['previous_target'] == accepted['waypoints'][-1]
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
        assert complete(run, runtime, clock, observation(x=0.01)) is PolicyEnd.COMPLETE
        assert len(requests) == 2


def test_a_late_step_does_not_combine_commands_past_a_waypoint(model):
    requests, replies = model
    replies.extend([move(0.04, 0.0), finish()])
    with execution(LLMPolicy(Endpoint('test'), Motion(linear_speed=0.05))) as (run, runtime, clock):
        complete(run, runtime, clock, observation())
        clock.advance_to_ns(10_000_000_000)
        step = run.send(observation())
        assert step.commands[keys.ROBOT_COMMAND].pose.translation[0] == pytest.approx(0.04)
        assert step.resume_at_ns == 10_040_000_000
        clock.advance_to_ns(20_000_000_000)
        step = run.send(observation())
        assert step.commands[keys.ROBOT_COMMAND].pose.translation[0] == pytest.approx(0)
        assert len(requests) == 1


def test_a_gripper_hold_after_a_late_step_lasts_its_full_duration(model):
    requests, replies = model
    waypoints = [{'x': x, 'y': 0, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': 0, 'gripper': 1} for x in (0.04, 0.0)]
    replies.extend([
        ModelResponse([
            ToolCallPart(Tool.MOVE_PATH.value, {'waypoints': waypoints, 'note': 'Grasp.'}, tool_call_id='m')
        ]),
        finish(),
    ])
    motion = Motion(linear_speed=0.05)
    with execution(LLMPolicy(Endpoint('test'), motion)) as (run, runtime, clock):
        complete(run, runtime, clock, observation())
        clock.advance_to_ns(10_000_000_000)
        assert run.send(observation()).commands[keys.TARGET_GRIP] == 1
        clock.advance_to_ns(10_040_000_000)
        assert run.send(observation()).commands[keys.ROBOT_COMMAND].pose.translation[0] == pytest.approx(0.04)
        clock.advance_to_ns(10_040_000_000 + round(motion.grip_hold * 1e9 / 2))
        step = run.send(observation())
        assert step.commands[keys.ROBOT_COMMAND].pose.translation[0] == pytest.approx(0.04)
        assert len(requests) == 1


def test_a_path_plays_every_waypoint_before_the_next_decision(model):
    requests, replies = model
    replies.extend([move(0.04, 0.0), finish()])
    motion = Motion(linear_speed=0.05)
    with execution(LLMPolicy(Endpoint('test'), motion)) as (run, runtime, clock):
        step = complete(run, runtime, clock, observation())
        xs = []
        while step.commands or len(requests) == 1:
            xs.extend([step.commands[keys.ROBOT_COMMAND].pose.translation[0]] if step.commands else [])
            clock.advance_to_ns(step.resume_at_ns)
            step = run.send(observation())
            assert runtime.wait(5).status is not WaitStatus.TIMED_OUT
        assert len(requests) == 2
        accepted = [e for e in runtime.metadata_snapshot()[policy_keys.TRANSCRIPT] if e['event'] == 'accepted'][0]
    assert max(xs) == pytest.approx(0.04)
    assert xs[-1] == pytest.approx(0)
    assert [waypoint['x'] for waypoint in accepted['waypoints']] == [0.04, 0.0]
    assert accepted['duration_s'] == pytest.approx(0.08 / motion.linear_speed + 1 / motion.fps)


def bimanual_observation(left_x=0.0, right_x=0.0):
    poses = {keys.LEFT_ARM: [left_x, 0.3, 0.1, 1, 0, 0, 0], keys.RIGHT_ARM: [right_x, -0.3, 0.1, 1, 0, 0, 0]}
    return {
        **{f'{keys.arm_channel(keys.ROBOT_STATE, arm)}{keys.EE_POSE_SUFFIX}': np.array(p) for arm, p in poses.items()},
        **{keys.arm_channel(keys.GRIP, arm): 0.0 for arm in keys.BIMANUAL_ARMS},
        keys.TASK: 'Pass the cube.',
        **{
            name: np.zeros((10, 20, 3), dtype=np.uint8)
            for name in (keys.EXTERIOR_IMAGE, keys.WRIST_LEFT_IMAGE, keys.WRIST_RIGHT_IMAGE)
        },
    }


def move_arms(*waypoints: dict[str, tuple[float, ...]], gripper=0):
    """A path of hand positions (x, y) at z 0.1, or (x, y, z)."""
    path = [
        {
            arm: {'x': x, 'y': y, 'z': z, 'roll': 0, 'pitch': 0, 'yaw': 0, 'gripper': gripper}
            for arm, position in w.items()
            for x, y, z in [(*position, 0.1)[:3]]
        }
        for w in waypoints
    ]
    return ModelResponse([ToolCallPart(Tool.MOVE_PATH.value, {'waypoints': path, 'note': 'Reach.'}, tool_call_id='m')])


def decide(conversation: Conversation, obs, response):
    conversation.request(obs, 0)
    decision = conversation.respond(response)
    assert decision is not None
    return conversation.accept(*decision, obs, 0)


def right_hand(parts) -> list[np.ndarray]:
    channel = keys.arm_channel(keys.ROBOT_COMMAND, keys.RIGHT_ARM)
    return [command[channel].pose.translation for part in parts for command in part]


def test_an_arm_that_a_waypoint_leaves_out_holds_its_previous_target_or_else_its_measured_pose():
    metadata = {}
    conversation = Conversation(llm_bimanual(model='test', image_size=8), metadata.__setitem__)
    parts = decide(conversation, bimanual_observation(right_x=0.01), move_arms({'left': (0.05, 0.3)}))
    np.testing.assert_allclose(right_hand(parts), [[0.01, -0.3, 0.1]] * len(right_hand(parts)))
    decide(conversation, bimanual_observation(left_x=0.05), move_arms({'left': (0.05, 0.3), 'right': (0.02, -0.3)}))
    parts = decide(conversation, bimanual_observation(left_x=0.05, right_x=0.015), move_arms({'left': (0.08, 0.3)}))
    np.testing.assert_allclose(right_hand(parts)[-1], [0.02, -0.3, 0.1])
    observed = [e for e in metadata[policy_keys.TRANSCRIPT] if e['event'] == 'observation'][-1]
    assert list(observed['arms']) == [keys.LEFT_ARM, keys.RIGHT_ARM]
    assert observed['arms']['right']['position_m'] == pytest.approx([0.015, -0.3, 0.1])
    assert observed['arms']['right']['previous_target']['x'] == pytest.approx(0.02)
    assert observed['cameras'] == [keys.EXTERIOR_IMAGE, keys.WRIST_LEFT_IMAGE, keys.WRIST_RIGHT_IMAGE]
    accepted = [e for e in metadata[policy_keys.TRANSCRIPT] if e['event'] == 'accepted'][-1]
    assert [(w['left']['x'], w['right']['x']) for w in accepted['waypoints']] == [(0.08, 0.02)]


def test_a_path_with_an_unknown_arm_is_rejected():
    metadata = {}
    conversation = Conversation(llm_bimanual(model='test', image_size=8), metadata.__setitem__)
    conversation.request(bimanual_observation(), 0)
    assert conversation.respond(move_arms({'left': (0.0, 0.3), 'middle': (0.0, 0.0)})) is None
    (rejected,) = [e for e in metadata[policy_keys.TRANSCRIPT] if e['event'] == 'rejected']
    assert "Unknown arms ['middle']" in rejected['reason']


def hands_at(obs, positions: dict[str, tuple[float, ...]]):
    """``obs`` with the measured hands at (x, y), or at (x, y, z)."""
    for arm, position in positions.items():
        obs[f'{keys.arm_channel(keys.ROBOT_STATE, arm)}{keys.EE_POSE_SUFFIX}'][: len(position)] = position
    return obs


@pytest.mark.parametrize(
    'start,path,accepted',
    [
        ({'left': (0.0, 0.3), 'right': (0.0, -0.3)}, {'left': (0.0, 0.05), 'right': (0.0, -0.03)}, False),
        ({'left': (0.0, 0.3), 'right': (0.0, -0.3)}, {'left': (0.0, 0.15), 'right': (0.0, -0.15)}, True),
        ({'left': (0.0, 0.04), 'right': (0.0, -0.04)}, {'left': (0.0, 0.3), 'right': (0.0, -0.3)}, True),
        ({'left': (0.054, 0.042), 'right': (0.094, -0.042)}, {'left': (0.004, 0.042), 'right': (0.044, -0.042)}, True),
        ({'left': (0.0, 0.04), 'right': (0.0, -0.04)}, {'left': (0.0, 0.03), 'right': (0.0, -0.03)}, False),
        ({'left': (0.0, 0.04), 'right': (0.0, -0.04)}, {'left': (0.0, 0.04), 'right': (0.0, -0.039995)}, False),
    ],
)
def test_a_path_that_brings_hands_closer_than_the_limit_or_than_they_start_plays_nothing(start, path, accepted):
    metadata = {}
    conversation = Conversation(llm_bimanual(model='test', image_size=8), metadata.__setitem__)
    obs = hands_at(bimanual_observation(), start)
    parts = decide(conversation, obs, move_arms(path))
    rejected = [e for e in metadata[policy_keys.TRANSCRIPT] if e['event'] == 'rejected']
    assert (parts != [[]]) is accepted
    assert (not rejected) is accepted
    if not accepted:
        assert 'Rejected: The path brings hands left and right' in str(conversation.request(obs, 1))


@pytest.mark.parametrize(
    'y,target,measured,path',
    [
        (0.04, {}, {'left': (0.0, 0.0405)}, {'right': (0.0, -0.04)}),
        (0.04, {}, {'left': (0.0, 0.0395)}, {'left': (0.0, 0.0395), 'right': (0.0, -0.04)}),
        (0.04, {'left': (-0.002, 0.04)}, {'left': (0.002, 0.04)}, {'left': (0.028, 0.04), 'right': (0.03, -0.04)}),
        (
            0.01,
            {'left': (-0.002, 0.01), 'right': (0.002, -0.01)},
            {'left': (0.002, 0.01), 'right': (-0.002, -0.01)},
            {'left': (0.028, 0.01), 'right': (0.032, -0.01)},
        ),
    ],
)
def test_hands_inside_the_limit_can_keep_their_gap_while_a_hand_trails_its_target(y, target, measured, path):
    conversation = Conversation(llm_bimanual(model='test', image_size=8), lambda *_: None)
    close = {'left': (0.0, y), 'right': (0.0, -y)}
    assert decide(conversation, hands_at(bimanual_observation(), close), move_arms(close | target)) != [[]]
    off = hands_at(bimanual_observation(), close | measured)
    assert decide(conversation, off, move_arms(path)) != [[]]


@pytest.mark.parametrize('arm,x', [('left', -0.003), ('right', 0.003)])
@pytest.mark.parametrize('max_tracking_error,accepted', [(0.0061, True), (0.0059, False)])
def test_hands_come_closer_than_both_of_their_gaps_only_within_the_tracking_error(arm, x, max_tracking_error, accepted):
    metadata = {}
    policy = llm_bimanual(model='test', image_size=8, max_tracking_error=max_tracking_error)
    conversation = Conversation(policy, metadata.__setitem__)
    close = {'left': (0.0, 0.04), 'right': (0.0, -0.04)}
    y = close[arm][1]
    assert decide(conversation, hands_at(bimanual_observation(), close), move_arms({arm: (x, y)})) != [[]]
    off = hands_at(bimanual_observation(), close | {arm: (-x, y)})
    assert (decide(conversation, off, move_arms({'left': (0.027, 0.04), 'right': (0.03, -0.04)})) != [[]]) is accepted
    assert f'within {max_tracking_error} m of its previous target' in str(conversation.request(off, 2))
    reasons = [e['reason'] for e in metadata[policy_keys.TRANSCRIPT] if e['event'] == 'rejected']
    assert reasons == (
        []
        if accepted
        else [
            'The path brings hands left and right 0.0800 m apart. They are 0.0801 m apart, and their previous '
            f'targets are 0.0801 m apart. Hand {arm} is more than {max_tracking_error} m off its previous target'
        ]
    )


@pytest.mark.parametrize(
    'y,measured,path,reason',
    [
        (
            0.04,
            {'left': (0.0, 0.0405)},
            {'right': (0.0, -0.03)},
            '0.0700 m apart. They are 0.0805 m apart, and their previous targets are 0.0800 m apart',
        ),
        (
            0.04,
            {'left': (0.0, 0.07)},
            {'right': (0.0, -0.03)},
            '0.0700 m apart. They are 0.1100 m apart, and their previous targets are 0.0800 m apart',
        ),
        (
            0.06,
            {'left': (0.0, 0.02)},
            {'right': (0.0, 0.0)},
            '0.0600 m apart. They are 0.0800 m apart, and their previous targets are 0.1200 m apart',
        ),
        (
            0.05,
            {'left': (0.1, -0.05)},
            {'left': (0.0995, -0.05)},
            '0.0995 m apart. Keep them at least 0.1000 m apart. Hand left is more than 0.005 m off its previous target',
        ),
    ],
)
def test_hands_off_their_targets_may_not_come_closer_than_the_limit_or_both_of_their_gaps(y, measured, path, reason):
    metadata = {}
    conversation = Conversation(llm_bimanual(model='test', image_size=8), metadata.__setitem__)
    targets = {'left': (0.0, y), 'right': (0.0, -y)}
    assert decide(conversation, hands_at(bimanual_observation(), targets), move_arms(targets)) != [[]]
    off = hands_at(bimanual_observation(), targets | measured)
    assert decide(conversation, off, move_arms(path)) == [[]]
    (rejected,) = [e for e in metadata[policy_keys.TRANSCRIPT] if e['event'] == 'rejected']
    assert rejected['reason'] == f'The path brings hands left and right {reason}'


def test_a_rejected_path_leaves_the_held_targets_as_they_were():
    metadata = {}
    conversation = Conversation(llm_bimanual(model='test', image_size=8), metadata.__setitem__)
    obs = bimanual_observation()
    decide(conversation, obs, move_arms({'left': (0.0, 0.15), 'right': (0.0, -0.15)}))
    assert decide(conversation, obs, move_arms({'left': (0.0, 0.05), 'right': (0.0, -0.03)})) == [[]]
    parts = decide(conversation, obs, move_arms({'right': (0.01, -0.15)}))
    left = keys.arm_channel(keys.ROBOT_COMMAND, keys.LEFT_ARM)
    np.testing.assert_allclose(parts[-1][-1][left].pose.translation, [0.0, 0.15, 0.1])
    observed = [e for e in metadata[policy_keys.TRANSCRIPT] if e['event'] == 'observation'][-1]
    assert observed['arms']['left']['previous_target']['y'] == pytest.approx(0.15)


def test_each_pair_of_hands_keeps_its_own_distance():
    metadata = {}
    policy = LLMPolicy(Endpoint('test'), Motion(), arms=('a', 'b', 'c'), camera_keys=(keys.EXTERIOR_IMAGE,))
    conversation = Conversation(policy, metadata.__setitem__)
    obs = {
        **{
            f'{keys.arm_channel(keys.ROBOT_STATE, arm)}{keys.EE_POSE_SUFFIX}': np.array([0, y, 0.1, 1, 0, 0, 0])
            for arm, y in (('a', 0.01), ('b', -0.01), ('c', 0.3))
        },
        **{keys.arm_channel(keys.GRIP, arm): 0.0 for arm in 'abc'},
        keys.TASK: 'Pass the cube.',
        keys.EXTERIOR_IMAGE: np.zeros((10, 20, 3), dtype=np.uint8),
    }
    assert decide(conversation, obs, move_arms({'c': (0.0, 0.04)})) == [[]]
    (rejected,) = [e for e in metadata[policy_keys.TRANSCRIPT] if e['event'] == 'rejected']
    assert 'hands a and c' in rejected['reason']


def test_an_arm_off_its_target_keeps_the_relief_of_a_pair_it_is_not_in():
    policy = LLMPolicy(Endpoint('test'), Motion(), arms=('a', 'b', 'c'), camera_keys=(keys.EXTERIOR_IMAGE,))
    conversation = Conversation(policy, lambda *_: None)

    def obs(positions):
        return {
            **{
                f'{keys.arm_channel(keys.ROBOT_STATE, arm)}{keys.EE_POSE_SUFFIX}': np.array([x, y, 0.1, 1, 0, 0, 0])
                for arm, (x, y) in positions.items()
            },
            **{keys.arm_channel(keys.GRIP, arm): 0.0 for arm in 'abc'},
            keys.TASK: 'Pass the cube.',
            keys.EXTERIOR_IMAGE: np.zeros((10, 20, 3), dtype=np.uint8),
        }

    start = {'a': (0.0, 0.01), 'b': (0.0, -0.01), 'c': (0.0, 0.3)}
    assert decide(conversation, obs(start), move_arms({'a': (-0.002, 0.01), 'b': (0.002, -0.01)})) != [[]]
    off = obs({'a': (0.002, 0.01), 'b': (-0.002, -0.01), 'c': (0.05, 0.3)})
    assert decide(conversation, off, move_arms({'a': (0.028, 0.01), 'b': (0.032, -0.01)})) != [[]]


def test_named_arms_get_the_arm_path_and_their_names_and_one_arm_keeps_its_own(model):
    requests, replies = model
    replies.extend([finish(), finish()])
    for policy, obs in [(llm_bimanual(model='test'), bimanual_observation()), (llm(model='test'), observation())]:
        with execution(policy) as (run, runtime, clock):
            complete(run, runtime, clock, obs)
    (named, named_tools), (one, one_tools) = requests
    schemas = {tool.name: tool.parameters_json_schema for tool in named_tools}
    assert schemas[Tool.MOVE_PATH.value] == MoveArmsPath.model_json_schema()
    assert {t.name: t.parameters_json_schema for t in one_tools}[Tool.MOVE_PATH.value] == MovePath.model_json_schema()
    for fact in (*keys.BIMANUAL_ARMS, '0.1 m', '0.005 m'):
        assert fact in str(named[0])
        assert fact not in str(one[0])


@pytest.mark.parametrize(
    'change',
    [
        {'arms': ()},
        {'arms': ('left', 'left')},
        {'arms': (None, 'left')},
        {'arms': ('',)},
        {'arms': ('a.b',)},
        {'arms': (1,)},
    ],
)
def test_arms_are_one_arm_or_distinct_names(change):
    with pytest.raises(ValueError, match='arms'):
        LLMPolicy(Endpoint('test'), Motion(), **change)
    assert LLMPolicy(Endpoint('test'), Motion()).arms == keys.ONE_ARM


@pytest.mark.parametrize(
    'change',
    [
        {'min_hand_distance': 0},
        {'min_hand_distance': np.inf},
        {'max_tracking_error': -0.001},
        {'max_tracking_error': np.inf},
        {'max_tracking_error': 0.1},
    ],
)
def test_the_hand_distance_and_the_tracking_error_are_checked(change):
    with pytest.raises(ValueError, match=next(iter(change))):
        LLMPolicy(Endpoint('test'), Motion(), arms=keys.BIMANUAL_ARMS, **change)
    assert LLMPolicy(Endpoint('test'), Motion(), arms=keys.BIMANUAL_ARMS, max_tracking_error=0).max_tracking_error == 0


def rejection(metadata) -> str | None:
    rejected = [e['reason'] for e in metadata[policy_keys.TRANSCRIPT] if e['event'] == 'rejected']
    return rejected[-1] if rejected else None


@pytest.mark.parametrize(
    'floor,z,accepted', [(0.05, 0.06, True), (0.05, 0.04, False), (0.2, 0.1, True), (0.2, 0.09, False)]
)
def test_a_path_that_takes_a_hand_below_the_floor_or_lower_than_it_is_plays_nothing(floor, z, accepted):
    metadata = {}
    conversation = Conversation(llm_bimanual(model='test', image_size=8, min_hand_height=floor), metadata.__setitem__)
    path = move_arms({'right': (0.0, -0.3, 0.15)}, {'right': (0.0, -0.3, z)})
    parts = decide(conversation, bimanual_observation(), path)
    assert (parts != [[]]) is accepted
    assert (rejection(metadata) is None) is accepted
    if not accepted:
        assert rejection(metadata) == f'Hand right goes below z {min(floor, 0.1):.4f} m on the way to waypoint 2'


def test_a_path_that_takes_a_hand_into_a_keep_out_box_that_does_not_hold_it_plays_nothing():
    metadata = {}
    box = [[-0.1, -0.5, 0.0], [0.05, -0.2, 0.5]]
    conversation = Conversation(llm_bimanual(model='test', image_size=8, keep_out=[box]), metadata.__setitem__)
    outside = hands_at(bimanual_observation(), {'right': (0.1, -0.3)})
    assert decide(conversation, outside, move_arms({'right': (0.2, -0.3, 0.1)})) != [[]]
    assert decide(conversation, bimanual_observation(), move_arms({'right': (0.03, -0.3, 0.1)})) != [[]]
    assert decide(conversation, outside, move_arms({'right': (0.0, -0.3, 0.1)})) == [[]]
    assert rejection(metadata) == (
        'Hand right goes into the box from [-0.1, -0.5, 0.0] to [0.05, -0.2, 0.5] m on the way to waypoint 1'
    )


class ScriptedReach:
    """A reach that refuses the first pose at or above ``height`` of the hand on the right (y < 0), and records the
    joints, the hand and the held target that it gets for each hand, by the side of the hand."""

    def __init__(self, height: float):
        self.height = height
        self.joints: dict[str, np.ndarray] = {}
        self.hands: dict[str, geom.Transform3D] = {}
        self.held: dict[str, geom.Transform3D | None] = {}

    def refusal(self, joints, hand, held, poses):
        side = keys.RIGHT_ARM if hand.translation[1] < 0 else keys.LEFT_ARM
        self.joints[side], self.hands[side], self.held[side] = joints, hand, held
        for index, pose in enumerate(poses):
            if hand.translation[1] < 0 and pose.translation[2] >= self.height:
                return index, 'joint 4 is at its lower limit (-1.69 rad)'
        return None

    def meta(self):
        return {'type': 'scripted'}


def with_joints(obs):
    for index, arm in enumerate(keys.BIMANUAL_ARMS):
        obs[f'{keys.arm_channel(keys.ROBOT_STATE, arm)}{keys.JOINTS_SUFFIX}'] = np.full(6, float(index))
    return obs


def test_a_path_that_a_hand_cannot_follow_plays_nothing_and_names_the_waypoint_after_a_gripper_hold():
    metadata = {}
    reach = ScriptedReach(0.25)
    conversation = Conversation(llm_bimanual(model='test', image_size=8, reach=reach), metadata.__setitem__)
    path = move_arms({'right': (0.0, -0.3, 0.15)}, {'right': (0.0, -0.3, 0.3)}, gripper=1)
    assert decide(conversation, with_joints(bimanual_observation()), path) == [[]]
    assert rejection(metadata) == 'Hand right cannot reach waypoint 2: joint 4 is at its lower limit (-1.69 rad)'
    np.testing.assert_array_equal(reach.joints[keys.LEFT_ARM], np.zeros(6))
    np.testing.assert_array_equal(reach.joints[keys.RIGHT_ARM], np.ones(6))


@pytest.mark.parametrize('measured_x,held', [(0.017, True), (0.014, False)])
def test_the_reach_gets_the_measured_hand_and_its_target_only_within_the_tracking_error(measured_x, held):
    reach = ScriptedReach(1.0)
    conversation = Conversation(llm_bimanual(model='test', image_size=8, reach=reach), lambda *_: None)
    assert decide(conversation, with_joints(bimanual_observation()), move_arms({'right': (0.02, -0.3)})) != [[]]
    trailing = hands_at(with_joints(bimanual_observation()), {'right': (measured_x, -0.3)})
    assert decide(conversation, trailing, move_arms({'left': (0.0, 0.3, 0.15)})) != [[]]
    np.testing.assert_allclose(reach.hands[keys.RIGHT_ARM].translation, [measured_x, -0.3, 0.1])
    target = reach.held[keys.RIGHT_ARM]
    assert (target is not None) is held
    if held:
        np.testing.assert_allclose(target.translation, [0.02, -0.3, 0.1])


YAM_KINEMATICS = YamKinematics()
YAM_MOUNTS = {
    keys.LEFT_ARM: geom.Transform3D(np.array([0.30, 0.305, 0.3125])),
    keys.RIGHT_ARM: geom.Transform3D(np.array([0.30, -0.305, 0.3125])),
}


def yam_observation(joints: dict[str, np.ndarray]):
    """A bimanual observation of YAM arms at ``joints``, with each hand where its joints put it."""
    obs = bimanual_observation()
    for arm, arm_joints in joints.items():
        hand = YAM_MOUNTS[arm] * YAM_KINEMATICS.fk(arm_joints)
        state = keys.arm_channel(keys.ROBOT_STATE, arm)
        obs[f'{state}{keys.EE_POSE_SUFFIX}'] = hand.as_vector(geom.Rotation.Representation.QUAT)
        obs[f'{state}{keys.JOINTS_SUFFIX}'] = arm_joints
    return obs


def move_hands(waypoint: dict[str, geom.Transform3D]):
    """A path of one waypoint of hand poses."""
    poses = {}
    for arm, pose in waypoint.items():
        (x, y, z), (roll, pitch, yaw) = map(float, pose.translation), map(float, pose.rotation.as_euler)
        poses[arm] = {'x': x, 'y': y, 'z': z, 'roll': roll, 'pitch': pitch, 'yaw': yaw, 'gripper': 0}
    return ModelResponse([
        ToolCallPart(Tool.MOVE_PATH.value, {'waypoints': [poses], 'note': 'Move.'}, tool_call_id='m')
    ])


@pytest.mark.parametrize('measured', [-1.700, -1.703, -1.71])
def test_an_arm_that_trails_its_target_past_a_joint_limit_holds_it_while_the_other_arm_moves(measured):
    # The right hand points down with joint 4 on its lower limit, then sags past it, up to 4.3 mm off its target.
    metadata = {}
    conversation = Conversation(llm_bimanual(model='test', image_size=8, reach=YamReach()), metadata.__setitem__)
    side_on = np.array([0.0, 1.2, 1.0, 0.0, 0.3, 0.0])
    on_limit = np.array([0.0, 1.9, 1.9, YAM_KINEMATICS.lower[3], 0.0, 0.0])
    right = YAM_MOUNTS[keys.RIGHT_ARM] * YAM_KINEMATICS.fk(on_limit)
    at_target = yam_observation({keys.LEFT_ARM: side_on, keys.RIGHT_ARM: on_limit})
    assert decide(conversation, at_target, move_hands({keys.RIGHT_ARM: right})) != [[]]
    sagged = on_limit.copy()
    sagged[3] = measured
    left = YAM_MOUNTS[keys.LEFT_ARM] * YAM_KINEMATICS.fk(side_on)
    moved = geom.Transform3D(left.translation + [0.03, 0, 0], left.rotation)
    trailing = yam_observation({keys.LEFT_ARM: side_on, keys.RIGHT_ARM: sagged})
    assert decide(conversation, trailing, move_hands({keys.LEFT_ARM: moved})) != [[]], rejection(metadata)


@pytest.mark.parametrize('measured_z,accepted', [(0.104, True), (0.106, False)])
def test_a_hand_within_the_tracking_error_of_its_target_can_hold_it_below_the_floor(measured_z, accepted):
    metadata = {}
    conversation = Conversation(llm_bimanual(model='test', image_size=8, min_hand_height=0.2), metadata.__setitem__)
    assert decide(conversation, bimanual_observation(), move_arms({'right': (0.0, -0.3, 0.1)})) != [[]]
    sagged = hands_at(bimanual_observation(), {'right': (0.0, -0.3, measured_z)})
    assert (decide(conversation, sagged, move_arms({'left': (0.0, 0.3, 0.15)})) != [[]]) is accepted


@pytest.mark.parametrize('measured_x,accepted', [(0.052, True), (0.054, False)])
def test_a_hand_within_the_tracking_error_of_its_target_can_hold_it_in_a_keep_out_box(measured_x, accepted):
    metadata = {}
    box = [[-0.1, -0.5, 0.0], [0.05, -0.2, 0.5]]
    conversation = Conversation(llm_bimanual(model='test', image_size=8, keep_out=[box]), metadata.__setitem__)
    assert decide(conversation, bimanual_observation(), move_arms({'right': (0.048, -0.3, 0.1)})) != [[]]
    outside = hands_at(bimanual_observation(), {'right': (measured_x, -0.3, 0.1)})
    assert (decide(conversation, outside, move_arms({'left': (0.0, 0.3, 0.15)})) != [[]]) is accepted


def test_the_floor_the_keep_out_boxes_and_the_reach_are_stated_only_when_set():
    def prompt(**settings) -> str:
        metadata = {}
        Conversation(llm_bimanual(model='test', image_size=8, **settings), metadata.__setitem__).request(
            with_joints(bimanual_observation()), 0
        )
        return metadata[policy_keys.TRANSCRIPT][0]['prompt']

    plain = prompt()
    assert 'below z' not in plain and 'into the box' not in plain and 'cannot follow with its arm' not in plain
    stated = prompt(min_hand_height=0.05, keep_out=[[[0, 0, 0], [1, 1, 1]]], reach=ScriptedReach(1.0))
    assert 'A path that takes a hand below z 0.05 m is rejected' in stated
    assert 'A path that takes a hand into the box from [0.0, 0.0, 0.0] to [1.0, 1.0, 1.0] m is rejected' in stated
    assert 'A path that a hand cannot follow with its arm is rejected' in stated


@pytest.mark.parametrize(
    'settings,match',
    [
        ({'min_hand_height': float('nan')}, 'min_hand_height must be finite'),
        ({'keep_out': (Box((0.0, 0.0, 0.0), (1.0, 0.0, 1.0)),)}, 'low below high'),
        ({'keep_out': (Box((0.0, 0.0), (1.0, 1.0)),)}, 'xyz corners'),
    ],
)
def test_the_floor_the_keep_out_boxes_and_the_reach_are_checked(settings, match):
    with pytest.raises(ValueError, match=match):
        LLMPolicy(Endpoint('test'), Motion(), **{'arms': keys.BIMANUAL_ARMS, **settings})
