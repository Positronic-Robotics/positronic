import hashlib
import json
import secrets
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, ModelResponse, ToolCallPart, UserPromptPart

from positronic import keys
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm.command import CartesianPosition
from positronic.drivers.roboarm.models import GRIPPER_SITE, EEFrame, parse_ee_frames
from positronic.offboard import server as offboard_server
from positronic.policy import keys as policy_keys
from positronic.policy.base import PolicyEnd
from positronic.policy.processors import ChunkedSchedule, EndOnField
from positronic.vendors.llm import server
from positronic.vendors.llm.client import ANTHROPIC_API_KEYS_ENV, Endpoint
from positronic.vendors.llm.motion import Motion
from positronic.vendors.llm.policy import Images, LLMPolicy, Tool
from positronic.vendors.llm.server import LLMModel, deployment, serve
from positronic.vendors.llm.tests.test_policy import (
    complete,
    execution,
    finish,
    move,
    observation,
    without_cache_points,
)


@pytest.fixture
def model(monkeypatch):
    requests, replies = [], []

    def request(endpoint, runner, model, messages, tools):
        requests.append(messages)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply() if callable(reply) else reply

    monkeypatch.setattr(Endpoint, 'request', request)
    return requests, replies


def observed_states(messages) -> list[dict]:
    _, stored = without_cache_points(messages)
    return [
        json.loads(part.content)
        for message in stored
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    ]


@pytest.fixture
def make_model(tmp_path):
    made = []

    def make(*, model_id: str = 'test', decision_timeout: float = 360.0, **kwargs) -> LLMModel:
        made.append(LLMModel(LLMPolicy(Endpoint(model_id), Motion(), **kwargs), tmp_path, decision_timeout))
        return made[-1]

    yield make
    for llm_model in made:
        llm_model.close()


@pytest.fixture
def clients(monkeypatch) -> list[dict[str, bool]]:
    """Whether each model client the endpoint opened has closed, in the order they opened."""
    opened = []
    open_endpoint = Endpoint.open

    @contextmanager
    def tracked(endpoint):
        state = {'closed': False}
        opened.append(state)
        with open_endpoint(endpoint) as client:
            yield client
        state['closed'] = True

    monkeypatch.setattr(Endpoint, 'open', tracked)
    return opened


def session_record(sessions_dir, session_id) -> dict:
    return json.loads((sessions_dir / f'{session_id}.json').read_text())


def test_a_move_answers_with_its_trajectory_after_one_held_period(model, make_model):
    requests, replies = model
    replies.append(move(0.01))
    chunk = make_model()(observation(), session_id='a')
    assert len(requests) == 1
    assert chunk[0] == {}
    waypoints = chunk[1:]
    assert all(isinstance(waypoint[keys.ROBOT_COMMAND], CartesianPosition) for waypoint in waypoints)
    assert all(waypoint[keys.TARGET_GRIP] == 0 for waypoint in waypoints)
    assert waypoints[-1][keys.ROBOT_COMMAND].pose.translation[0] == pytest.approx(0.01)
    assert policy_keys.END not in str(chunk)


@pytest.mark.parametrize('tool,end', [('done', PolicyEnd.COMPLETE), ('give_up', PolicyEnd.GIVE_UP)])
def test_a_finish_answers_with_its_end_and_every_later_call_repeats_it_without_a_request(model, make_model, tool, end):
    requests, replies = model
    replies.append(finish(tool))
    llm_model = make_model()
    assert llm_model(observation(), session_id='a') == [{policy_keys.END: end.value}]
    assert llm_model(observation(), session_id='a') == [{policy_keys.END: end.value}]
    assert len(requests) == 1


def test_a_spent_call_budget_gives_up(model, make_model):
    requests, replies = model
    replies.append(move())
    llm_model = make_model(max_calls=1)
    assert len(llm_model(observation(), session_id='a')) > 1
    assert llm_model(observation(), session_id='a') == [{policy_keys.END: PolicyEnd.GIVE_UP.value}]
    assert len(requests) == 1


def test_corrections_and_pictures_of_one_decision_stay_in_one_call_about_its_observation(model, make_model):
    requests, replies = model
    picture = ModelResponse([
        ToolCallPart(Tool.REVEAL_FRAMES.value, {'cameras': [keys.WRIST_IMAGE], 'note': 'Inspect wrist.'})
    ])
    replies.extend([picture, move('invalid'), move(0.03)])
    chunk = make_model(images=Images.ON_DEMAND)(observation(x=0.02), session_id='a')
    assert len(requests) == 3
    assert [len(observed_states(messages)) for messages in requests] == [1, 1, 1]
    assert 'Rejected' in str(requests[-1])
    assert chunk[-1][keys.ROBOT_COMMAND].pose.translation[0] == pytest.approx(0.03)


def test_two_sessions_keep_separate_conversations(model, make_model):
    requests, replies = model
    replies.extend([move(0.01), move(0.02), move(0.03)])
    llm_model = make_model()
    llm_model(observation(), session_id='a')
    llm_model(observation(x=0.5), session_id='b')
    llm_model(observation(x=0.01), session_id='a')
    a_first, b_first, a_second = (observed_states(messages) for messages in requests)
    assert [state['position_m'][0] for state in b_first] == [0.5]
    assert 'previous_target' not in b_first[0]
    assert [state['position_m'][0] for state in a_second] == [0, 0.01]
    assert a_second[-1]['previous_target']['x'] == 0.01


def test_an_ended_session_drops_its_conversation(model, make_model):
    requests, replies = model
    replies.extend([finish(), move()])
    llm_model = make_model()
    llm_model(observation(), session_id='a')
    llm_model.end_session('a')
    assert len(llm_model(observation(), session_id='a')) > 1
    assert len(observed_states(requests[-1])) == 1
    assert 'Attempt ended.' not in str(requests[-1])


def test_a_decided_end_writes_the_session_record_as_a_local_episode_records_it(model, make_model, tmp_path):
    _, replies = model
    replies.extend([move('invalid'), finish('give_up')])
    llm_model = make_model()
    llm_model(observation(), session_id='a')
    record = session_record(tmp_path, 'a')
    assert record[policy_keys.SESSION_ID] == 'a'
    assert record[policy_keys.MODEL] == 'test'
    assert record[policy_keys.STOP_REASON] == 'give_up'
    assert record[policy_keys.HINDSIGHT] == 'Inspect the final image.'
    transcript = record[policy_keys.TRANSCRIPT]
    assert [e['call'] for e in transcript if e['event'] == 'response'] == [1, 2]
    assert all('usage' in e for e in transcript if e['event'] == 'response')
    assert [path.name for path in tmp_path.iterdir()] == ['a.json']


def test_a_failed_record_write_still_latches_the_end(model, make_model, tmp_path, monkeypatch):
    requests, replies = model
    replies.append(finish())
    llm_model = make_model()
    with monkeypatch.context() as patch, pytest.raises(OSError, match='No space'):
        patch.setattr(LLMModel, '_write', Mock(side_effect=OSError('No space left on device')))
        llm_model(observation(), session_id='a')
    assert llm_model(observation(), session_id='a') == [{policy_keys.END: PolicyEnd.COMPLETE.value}]
    assert len(requests) == 1


def test_a_session_that_closes_before_its_end_writes_what_it_has(model, make_model, tmp_path):
    _, replies = model
    replies.append(move())
    llm_model = make_model()
    llm_model(observation(), session_id='a')
    assert not (tmp_path / 'a.json').exists()
    llm_model.end_session('a')
    record = session_record(tmp_path, 'a')
    assert policy_keys.STOP_REASON not in record
    assert [e['event'] for e in record[policy_keys.TRANSCRIPT]][-1] == 'accepted'


def test_closing_the_model_writes_the_record_of_each_open_session(model, make_model, tmp_path):
    _, replies = model
    replies.append(move())
    llm_model = make_model()
    llm_model(observation(), session_id='a')
    llm_model.close()
    record = session_record(tmp_path, 'a')
    assert [e['event'] for e in record[policy_keys.TRANSCRIPT]][-1] == 'accepted'


def test_a_session_that_closes_after_its_end_keeps_the_end_in_its_record(model, make_model, tmp_path):
    _, replies = model
    replies.append(finish())
    llm_model = make_model()
    llm_model(observation(), session_id='a')
    llm_model.end_session('a')
    assert session_record(tmp_path, 'a')[policy_keys.STOP_REASON] == 'done'


def test_a_reconnect_keeps_a_record_for_each_session_it_opened(model, make_model, tmp_path):
    _, replies = model
    replies.extend([move(), move()])
    llm_model = make_model()
    llm_model(observation(), session_id='dropped')
    llm_model.end_session('dropped')
    llm_model(observation(), session_id='reopened')
    llm_model.end_session('reopened')
    llm_model.end_session('never-called')
    assert sorted(path.name for path in tmp_path.iterdir()) == ['dropped.json', 'reopened.json']


def test_each_session_opens_its_own_client_and_closes_it_when_the_session_ends_fails_or_closes(
    model, make_model, clients
):
    _, replies = model
    replies.extend([finish(), RuntimeError('API unavailable'), move(), move()])
    llm_model = make_model()
    assert clients == []
    llm_model(observation(), session_id='ended')
    with pytest.raises(RuntimeError, match='API unavailable'):
        llm_model(observation(), session_id='failed')
    llm_model(observation(), session_id='closed')
    llm_model.end_session('closed')
    llm_model(observation(), session_id='open')
    assert [client['closed'] for client in clients] == [True, True, True, False]
    llm_model.close()
    assert all(client['closed'] for client in clients)


def test_each_session_draws_a_key_from_the_pool_and_records_only_its_fingerprint(
    model, make_model, monkeypatch, tmp_path
):
    pytest.importorskip('anthropic')
    pool = ['test-secret-first-org', 'test-secret-second-org']
    draws = iter([1, 0])
    monkeypatch.setattr(secrets, 'choice', lambda keys: keys[next(draws)])
    monkeypatch.setenv(ANTHROPIC_API_KEYS_ENV, ','.join(pool))
    _, replies = model
    replies.extend([finish(), finish()])
    llm_model = make_model(model_id='anthropic:test-model')
    for session_id in ('a', 'b'):
        llm_model(observation(), session_id=session_id)
    for session_id, key in zip(('a', 'b'), (pool[1], pool[0]), strict=True):
        record = session_record(tmp_path, session_id)
        assert record[policy_keys.API_KEY_FINGERPRINT] == hashlib.sha256(key.encode()).hexdigest()[:8]
        assert 'test-secret' not in json.dumps(record)


@pytest.mark.parametrize('error', [TimeoutError(), RuntimeError('API unavailable')])
def test_a_failed_decision_fails_the_session_and_writes_its_record(model, make_model, tmp_path, error):
    requests, replies = model
    replies.extend([error, move()])
    llm_model = make_model()
    with pytest.raises(type(error)):
        llm_model(observation(), session_id='a')
    assert [e['event'] for e in session_record(tmp_path, 'a')[policy_keys.TRANSCRIPT]][-1] == 'request'
    with pytest.raises(RuntimeError, match='failed in an earlier decision'):
        llm_model(observation(x=0.3), session_id='a')
    assert len(requests) == 1
    llm_model.end_session('a')
    assert len(llm_model(observation(), session_id='a')) > 1


def test_too_many_invalid_replies_fail_the_session(model, make_model):
    requests, replies = model
    replies.extend([move('invalid')] * 3)
    llm_model = make_model()
    with pytest.raises(RuntimeError, match='3 consecutive invalid'):
        llm_model(observation(), session_id='a')
    with pytest.raises(RuntimeError, match='failed in an earlier decision'):
        llm_model(observation(), session_id='a')
    assert len(requests) == 3


def test_a_decision_past_its_deadline_makes_no_further_call_and_fails_the_session(model, make_model, tmp_path):
    requests, replies = model

    def slow_invalid():
        time.sleep(0.05)
        return move('invalid')

    replies.extend([slow_invalid, move()])
    llm_model = make_model(decision_timeout=0.01)
    with pytest.raises(TimeoutError, match='deadline'):
        llm_model(observation(), session_id='a')
    assert len(requests) == 1
    record = session_record(tmp_path, 'a')
    assert [e['event'] for e in record[policy_keys.TRANSCRIPT]].count('request') == 1


def test_a_budget_spent_past_the_deadline_gives_up(model, make_model, tmp_path):
    requests, replies = model

    def slow_invalid():
        time.sleep(0.05)
        return move('invalid')

    replies.append(slow_invalid)
    llm_model = make_model(decision_timeout=0.01, max_calls=1)
    assert llm_model(observation(), session_id='a') == [{policy_keys.END: PolicyEnd.GIVE_UP.value}]
    assert len(requests) == 1
    assert session_record(tmp_path, 'a')[policy_keys.STOP_REASON] == 'call_budget'


def test_a_session_whose_client_fails_to_close_still_writes_its_record(model, make_model, tmp_path, monkeypatch):
    _, replies = model
    replies.append(finish())
    llm_model = make_model()
    with monkeypatch.context() as patch, pytest.raises(RuntimeError, match='aexit failed'):
        patch.setattr(server._Session, 'close', Mock(side_effect=RuntimeError('aexit failed')))
        llm_model(observation(), session_id='a')
    assert session_record(tmp_path, 'a')[policy_keys.STOP_REASON] == 'done'


def test_a_decision_timeout_must_be_finite_and_positive(model, make_model):
    for bad in (0.0, -1.0, float('inf')):
        with pytest.raises(ValueError, match='decision_timeout'):
            make_model(decision_timeout=bad)


def comparable(messages) -> list:
    """The messages as data, without the creation times that differ between two runs."""

    def strip(value):
        if isinstance(value, dict):
            return {key: strip(item) for key, item in value.items() if key != 'timestamp'}
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    return strip(ModelMessagesTypeAdapter.dump_python(messages, mode='json'))


def test_a_served_session_sends_the_model_what_a_local_run_sends(model, make_model, monkeypatch):
    requests, replies = model
    replies.extend([move(0.01), finish()])
    with execution(LLMPolicy(Endpoint('test'), Motion())) as (run, runtime, clock):
        complete(run, runtime, clock, observation())
        clock.advance_to_ns(1_000_000_000)
        assert complete(run, runtime, clock, observation(x=0.01)) is PolicyEnd.COMPLETE
    local = [comparable(messages) for messages in requests]
    requests.clear()
    replies.extend([move(0.01), finish()])
    now_ns = [0]
    monkeypatch.setattr(server, 'time', SimpleNamespace(monotonic=time.monotonic, monotonic_ns=lambda: now_ns[0]))
    llm_model = make_model()
    llm_model(observation(), session_id='a')
    now_ns[0] = 1_000_000_000
    assert llm_model(observation(x=0.01), session_id='a') == [{policy_keys.END: PolicyEnd.COMPLETE.value}]
    assert len(local) == 2
    assert [comparable(messages) for messages in requests] == local


def test_the_model_states_the_policy_meta_and_the_frames_of_its_poses(model, make_model):
    meta = make_model().meta()
    assert meta[policy_keys.TYPE] == policy_keys.LLM_TYPE
    assert meta[policy_keys.MODEL] == 'test'
    frames = parse_ee_frames(meta[roboarm_keys.EE_FRAMES])
    assert frames == dict.fromkeys((keys.EE_POSE, keys.TARGET_EE_POSE), EEFrame(site=GRIPPER_SITE))


def test_the_client_stack_ends_on_the_end_field_and_plays_at_the_motion_rate():
    stack = deployment(LLMPolicy(Endpoint('test'), Motion(fps=10.0))).local.to_spec()
    assert stack['seq'][0] == EndOnField().to_spec()
    assert stack['seq'][1] == ChunkedSchedule(10.0, replan_on_error=False).to_spec()


def test_the_entry_point_takes_the_flags_of_the_local_config(monkeypatch, model, tmp_path):
    model[1].append(finish())
    built = {}

    class Recorded:
        def __init__(self, build_model, pipeline, **kwargs):
            built.update(build_model=build_model, pipeline=pipeline)

        def serve(self, wires):
            built['wires'] = wires

    monkeypatch.setattr(offboard_server, 'PolicyServer', Recorded)
    sessions_dir = tmp_path / 'records'
    serve.override(**{
        'policy.model': 'test',
        'policy.motion.fps': 10.0,
        'policy.max_calls': 7,
        'sessions_dir': str(sessions_dir),
        'decision_timeout': 30.0,
    }).instantiate()
    assert sessions_dir.is_dir()
    assert len(built['wires']) == 1
    assert (
        built['pipeline'].local.to_spec() == deployment(LLMPolicy(Endpoint('test'), Motion(fps=10.0))).local.to_spec()
    )
    llm_model = built['build_model']()
    try:
        assert llm_model.meta()['max_calls'] == 7
        assert llm_model.meta()['motion.fps'] == 10.0
        llm_model(observation(), session_id='a')
        llm_model.end_session('a')
        assert (sessions_dir / 'a.json').exists()
    finally:
        llm_model.close()
