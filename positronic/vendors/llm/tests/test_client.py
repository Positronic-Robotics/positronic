import asyncio
import json
import multiprocessing
import pickle
from contextvars import ContextVar, copy_context
from copy import copy
from multiprocessing.reduction import ForkingPickler

import httpx
import pytest
from pydantic_ai.messages import (
    BinaryContent,
    CachePoint,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from positronic import keys
from positronic.policy import PolicyEnd
from positronic.policy import keys as policy_keys
from positronic.vendors.llm.client import Endpoint
from positronic.vendors.llm.policy import llm
from positronic.vendors.llm.tests.test_policy import complete, execution, finish, observation


@pytest.mark.timeout(30)
@pytest.mark.parametrize('model', ['openai-chat:test-model', 'anthropic:test-model', 'google:test-model'])
def test_model_name_policy_and_metadata_cross_a_spawned_process(model):
    policy = llm(model=model, settings={'temperature': 0.5})
    metadata = policy.meta()
    with multiprocessing.get_context('spawn').Pool(1) as pool:
        restored = pool.apply(copy, (policy,))
    assert restored.meta() == metadata
    assert metadata['model'] == model
    assert metadata['settings.temperature'] == 0.5


def test_openai_compatible_endpoint_round_trip(monkeypatch):
    pytest.importorskip('openai')
    requests, clients = [], []

    async def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                'id': 'chatcmpl-1',
                'object': 'chat.completion',
                'created': 1,
                'model': 'test-model',
                'choices': [
                    {
                        'index': 0,
                        'finish_reason': 'tool_calls',
                        'message': {
                            'role': 'assistant',
                            'content': None,
                            'tool_calls': [
                                {
                                    'id': 'call_1',
                                    'type': 'function',
                                    'function': {
                                        'name': 'move_to' if len(requests) == 1 else 'done',
                                        'arguments': json.dumps(
                                            {
                                                'x': 0.01,
                                                'y': 0,
                                                'z': 0,
                                                'roll': 0,
                                                'pitch': 0,
                                                'yaw': 0,
                                                'gripper': 0,
                                                'note': 'approach',
                                            }
                                            if len(requests) == 1
                                            else {'reason': 'finished', 'hindsight': 'check the image'}
                                        ),
                                    },
                                }
                            ],
                        },
                    }
                ],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120},
            },
        )

    initialize = httpx.AsyncClient.__init__

    def with_transport(client, **kwargs):
        initialize(client, **kwargs, transport=httpx.MockTransport(respond))
        clients.append(client)

    monkeypatch.setattr(httpx.AsyncClient, '__init__', with_transport)
    monkeypatch.setenv('OPENAI_API_KEY', 'test-secret-do-not-record')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://endpoint-secret.invalid/path-secret/v1')
    policy = llm(model='openai-chat:test-model')
    metadata = policy.meta()
    assert not clients
    policy = pickle.loads(ForkingPickler.dumps(policy))
    assert policy.meta() == metadata
    with execution(policy) as (active, rt, clock):
        assert not complete(active, rt, clock, observation()).commands
        assert len(clients) == 1 and not clients[0].is_closed
        assert complete(active, rt, clock, observation()) is PolicyEnd.COMPLETE
        meta = policy.meta() | rt.metadata_snapshot()
        assert pickle.loads(ForkingPickler.dumps(policy)).meta() == metadata
    assert len(requests) == 2
    assert requests[0].url.host == 'endpoint-secret.invalid'
    assert requests[0].url.path == '/path-secret/v1/chat/completions'
    assert requests[0].headers['authorization'] == 'Bearer test-secret-do-not-record'
    body = json.loads(requests[0].content)
    assert 'data:image/png;base64,' in str(body['messages'])
    assert {tool['function']['name'] for tool in body['tools']} == {'move_to', 'done', 'give_up'}
    assert meta[policy_keys.STOP_REASON] == 'done'
    recorded = json.dumps(meta)
    assert 'test-secret-do-not-record' not in recorded
    assert 'base_url' not in meta
    assert 'endpoint-secret' not in recorded
    assert 'path-secret' not in recorded
    assert clients and all(client.is_closed for client in clients)
    with execution(policy) as (active, rt, clock):
        assert complete(active, rt, clock, observation()) is PolicyEnd.COMPLETE
        assert rt.metadata_snapshot()[policy_keys.STOP_REASON] == 'done'
        assert len(clients) == 2
    assert all(client.is_closed for client in clients)
    assert pickle.loads(ForkingPickler.dumps(policy)).meta() == metadata


def test_openai_responses_endpoint_sends_each_cache_point_as_a_breakpoint_on_the_block_before_it(monkeypatch):
    pytest.importorskip('openai')
    requests = []

    async def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                'id': 'resp_1',
                'object': 'response',
                'created_at': 1,
                'model': 'test-model',
                'status': 'completed',
                'output': [
                    {
                        'type': 'function_call',
                        'id': 'fc_1',
                        'call_id': 'call_1',
                        'name': 'done',
                        'arguments': json.dumps({'reason': 'finished', 'hindsight': 'check the image'}),
                        'status': 'completed',
                    }
                ],
                'parallel_tool_calls': True,
                'tool_choice': 'auto',
                'tools': [],
                'usage': {
                    'input_tokens': 100,
                    'output_tokens': 20,
                    'total_tokens': 120,
                    'input_tokens_details': {'cached_tokens': 0},
                    'output_tokens_details': {'reasoning_tokens': 0},
                },
            },
        )

    initialize = httpx.AsyncClient.__init__
    monkeypatch.setattr(
        httpx.AsyncClient,
        '__init__',
        lambda client, **kwargs: initialize(client, **kwargs, transport=httpx.MockTransport(respond)),
    )
    monkeypatch.setenv('OPENAI_API_KEY', 'test-secret-do-not-record')
    image = BinaryContent(b'png', media_type='image/png')
    messages = [
        ModelRequest([SystemPromptPart('Control the arm.')]),
        ModelRequest([UserPromptPart(['Camera wrist:', image, CachePoint()])]),
        ModelRequest([UserPromptPart(['Observation.', CachePoint()])]),
    ]
    with Endpoint('openai-responses:test-model').open() as request:
        request(messages, [])
    system, frames, last = json.loads(requests[0].content)['input']
    breakpoint = {'mode': 'explicit'}
    assert [block.get('prompt_cache_breakpoint') for block in frames['content']] == [None, breakpoint]
    assert frames['content'][1]['type'] == 'input_image'
    assert [block.get('prompt_cache_breakpoint') for block in last['content']] == [breakpoint]
    assert system == {'role': 'system', 'content': 'Control the arm.'}


def test_run_records_compact_reply_and_usage(monkeypatch):
    def respond(messages, info):
        return ModelResponse(
            [ThinkingPart('private reasoning', signature='native-signature'), TextPart('Finished.'), *finish().parts],
            usage=RequestUsage(input_tokens=100, output_tokens=20),
        )

    monkeypatch.setattr(FunctionModel, 'base_url', property(lambda _: 'https://endpoint-secret.invalid/path-secret/v1'))
    policy = llm(model=FunctionModel(respond))
    with execution(policy) as (active, rt, clock):
        clock.advance_to_ns(1000)
        assert complete(active, rt, clock, observation()) is PolicyEnd.COMPLETE
        meta = policy.meta() | rt.metadata_snapshot()
    recorded = json.dumps(meta)
    assert 'base_url' not in meta
    assert 'endpoint-secret' not in recorded
    assert 'path-secret' not in recorded
    assert 'private reasoning' not in recorded
    assert 'native-signature' not in recorded
    assert 'iVBOR' not in recorded
    events = meta[policy_keys.TRANSCRIPT]
    request = next(e for e in events if e['event'] == 'request')
    assert request[policy_keys.OBS_TIME_NS] == 1000
    assert request['cameras'] == [keys.EXTERIOR_IMAGE, keys.WRIST_IMAGE]
    response = next(e for e in events if e['event'] == 'response')
    assert response['tools'][0]['name'] == 'done'
    assert response['text'] == ['Finished.']
    assert response['usage']['input_tokens'] == 100
    assert response['usage']['output_tokens'] == 20


@pytest.mark.parametrize('base_url', ['https://key@example.com', 'https://example.com?key=secret', 'file:///tmp/api'])
def test_endpoint_rejects_userinfo_query_and_non_http_urls(monkeypatch, base_url):
    monkeypatch.setattr(FunctionModel, 'base_url', property(lambda _: base_url))
    with pytest.raises(ValueError, match='base_url'):
        with Endpoint(FunctionModel(lambda messages, info: finish())).open():
            pytest.fail('The endpoint must reject this URL before yielding a client')


def test_endpoint_deadline_cancels_the_model_request():
    cancelled = False

    async def delayed(messages, info):
        nonlocal cancelled
        try:
            await asyncio.Event().wait()
        finally:
            cancelled = True
        raise AssertionError('The timeout should cancel this request')

    endpoint = Endpoint(FunctionModel(delayed), timeout=1.0)
    with endpoint.open() as request, pytest.raises(TimeoutError):
        request([ModelRequest.user_text_prompt('stop')], [])
    assert cancelled


def test_endpoint_propagates_model_errors():
    async def fail(messages, info):
        raise RuntimeError('Request failed')

    with Endpoint(FunctionModel(fail)).open() as request, pytest.raises(RuntimeError, match='Request failed'):
        request([ModelRequest.user_text_prompt('stop')], [])


def test_each_request_inherits_its_callers_context():
    marker = ContextVar('request_marker')
    seen = []

    async def respond(messages, info):
        seen.append(marker.get())
        return finish()

    with Endpoint(FunctionModel(respond)).open() as request:
        for value in ('first', 'second'):
            context = copy_context()
            context.run(marker.set, value)
            context.run(request, [ModelRequest.user_text_prompt('stop')], [])
    assert seen == ['first', 'second']


def test_config_accepts_a_configured_model_and_records_generation_settings():
    def respond(messages, info):
        assert info.model_settings['temperature'] == 0.5
        assert info.model_settings['timeout'] == 2
        assert not info.allow_text_output
        return finish()

    model = FunctionModel(respond, model_name='configured', settings={'temperature': 0.25})
    policy = llm(model=model, timeout=2, settings={'temperature': 0.5})
    with execution(policy) as (active, rt, clock):
        assert complete(active, rt, clock, observation()) is PolicyEnd.COMPLETE
        assert policy.meta()['model'] == 'function:configured'
        assert policy.meta()['settings.temperature'] == 0.5
