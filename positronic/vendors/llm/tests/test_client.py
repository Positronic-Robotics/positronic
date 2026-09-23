import asyncio
import json

import httpx
import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ThinkingPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from positronic import keys
from positronic.vendors.llm.client import Endpoint
from positronic.vendors.llm.policy import OBS_TIME_NS, llm
from positronic.vendors.llm.tests.test_policy import complete, execution, finish, observation


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
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://test.invalid/v1')
    policy = llm(model='openai-chat:test-model')
    with execution(policy) as (active, rt, clock):
        assert not complete(active, rt, clock, observation()).commands
        complete(active, rt, clock, observation())
        meta = policy.meta() | rt.metadata
    assert len(requests) == 2
    assert requests[0].url.host == 'test.invalid'
    assert requests[0].headers['authorization'] == 'Bearer test-secret-do-not-record'
    body = json.loads(requests[0].content)
    assert 'data:image/png;base64,' in str(body['messages'])
    assert {tool['function']['name'] for tool in body['tools']} == {'move_to', 'done', 'give_up'}
    assert meta['stop_reason'] == 'done'
    assert 'test-secret-do-not-record' not in json.dumps(meta)
    assert clients and all(client.is_closed for client in clients)


def test_run_records_compact_reply_and_usage():
    def respond(messages, info):
        return ModelResponse(
            [ThinkingPart('private reasoning', signature='native-signature'), TextPart('Finished.'), *finish().parts],
            usage=RequestUsage(input_tokens=100, output_tokens=20),
        )

    with execution(llm(model=FunctionModel(respond))) as (active, rt, clock):
        clock.advance_to_ns(1000)
        complete(active, rt, clock, observation())
        meta = rt.metadata
    recorded = json.dumps(meta)
    assert 'private reasoning' not in recorded
    assert 'native-signature' not in recorded
    assert 'iVBOR' not in recorded
    events = meta['transcript']
    request = next(e for e in events if e['event'] == 'request')
    assert request[OBS_TIME_NS] == 1000
    assert request['cameras'] == [keys.EXTERIOR_IMAGE, keys.WRIST_IMAGE]
    response = next(e for e in events if e['event'] == 'response')
    assert response['tools'][0]['name'] == 'done'
    assert response['text'] == ['Finished.']
    assert response['usage']['input_tokens'] == 100
    assert response['usage']['output_tokens'] == 20


@pytest.mark.parametrize('base_url', ['https://key@example.com', 'https://example.com?key=secret', 'file:///tmp/api'])
def test_endpoint_rejects_credentials_in_recorded_url(monkeypatch, base_url):
    monkeypatch.setattr(FunctionModel, 'base_url', property(lambda _: base_url))
    with pytest.raises(ValueError, match='base_url'):
        Endpoint(FunctionModel(lambda messages, info: finish()))


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
    with pytest.raises(TimeoutError):
        endpoint.request([ModelRequest.user_text_prompt('stop')], [])
    assert cancelled


def test_endpoint_propagates_model_errors():
    async def fail(messages, info):
        raise RuntimeError('Request failed')

    with pytest.raises(RuntimeError, match='Request failed'):
        Endpoint(FunctionModel(fail)).request([ModelRequest.user_text_prompt('stop')], [])


def test_config_accepts_a_configured_model_and_records_generation_settings():
    def respond(messages, info):
        assert info.model_settings['temperature'] == 0.5
        assert info.model_settings['timeout'] == 2
        assert not info.allow_text_output
        return finish()

    model = FunctionModel(respond, model_name='configured', settings={'temperature': 0.25})
    policy = llm(model=model, timeout=2, settings={'temperature': 0.5})
    with execution(policy) as (active, rt, clock):
        assert not complete(active, rt, clock, observation()).commands
        assert policy.meta()['model'] == 'function:configured'
        assert policy.meta()['settings.temperature'] == 0.5
