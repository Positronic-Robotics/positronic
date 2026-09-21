import asyncio
import base64
import io
import json
from collections.abc import Awaitable, Callable

import httpx
import pytest
from PIL import Image
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import infer_model
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from positronic import keys
from positronic.vendors.llm.client import Endpoint
from positronic.vendors.llm.motion import Motion
from positronic.vendors.llm.policy import LLMPolicy, llm
from positronic.vendors.llm.tests.test_policy import complete, observation, session


@pytest.fixture
def http(monkeypatch):
    requests = []
    responses: list[httpx.Response | Callable[[], Awaitable[httpx.Response]]] = []
    clients = []

    async def respond(request):
        requests.append(request)
        response = responses.pop(0)
        return await response() if callable(response) else response

    initialize = httpx.AsyncClient.__init__

    def with_transport(client, **kwargs):
        initialize(client, **kwargs, transport=httpx.MockTransport(respond))
        clients.append(client)

    monkeypatch.setattr(httpx.AsyncClient, '__init__', with_transport)
    for env in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY'):
        monkeypatch.setenv(env, 'test-secret-do-not-record')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://test.invalid/v1')
    monkeypatch.setenv('ANTHROPIC_BASE_URL', 'https://test.invalid')
    return requests, responses, clients


@pytest.fixture(params=['openai-responses', 'openai-chat', 'anthropic', 'google'])
def provider(request):
    pytest.importorskip(
        {'openai-responses': 'openai', 'openai-chat': 'openai', 'anthropic': 'anthropic', 'google': 'google.genai'}[
            request.param
        ]
    )
    args = {'reason': 'finished', 'hindsight': 'check the image'}
    match request.param:
        case 'openai-responses':
            return (
                request.param,
                'gpt-5',
                {
                    'id': 'resp_1',
                    'object': 'response',
                    'created_at': 1,
                    'model': 'gpt-5',
                    'status': 'completed',
                    'parallel_tool_calls': False,
                    'tool_choice': 'required',
                    'tools': [],
                    'output': [
                        {'type': 'reasoning', 'id': 'rs_1', 'summary': [], 'encrypted_content': 'encrypted-reasoning'},
                        {
                            'type': 'function_call',
                            'id': 'fc_1',
                            'call_id': 'call_1',
                            'name': 'done',
                            'arguments': json.dumps(args),
                        },
                    ],
                    'usage': {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120},
                },
                'encrypted-reasoning',
            )
        case 'openai-chat':
            return (
                request.param,
                'test-model',
                {
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
                                        'function': {'name': 'done', 'arguments': json.dumps(args)},
                                    }
                                ],
                            },
                        }
                    ],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120},
                },
                'call_1',
            )
        case 'anthropic':
            return (
                request.param,
                'claude-sonnet-4-5',
                {
                    'id': 'msg_1',
                    'type': 'message',
                    'role': 'assistant',
                    'model': 'claude-sonnet-4-5',
                    'stop_reason': 'tool_use',
                    'stop_sequence': None,
                    'content': [
                        {'type': 'thinking', 'thinking': 'check the scene', 'signature': 'native-thinking-signature'},
                        {'type': 'tool_use', 'id': 'call_1', 'name': 'done', 'input': args},
                    ],
                    'usage': {'input_tokens': 100, 'output_tokens': 20},
                },
                'native-thinking-signature',
            )
        case 'google':
            return (
                request.param,
                'gemini-2.5-pro',
                {
                    'candidates': [
                        {
                            'content': {
                                'role': 'model',
                                'parts': [
                                    {
                                        'functionCall': {'name': 'done', 'args': args},
                                        'thoughtSignature': base64.b64encode(b'google-native-signature').decode(),
                                    }
                                ],
                            },
                            'finishReason': 'STOP',
                            'index': 0,
                        }
                    ],
                    'usageMetadata': {'promptTokenCount': 100, 'candidatesTokenCount': 20, 'totalTokenCount': 120},
                    'modelVersion': 'gemini-2.5-pro',
                },
                base64.b64encode(b'google-native-signature').decode(),
            )


def test_native_requests_round_trip_images_tools_and_reasoning(http, provider):
    requests, responses, clients = http
    api, model, payload, reasoning = provider
    responses.extend([httpx.Response(200, json=payload), httpx.Response(200, json=payload)])
    png = io.BytesIO()
    Image.new('RGB', (2, 2), 'red').save(png, format='PNG')
    messages: list[ModelMessage] = [
        ModelRequest([
            SystemPromptPart('Use one tool.'),
            UserPromptPart(['Camera wrist:', BinaryContent(png.getvalue(), media_type='image/png')]),
        ])
    ]
    tools = [
        ToolDefinition(
            name='done',
            parameters_json_schema={
                'type': 'object',
                'properties': {'reason': {'type': 'string'}, 'hindsight': {'type': 'string'}},
                'required': ['reason', 'hindsight'],
                'additionalProperties': False,
            },
        )
    ]
    settings: ModelSettings = {}
    if api.startswith('openai'):
        settings = pytest.importorskip('pydantic_ai.models.openai').OpenAIResponsesModelSettings(openai_store=False)
    endpoint = Endpoint(f'{api}:{model}', settings=settings)
    response = endpoint.request(messages, tools)
    call = response.tool_calls[0]
    assert call.tool_name == 'done'
    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 20
    messages += [response, ModelRequest([ToolReturnPart('done', 'Ended.', tool_call_id=call.tool_call_id)])]
    endpoint.request(messages, tools)
    assert len(requests) == 2
    first, second = [request.content.decode() for request in requests]
    encoded_image = (base64.urlsafe_b64encode if api == 'google' else base64.b64encode)(png.getvalue()).decode()
    assert encoded_image in first
    assert 'done' in first and 'hindsight' in first
    assert reasoning in second
    assert all(client.is_closed for client in clients)
    if api != 'google':
        assert 'test.invalid' == requests[0].url.host
    if api == 'openai-responses':
        assert json.loads(first)['store'] is False
        assert 'previous_response_id' not in json.loads(second)


def test_session_metadata_records_compact_provider_reply(http, provider):
    _, responses, _ = http
    api, model, payload, reasoning = provider
    responses.append(httpx.Response(200, json=payload))
    policy = LLMPolicy(Endpoint(f'{api}:{model}'), Motion())
    with session(policy) as (active, rt):
        complete(active, rt, observation(1000))
        meta = active.meta
    recorded = json.dumps(meta)
    assert 'test-secret-do-not-record' not in recorded
    if api != 'openai-chat':
        assert reasoning not in recorded
    assert 'iVBOR' not in recorded
    events = meta['transcript']
    request = next(e for e in events if e['event'] == 'request')
    assert request[keys.OBS_TIME_NS] == 1000
    assert request['cameras'] == [keys.EXTERIOR_IMAGE, keys.WRIST_IMAGE]
    response = next(e for e in events if e['event'] == 'response')
    assert response['tools'][0]['name'] == 'done'
    assert response['usage']['input_tokens'] == 100
    assert response['usage']['output_tokens'] == 20


@pytest.mark.parametrize('provider', ['openai-responses', 'openai-chat', 'anthropic'], indirect=True)
def test_sdk_retry_counts_as_one_model_invocation(http, provider):
    requests, responses, clients = http
    api, model, payload, _ = provider
    responses.extend([
        httpx.Response(429, headers={'retry-after-ms': '1'}, json={'error': {'message': 'rate limit'}}),
        httpx.Response(200, json=payload),
    ])
    policy = LLMPolicy(Endpoint(f'{api}:{model}'), Motion(), max_calls=1)
    with session(policy) as (active, rt):
        assert complete(active, rt, observation()) == []
        assert active.meta['stop_reason'] == 'done'
        assert len([e for e in active.meta['transcript'] if e['event'] == 'request']) == 1
    assert len(requests) == 2
    assert all(client.is_closed for client in clients)


def test_terminal_http_error_surfaces_and_closes_the_client(http):
    pytest.importorskip('openai')
    requests, responses, clients = http
    responses.append(httpx.Response(400, json={'error': {'message': 'bad request'}}))
    with pytest.raises(ModelHTTPError, match='400'):
        Endpoint('openai-responses:gpt-5').request([ModelRequest.user_text_prompt('stop')], [])
    assert len(requests) == 1
    assert all(client.is_closed for client in clients)


@pytest.mark.parametrize('base_url', ['https://key@example.com', 'https://example.com?key=secret', 'file:///tmp/api'])
def test_endpoint_rejects_credentials_in_recorded_url(monkeypatch, base_url):
    pytest.importorskip('openai')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-secret')
    monkeypatch.setenv('OPENAI_BASE_URL', base_url)
    with pytest.raises(ValueError, match='base_url'):
        Endpoint('openai-responses:gpt-5')


@pytest.mark.parametrize('retry', [False, True])
def test_timeout_covers_response_and_sdk_retry_wait(http, retry):
    pytest.importorskip('openai')
    requests, responses, clients = http

    async def delayed():
        if retry:
            return httpx.Response(429, headers={'retry-after': '10'}, json={'error': {'message': 'rate limit'}})
        await asyncio.sleep(5)
        raise AssertionError('The timeout should cancel this request')

    responses.append(delayed)
    endpoint = Endpoint('openai-responses:gpt-5', timeout=1.0)
    with pytest.raises(TimeoutError):
        endpoint.request([ModelRequest.user_text_prompt('stop')], [])
    assert len(requests) == 1
    assert all(client.is_closed for client in clients)


def test_config_accepts_a_configured_model_and_records_generation_settings():
    def respond(messages, info):
        assert info.model_settings['temperature'] == 0.25
        assert info.model_settings['timeout'] == 2
        from_response = {'reason': 'finished', 'hindsight': 'check the image'}
        return ModelResponse([ToolCallPart('done', from_response)])

    model = FunctionModel(respond, model_name='configured', settings={'temperature': 0.25})
    policy = llm(model=model, timeout=2)
    with session(policy) as (active, rt):
        assert complete(active, rt, observation()) == []
        assert active.meta['model'] == 'function:configured'
        assert active.meta['settings'] == {'temperature': 0.25}


def test_custom_endpoint_uses_a_library_model(http):
    pytest.importorskip('openai')
    openai = pytest.importorskip('pydantic_ai.providers.openai')
    requests, responses, clients = http
    responses.append(httpx.Response(400, json={'error': {'message': 'bad request'}}))
    model = infer_model(
        'openai-chat:test',
        provider_factory=lambda _: openai.OpenAIProvider(
            api_key='custom-test-secret', base_url='https://custom.invalid/v1'
        ),
    )
    endpoint = Endpoint(model)
    with pytest.raises(ModelHTTPError):
        endpoint.request([ModelRequest.user_text_prompt('stop')], [])
    assert requests[0].url.host == 'custom.invalid'
    assert requests[0].headers['authorization'] == 'Bearer custom-test-secret'
    assert all(client.is_closed for client in clients)
