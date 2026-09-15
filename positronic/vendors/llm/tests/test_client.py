import asyncio
import base64
import io
import json

import httpx
import pytest
from PIL import Image
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    SystemPromptPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import ToolDefinition

from positronic import keys
from positronic.vendors.llm.client import API, Endpoint
from positronic.vendors.llm.motion import Motion
from positronic.vendors.llm.policy import LLMPolicy
from positronic.vendors.llm.tests.test_policy import complete, observation, session


@pytest.fixture
def http(monkeypatch):
    requests = []
    responses = []

    def respond(request):
        requests.append(request)
        return responses.pop(0)

    class Client(httpx.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, transport=httpx.MockTransport(respond))

    monkeypatch.setattr(httpx, 'AsyncClient', Client)
    for api in API:
        monkeypatch.setenv(api.key_env, 'test-secret-do-not-record')
    return requests, responses


@pytest.fixture(params=list(API))
def provider(request):
    args = {'reason': 'finished', 'hindsight': 'check the image'}
    match request.param:
        case API.OPENAI_RESPONSES:
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
        case API.OPENAI_CHAT:
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
        case API.ANTHROPIC:
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
        case API.GOOGLE:
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
    requests, responses = http
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
    endpoint = Endpoint(model, api=api, base_url='https://test.invalid/v1')
    response = endpoint.request(messages, tools)
    call = response.tool_calls[0]
    assert call.tool_name == 'done'
    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 20
    messages += [response, ModelRequest([ToolReturnPart('done', 'Ended.', tool_call_id=call.tool_call_id)])]
    endpoint.request(messages, tools)
    assert len(requests) == 2
    first, second = [request.content.decode() for request in requests]
    encoded_image = (base64.urlsafe_b64encode if api is API.GOOGLE else base64.b64encode)(png.getvalue()).decode()
    assert encoded_image in first
    assert 'done' in first and 'hindsight' in first
    assert reasoning in second
    assert 'test.invalid' == requests[0].url.host
    if api is API.OPENAI_RESPONSES:
        assert json.loads(first)['store'] is False
        assert 'previous_response_id' not in json.loads(second)


def test_session_metadata_records_compact_provider_reply(http, provider):
    _, responses = http
    api, model, payload, reasoning = provider
    responses.append(httpx.Response(200, json=payload))
    policy = LLMPolicy(Endpoint(model, api=api, base_url='https://test.invalid/v1'), Motion())
    with session(policy) as (active, rt):
        complete(active, rt, observation(1000))
        meta = active.meta
    recorded = json.dumps(meta)
    assert 'test-secret-do-not-record' not in recorded
    if api is not API.OPENAI_CHAT:
        assert reasoning not in recorded
    assert 'iVBOR' not in recorded
    events = meta['transcript']
    request = next(e for e in events if e['event'] == 'request')
    assert request['obs_time_ns'] == 1000
    assert request['cameras'] == [keys.EXTERIOR_IMAGE, keys.WRIST_IMAGE]
    response = next(e for e in events if e['event'] == 'response')
    assert response['tools'][0]['name'] == 'done'
    assert response['usage']['input_tokens'] == 100
    assert response['usage']['output_tokens'] == 20


def test_http_error_is_not_retried(http):
    requests, responses = http
    responses.append(httpx.Response(429, json={'error': {'message': 'rate limit', 'type': 'rate_limit'}}))
    with pytest.raises(Exception, match='429'):
        Endpoint('gpt-5').request([ModelRequest.user_text_prompt('stop')], [])
    assert len(requests) == 1


@pytest.mark.parametrize('base_url', ['https://key@example.com', 'https://example.com?key=secret', 'file:///tmp/api'])
def test_endpoint_rejects_credentials_in_recorded_url(base_url):
    with pytest.raises(ValueError, match='base_url'):
        Endpoint('gpt-5', base_url=base_url)


def test_timeout_bounds_the_whole_model_request(monkeypatch):
    async def delayed(request):
        await asyncio.sleep(5)
        raise AssertionError('The timeout should cancel this request')

    class Client(httpx.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, transport=httpx.MockTransport(delayed))

    monkeypatch.setattr(httpx, 'AsyncClient', Client)
    monkeypatch.setenv(API.OPENAI_RESPONSES.key_env, 'test-secret')
    with pytest.raises(TimeoutError):
        Endpoint('gpt-5', timeout=0.02).request([ModelRequest.user_text_prompt('stop')], [])
