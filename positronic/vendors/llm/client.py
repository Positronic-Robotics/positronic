"""Native provider APIs, with one bounded request per call."""

import asyncio
import math
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from enum import StrEnum
from urllib.parse import urlsplit

import httpx
from anthropic import AsyncAnthropic
from google.genai import Client as GoogleClient
from google.genai.types import HttpOptions, HttpRetryOptions
from openai import AsyncOpenAI
from pydantic_ai.direct import model_request
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition


class API(StrEnum):
    OPENAI_RESPONSES = 'openai-responses'
    OPENAI_CHAT = 'openai-chat'
    ANTHROPIC = 'anthropic'
    GOOGLE = 'google'

    @property
    def key_env(self) -> str:
        return {
            API.OPENAI_RESPONSES: 'OPENAI_API_KEY',
            API.OPENAI_CHAT: 'OPENAI_API_KEY',
            API.ANTHROPIC: 'ANTHROPIC_API_KEY',
            API.GOOGLE: 'GEMINI_API_KEY',
        }[self]


@dataclass(frozen=True)
class Endpoint:
    """A model's public endpoint. Credentials are read from the named environment variable."""

    model: str
    api: API = API.OPENAI_RESPONSES
    base_url: str | None = None
    api_key_env: str | None = None
    timeout: float = 120.0
    settings: ModelSettings = field(default_factory=ModelSettings)

    def __post_init__(self):
        if not self.model.strip():
            raise ValueError('model must be nonempty')
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError('timeout must be finite and positive')
        if self.base_url is not None:
            url = urlsplit(self.base_url)
            if (
                url.scheme not in ('http', 'https')
                or not url.netloc
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ValueError('base_url must be an HTTP(S) URL without credentials or query parameters')
        forbidden = {
            'extra_headers',
            'extra_body',
            'extra_query',
            'openai_previous_response_id',
            'openai_conversation_id',
        }
        if forbidden.intersection(self.settings):
            raise ValueError('settings must describe generation, without transport overrides or server-side history')

    def _model(self, http: httpx.AsyncClient, stack: AsyncExitStack) -> Model:
        env = self.api_key_env if self.api_key_env is not None else self.api.key_env
        secret = os.environ.get(env)
        if not secret:
            raise ValueError(f'{env} is not set')
        match self.api:
            case API.OPENAI_RESPONSES | API.OPENAI_CHAT:
                sdk = AsyncOpenAI(api_key=secret, base_url=self.base_url, http_client=http, max_retries=0)
                provider = OpenAIProvider(openai_client=sdk)
                cls = OpenAIResponsesModel if self.api is API.OPENAI_RESPONSES else OpenAIChatModel
                return cls(self.model, provider=provider)
            case API.ANTHROPIC:
                anthropic = AsyncAnthropic(api_key=secret, base_url=self.base_url, http_client=http, max_retries=0)
                return AnthropicModel(self.model, provider=AnthropicProvider(anthropic_client=anthropic))
            case API.GOOGLE:
                google = GoogleClient(
                    api_key=secret,
                    http_options=HttpOptions(
                        base_url=self.base_url,
                        httpx_async_client=http,
                        timeout=round(self.timeout * 1000),
                        retry_options=HttpRetryOptions(attempts=1),
                    ),
                )
                stack.callback(google.close)
                stack.push_async_callback(google.aio.aclose)
                return GoogleModel(self.model, provider=GoogleProvider(client=google))
        raise ValueError(f'Unsupported API: {self.api}')

    async def _request(self, messages: list[ModelMessage], tools: list[ToolDefinition]) -> ModelResponse:
        async with AsyncExitStack() as stack:
            http = await stack.enter_async_context(httpx.AsyncClient(timeout=self.timeout))
            model = self._model(http, stack)
            settings: ModelSettings = {**self.settings, 'timeout': self.timeout}
            if self.api in (API.OPENAI_RESPONSES, API.OPENAI_CHAT):
                settings = {**settings, **OpenAIResponsesModelSettings(openai_store=False)}
            return await asyncio.wait_for(
                model_request(
                    model,
                    messages,
                    model_settings=settings,
                    model_request_parameters=ModelRequestParameters(function_tools=tools, allow_text_output=False),
                ),
                timeout=self.timeout,
            )

    def request(self, messages: list[ModelMessage], tools: list[ToolDefinition]) -> ModelResponse:
        return asyncio.run(self._request(messages, tools))
