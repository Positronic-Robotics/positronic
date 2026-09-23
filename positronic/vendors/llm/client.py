"""Direct model invocations with a deadline covering SDK retries."""

import asyncio
import math
from urllib.parse import urlsplit

from pydantic_ai.direct import model_request
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, infer_model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from positronic import telemetry, telemetry_keys


class Endpoint:
    """A Pydantic AI model and its recorded generation settings."""

    def __init__(self, model: str | Model, *, timeout: float = 120.0, settings: ModelSettings | None = None):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be finite and positive')
        self.model = infer_model(model)
        self.timeout = timeout
        self.settings: ModelSettings = {**(self.model.settings or {}), **(settings or {})}
        if self.model.base_url is not None:
            url = urlsplit(self.model.base_url)
            if (
                url.scheme not in ('http', 'https')
                or not url.netloc
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ValueError('base_url must be an HTTP(S) URL without credentials or query parameters')
        if {'extra_headers', 'extra_body', 'extra_query'}.intersection(self.settings):
            raise ValueError('settings must describe generation; configure transport on the model provider')

    async def _request(self, messages: list[ModelMessage], tools: list[ToolDefinition]) -> ModelResponse:
        async with self.model:
            return await asyncio.wait_for(
                model_request(
                    self.model,
                    messages,
                    model_settings={**self.settings, 'timeout': self.timeout},
                    model_request_parameters=ModelRequestParameters(function_tools=tools, allow_text_output=False),
                ),
                timeout=self.timeout,
            )

    @telemetry.traced(telemetry_keys.SPAN_POLICY_INFER)
    def request(self, messages: list[ModelMessage], tools: list[ToolDefinition]) -> ModelResponse:
        return asyncio.run(self._request(messages, tools))
