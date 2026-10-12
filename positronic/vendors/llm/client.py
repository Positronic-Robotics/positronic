"""Direct model invocations with a deadline covering SDK retries."""

import asyncio
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import copy_context
from functools import partial
from urllib.parse import urlsplit

from pydantic_ai.direct import model_request
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, infer_model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from positronic import telemetry, telemetry_keys


class Endpoint:
    """Model configuration and recorded generation settings."""

    def __init__(self, model: str | Model, *, timeout: float = 120.0, settings: ModelSettings | None = None):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be finite and positive')
        # TODO: Represent custom providers without retaining preconstructed SDK models.
        self.model = model
        self.timeout = timeout
        defaults = model.settings if isinstance(model, Model) else None
        self.settings: ModelSettings = {**(defaults or {}), **(settings or {})}
        if {'extra_headers', 'extra_body', 'extra_query'}.intersection(self.settings):
            raise ValueError('settings must describe generation; configure transport on the model provider')

    @property
    def model_id(self) -> str:
        return self.model.model_id if isinstance(self.model, Model) else self.model

    @contextmanager
    def open(self) -> Iterator[Callable[[list[ModelMessage], list[ToolDefinition]], ModelResponse]]:
        """Own one model client and event loop until all requests have finished."""
        with asyncio.Runner() as runner:
            model = infer_model(self.model)
            runner.run(model.__aenter__())
            try:
                if model.base_url is not None:
                    url = urlsplit(model.base_url)
                    if (
                        url.scheme not in ('http', 'https')
                        or not url.netloc
                        or url.username
                        or url.password
                        or url.query
                        or url.fragment
                    ):
                        raise ValueError('base_url must be an HTTP(S) URL without credentials or query parameters')
                yield partial(self.request, runner, model)
            finally:
                runner.run(model.__aexit__(None, None, None))

    async def _request(self, model: Model, messages: list[ModelMessage], tools: list[ToolDefinition]) -> ModelResponse:
        return await asyncio.wait_for(
            model_request(
                model,
                messages,
                model_settings={**self.settings, 'timeout': self.timeout},
                model_request_parameters=ModelRequestParameters(function_tools=tools, allow_text_output=False),
            ),
            timeout=self.timeout,
        )

    @telemetry.traced(telemetry_keys.SPAN_POLICY_INFER)
    def request(
        self, runner: asyncio.Runner, model: Model, messages: list[ModelMessage], tools: list[ToolDefinition]
    ) -> ModelResponse:
        return runner.run(self._request(model, messages, tools), context=copy_context())
