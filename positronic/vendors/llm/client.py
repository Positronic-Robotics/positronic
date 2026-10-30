"""Direct model invocations with a deadline covering SDK retries."""

import asyncio
import hashlib
import math
import os
import secrets
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import copy_context
from functools import partial
from typing import NamedTuple
from urllib.parse import urlsplit

from pydantic_ai.direct import model_request
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, infer_model, parse_model_id
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from positronic import telemetry, telemetry_keys


class Client(NamedTuple):
    """An open model client: its request function, and the fingerprint of the API key it uses, if one was drawn."""

    request: Callable[[list[ModelMessage], list[ToolDefinition]], ModelResponse]
    api_key_fingerprint: str | None


# The environment variable that holds the comma-separated pool of Anthropic API keys.
ANTHROPIC_API_KEYS_ENV = 'ANTHROPIC_API_KEYS'


def draw_anthropic_key() -> str | None:
    """A random key from the comma-separated ``ANTHROPIC_API_KEYS`` pool, or ``None`` when the variable is unset."""
    pool = os.environ.get(ANTHROPIC_API_KEYS_ENV)
    if pool is None:
        return None
    keys = [key.strip() for key in pool.split(',')]
    if not all(keys):
        raise ValueError(f'{ANTHROPIC_API_KEYS_ENV} must list API keys separated by commas, without blank entries')
    # A system draw: no seed that a sim or a library sets makes every episode take the same key.
    return secrets.choice(keys)


class Endpoint:
    """Model configuration and recorded generation settings.

    ``cache_breakpoints`` sends each cache point to an OpenAI Responses model as a ``prompt_cache_breakpoint``. Only
    the GPT-5.6 family and later accept it; an earlier model answers a request that carries it with HTTP 400.
    """

    def __init__(
        self,
        model: str | Model,
        *,
        timeout: float = 120.0,
        settings: ModelSettings | None = None,
        cache_breakpoints: bool = False,
    ):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be finite and positive')
        # An `OpenAIResponsesModel` exists only after its module loads the SDK of the optional `llm-openai` extra.
        if cache_breakpoints and isinstance(model, Model) and 'pydantic_ai.models.openai' in sys.modules:
            from pydantic_ai.models.openai import OpenAIResponsesModel  # noqa: PLC0415

            from .responses import BreakpointResponsesModel  # noqa: PLC0415

            if isinstance(model, OpenAIResponsesModel) and not isinstance(model, BreakpointResponsesModel):
                raise ValueError('OpenAIResponsesModel drops cache points; configure a BreakpointResponsesModel')
        # TODO: Represent custom providers without retaining preconstructed SDK models.
        self.model = model
        self.timeout = timeout
        self.cache_breakpoints = cache_breakpoints
        defaults = model.settings if isinstance(model, Model) else None
        self.settings: ModelSettings = {**(defaults or {}), **(settings or {})}
        if {'extra_headers', 'extra_body', 'extra_query'}.intersection(self.settings):
            raise ValueError('settings must describe generation; configure transport on the model provider')

    @property
    def model_id(self) -> str:
        return self.model.model_id if isinstance(self.model, Model) else self.model

    def _infer_model(self) -> tuple[Model, str | None]:
        """The model `infer_model` builds, with the fingerprint of the API key drawn for it, if any.

        An `openai-responses:` model keeps cache points as breakpoints when the endpoint asks for them.
        """
        model = self.model
        if isinstance(model, Model):
            return model, None
        kind, name = parse_model_id(model)
        if kind == 'openai-responses' and self.cache_breakpoints:
            # The OpenAI SDK comes with the optional `llm-openai` extra.
            from .responses import BreakpointResponsesModel  # noqa: PLC0415

            return BreakpointResponsesModel(name, provider=kind), None
        if kind == 'anthropic' and (key := draw_anthropic_key()) is not None:
            # The Anthropic SDK comes with the optional `llm-anthropic` extra.
            from pydantic_ai.models.anthropic import AnthropicModel  # noqa: PLC0415
            from pydantic_ai.providers.anthropic import AnthropicProvider  # noqa: PLC0415

            fingerprint = hashlib.sha256(key.encode()).hexdigest()[:8]
            return AnthropicModel(name, provider=AnthropicProvider(api_key=key)), fingerprint
        return infer_model(model), None

    @contextmanager
    def open(self) -> Iterator[Client]:
        """Own one model client and event loop until all requests have finished."""
        with asyncio.Runner() as runner:
            model, fingerprint = self._infer_model()
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
                yield Client(partial(self.request, runner, model), fingerprint)
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
