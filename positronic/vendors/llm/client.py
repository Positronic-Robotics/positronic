"""Direct model invocations with a deadline covering SDK retries."""

import asyncio
import hashlib
import logging
import os
import secrets
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import AsyncExitStack, contextmanager
from contextvars import copy_context
from functools import partial
from typing import NamedTuple, Self
from urllib.parse import urlsplit

from pydantic_ai.direct import model_request
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, infer_model, parse_model_id
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from positronic import telemetry, telemetry_keys
from positronic.utils import check_finite_positive

logger = logging.getLogger(__name__)


class Client(NamedTuple):
    """An open model client: its request function, and the fingerprints of the pool keys it used, in order of use."""

    request: Callable[[list[ModelMessage], list[ToolDefinition]], ModelResponse]
    api_key_fingerprints: Sequence[str]


# The environment variable that holds the comma-separated pool of Anthropic API keys.
ANTHROPIC_API_KEYS_ENV = 'ANTHROPIC_API_KEYS'
# The pool keys whose organization has no credit. No episode of this process draws them again.
_EMPTY_ANTHROPIC_KEYS: set[str] = set()


def _draw_key(pool: list[str]) -> str:
    if not (keys := [key for key in pool if key not in _EMPTY_ANTHROPIC_KEYS]):
        raise RuntimeError(f'The organization of each key in {ANTHROPIC_API_KEYS_ENV} has no credit')
    # A system draw: no seed that a sim or a library sets makes every episode take the same key.
    return secrets.choice(keys)


def _fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def _has_no_credit(error: ModelHTTPError) -> bool:
    # Anthropic gives no error type of its own to an organization with no credit, only this message.
    match error.body:
        case {'error': {'type': 'invalid_request_error', 'message': str(message)}}:
            return error.status_code == 400 and message.startswith('Your credit balance is too low')
    return False


def _anthropic_model(name: str, key: str) -> Model:
    # The Anthropic SDK comes with the optional `llm-anthropic` extra.
    from pydantic_ai.models.anthropic import AnthropicModel  # noqa: PLC0415
    from pydantic_ai.providers.anthropic import AnthropicProvider  # noqa: PLC0415

    return AnthropicModel(name, provider=AnthropicProvider(api_key=key))


class AnthropicPoolModel(WrapperModel):
    """An Anthropic model on a key from the ``ANTHROPIC_API_KEYS`` pool.

    A request that finds no credit on its key marks the key empty for this process and retries on another key.
    """

    def __init__(self, name: str, pool: list[str]):
        key = _draw_key(pool)
        super().__init__(_anthropic_model(name, key))
        self._pool = pool
        self._key = key
        self.api_key_fingerprints = [_fingerprint(key)]
        self._models = AsyncExitStack()

    async def __aenter__(self) -> Self:
        await self._models.enter_async_context(self.wrapped)
        return self

    async def __aexit__(self, *args) -> None:
        await self._models.aclose()

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        while True:
            try:
                return await self.wrapped.request(messages, model_settings, model_request_parameters)
            except ModelHTTPError as error:
                if not _has_no_credit(error):
                    raise
                _EMPTY_ANTHROPIC_KEYS.add(self._key)
                key = _draw_key(self._pool)
                logger.warning(
                    'Anthropic key %s has no credit; the request retries on key %s',
                    self.api_key_fingerprints[-1],
                    _fingerprint(key),
                )
                self.wrapped = await self._models.enter_async_context(_anthropic_model(self.model_name, key))
                self._key = key
                self.api_key_fingerprints.append(_fingerprint(key))


def _anthropic_key_pool() -> list[str] | None:
    """The keys of the comma-separated ``ANTHROPIC_API_KEYS`` pool, or ``None`` when the variable is unset."""
    pool = os.environ.get(ANTHROPIC_API_KEYS_ENV)
    if pool is None:
        return None
    keys = [key.strip() for key in pool.split(',')]
    if not all(keys):
        raise ValueError(f'{ANTHROPIC_API_KEYS_ENV} must list API keys separated by commas, without blank entries')
    return keys


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
        check_finite_positive(timeout, 'timeout')
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

    def _infer_model(self) -> Model:
        """The model `infer_model` builds.

        An `openai-responses:` model keeps cache points as breakpoints when the endpoint asks for them. An
        `anthropic:` model takes its key from the pool when ``ANTHROPIC_API_KEYS`` is set.
        """
        model = self.model
        if isinstance(model, Model):
            return model
        kind, name = parse_model_id(model)
        if kind == 'openai-responses' and self.cache_breakpoints:
            # The OpenAI SDK comes with the optional `llm-openai` extra.
            from .responses import BreakpointResponsesModel  # noqa: PLC0415

            return BreakpointResponsesModel(name, provider=kind)
        if kind == 'anthropic' and (pool := _anthropic_key_pool()) is not None:
            return AnthropicPoolModel(name, pool)
        return infer_model(model)

    @contextmanager
    def open(self) -> Iterator[Client]:
        """Own one model client and event loop until all requests have finished."""
        with asyncio.Runner() as runner:
            model = self._infer_model()
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
                fingerprints = model.api_key_fingerprints if isinstance(model, AnthropicPoolModel) else ()
                yield Client(partial(self.request, runner, model), fingerprints)
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
