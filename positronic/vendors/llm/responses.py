"""An OpenAI Responses model that keeps prompt cache points."""

from typing import Any, cast

from openai.types import responses
from pydantic_ai.messages import CachePoint, UserPromptPart
from pydantic_ai.models.openai import OpenAIResponsesModel


class BreakpointResponsesModel(OpenAIResponsesModel):
    """Sends each `CachePoint` as an explicit `prompt_cache_breakpoint` on the content block before it."""

    async def _map_user_prompt(self, part: UserPromptPart) -> responses.EasyInputMessageParam:
        message = await super()._map_user_prompt(part)
        if isinstance(part.content, str):
            return message
        content = cast(list[dict[str, Any]], message['content'])
        # The parent maps every content item except a `CachePoint` to exactly one block, in order.
        blocks = 0
        for item in part.content:
            if not isinstance(item, CachePoint):
                blocks += 1
            elif blocks:
                content[blocks - 1]['prompt_cache_breakpoint'] = {'mode': 'explicit'}
        return message
