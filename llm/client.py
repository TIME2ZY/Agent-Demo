import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

from openai import AsyncOpenAI

from config import Settings


@dataclass
class LLMResult:
    type: Literal["message", "tool_call"]
    content: str
    reasoning_content: str | None
    tool_name: str | None
    tool_args: dict[str, Any] | None
    raw_response: dict[str, Any]


class DeepSeekClient:
    def __init__(self, settings: Settings, logger: logging.Logger | None = None) -> None:
        self._settings = settings
        self._logger = logger or logging.getLogger(__name__)
        self._client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResult:
        response = await self._client.chat.completions.create(
            model=self._settings.model_name,
            messages=messages,
            tools=tools,
        )
        payload = response.model_dump()
        self._logger.debug("deepseek_raw_response=%s", payload)
        return self.normalize_response(payload)

    @staticmethod
    def normalize_response(payload: dict[str, Any]) -> LLMResult:
        choice = payload["choices"][0]
        message = choice["message"]
        tool_calls = message.get("tool_calls") or []
        reasoning_content = (
            message.get("reasoning_content")
            or message.get("reasoning_conttent")
            or None
        )

        if tool_calls:
            tool_call = tool_calls[0]
            arguments = tool_call["function"]["arguments"]
            parsed_arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            return LLMResult(
                type="tool_call",
                content=message.get("content", "") or "",
                reasoning_content=reasoning_content,
                tool_name=tool_call["function"]["name"],
                tool_args=parsed_arguments,
                raw_response=payload,
            )

        return LLMResult(
            type="message",
            content=message.get("content", "") or "",
            reasoning_content=reasoning_content,
            tool_name=None,
            tool_args=None,
            raw_response=payload,
        )
