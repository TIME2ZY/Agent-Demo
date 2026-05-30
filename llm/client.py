import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal

from openai import AsyncOpenAI

from config import Settings


@dataclass
class LLMResult:
    type: Literal["message", "tool_call"]
    content: str
    reasoning_content: str | None
    tool_call_id: str | None
    tool_name: str | None
    tool_args: dict[str, Any] | None
    raw_response: dict[str, Any]
    tool_calls: list[dict[str, Any]] | None = None


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

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_content_chunk: Callable[[str], None] | None = None,
    ) -> LLMResult:
        stream = await self._client.chat.completions.create(
            model=self._settings.model_name,
            messages=messages,
            tools=tools,
            stream=True,
        )

        full_content: list[str] = []
        full_reasoning: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None

        async for chunk in stream:
            payload = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
            choice = (payload.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}

            content_piece = delta.get("content") or ""
            if content_piece:
                full_content.append(content_piece)
                if on_content_chunk is not None:
                    on_content_chunk(content_piece)

            reasoning_piece = (
                delta.get("reasoning_content")
                or delta.get("reasoning_conttent")
                or ""
            )
            if reasoning_piece:
                full_reasoning.append(reasoning_piece)

            for tool_call in delta.get("tool_calls") or []:
                index = tool_call.get("index", 0)
                merged_tool_call = tool_calls_by_index.setdefault(
                    index,
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if tool_call.get("id"):
                    merged_tool_call["id"] = tool_call["id"]
                if tool_call.get("type"):
                    merged_tool_call["type"] = tool_call["type"]
                function = tool_call.get("function") or {}
                if function.get("name"):
                    merged_tool_call["function"]["name"] += function["name"]
                if function.get("arguments"):
                    merged_tool_call["function"]["arguments"] += function["arguments"]

            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]

        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(full_content),
        }
        if full_reasoning:
            message["reasoning_content"] = "".join(full_reasoning)
        if tool_calls_by_index:
            message["tool_calls"] = [
                tool_calls_by_index[index]
                for index in sorted(tool_calls_by_index)
            ]

        rebuilt_payload = {
            "choices": [
                {
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ]
        }
        self._logger.debug("deepseek_stream_response=%s", rebuilt_payload)
        return self.normalize_response(rebuilt_payload)

    @staticmethod
    def normalize_response(payload: dict[str, Any]) -> LLMResult:
        choice = payload["choices"][0]
        message = choice["message"]
        raw_tool_calls = message.get("tool_calls") or []
        reasoning_content = (
            message.get("reasoning_content")
            or message.get("reasoning_conttent")
            or None
        )

        if raw_tool_calls:
            parsed_tool_calls = []
            for tool_call in raw_tool_calls:
                arguments = tool_call["function"]["arguments"]
                parsed_arguments = (
                    json.loads(arguments) if isinstance(arguments, str) else arguments
                )
                parsed_tool_calls.append(
                    {
                        "id": tool_call.get("id"),
                        "name": tool_call["function"]["name"],
                        "arguments": parsed_arguments,
                    }
                )

            first_tool_call = parsed_tool_calls[0]
            return LLMResult(
                type="tool_call",
                content=message.get("content", "") or "",
                reasoning_content=reasoning_content,
                tool_call_id=first_tool_call["id"],
                tool_name=first_tool_call["name"],
                tool_args=first_tool_call["arguments"],
                raw_response=payload,
                tool_calls=parsed_tool_calls,
            )

        return LLMResult(
            type="message",
            content=message.get("content", "") or "",
            reasoning_content=reasoning_content,
            tool_call_id=None,
            tool_name=None,
            tool_args=None,
            raw_response=payload,
            tool_calls=None,
        )
