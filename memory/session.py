import json
from typing import Any


class SessionMemory:
    def __init__(self, max_messages: int | None = None) -> None:
        self._messages: list[dict[str, Any]] = []
        self._max_messages = max_messages

    def _append_message(self, message: dict[str, Any]) -> None:
        self._messages.append(message)
        if self._max_messages is not None and self._max_messages > 0:
            self._messages = self._messages[-self._max_messages :]

    def add_user_message(self, content: str) -> None:
        self._append_message({"role": "user", "content": content})

    def add_assistant_message(self, content: str, reasoning_content: str | None = None) -> None:
        message = {"role": "assistant", "content": content}
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        self._append_message(message)

    def add_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        content: str = "",
        reasoning_content: str | None = None,
    ) -> None:
        message = {
            "role": "assistant",
            "content": content,
            "tool_calls": [],
        }
        for tool_call in tool_calls:
            message["tool_calls"].append(
                {
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": json.dumps(
                            tool_call["arguments"],
                            ensure_ascii=False,
                        ),
                    },
                }
            )
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        self._append_message(message)

    def add_tool_call(
        self,
        tool_call_id: str,
        name: str,
        arguments: dict[str, Any],
        content: str = "",
        reasoning_content: str | None = None,
    ) -> None:
        self.add_tool_calls(
            [
                {
                    "id": tool_call_id,
                    "name": name,
                    "arguments": arguments,
                }
            ],
            content=content,
            reasoning_content=reasoning_content,
        )

    def add_tool_result(self, tool_call_id: str, name: str, result: dict[str, Any]) -> None:
        self._append_message(
            {
                "role": "tool",
                "name": name,
                "tool_call_id": tool_call_id,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )

    def recent_messages(self, limit: int | None = None) -> list[dict[str, Any]]:
        if limit is None or limit >= len(self._messages):
            return list(self._messages)

        return list(self._messages[-limit:])
