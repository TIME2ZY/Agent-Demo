import json
from typing import Any


class SessionMemory:
    def __init__(self) -> None:
        self._messages: list[dict[str, Any]] = []

    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str, reasoning_content: str | None = None) -> None:
        message = {"role": "assistant", "content": content}
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        self._messages.append(message)

    def add_tool_call(
        self,
        tool_call_id: str,
        name: str,
        arguments: dict[str, Any],
        content: str = "",
        reasoning_content: str | None = None,
    ) -> None:
        message = {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            ],
        }
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        self._messages.append(message)

    def add_tool_result(self, tool_call_id: str, name: str, result: dict[str, Any]) -> None:
        self._messages.append(
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
