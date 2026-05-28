import json
from typing import Any


class SessionMemory:
    def __init__(self) -> None:
        self._messages: list[dict[str, Any]] = []

    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def add_tool_call(self, name: str, arguments: dict[str, Any]) -> None:
        self._messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"{name}-call",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        },
                    }
                ],
            }
        )

    def add_tool_result(self, name: str, result: dict[str, Any]) -> None:
        self._messages.append(
            {
                "role": "tool",
                "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )

    def recent_messages(self, limit: int | None = None) -> list[dict[str, Any]]:
        if limit is None or limit >= len(self._messages):
            return list(self._messages)

        return list(self._messages[-limit:])
