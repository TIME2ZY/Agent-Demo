from dataclasses import dataclass
from typing import Any, Awaitable, Callable


Validator = Callable[[dict[str, Any]], dict[str, Any]]
Handler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class RegisteredTool:
    name: str
    description: str
    parameters: dict[str, Any]
    validator: Validator
    handler: Handler


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")

        tool = self._tools[name]
        validated_arguments = tool.validator(arguments)
        return await tool.handler(validated_arguments)
