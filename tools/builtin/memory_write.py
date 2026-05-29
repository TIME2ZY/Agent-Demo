from typing import Any

from memory.longterm import LongTermMemoryStore
from memory.project import ProjectMemoryStore
from tools.registry import RegisteredTool


def validate_memory_write_args(arguments: dict[str, Any]) -> dict[str, Any]:
    required_fields = {"memory_level", "key", "value", "reason"}
    missing_fields = required_fields - arguments.keys()
    if missing_fields:
        raise ValueError(f"Missing required fields: {sorted(missing_fields)}")

    memory_level = arguments["memory_level"]
    if memory_level not in {"project", "longterm"}:
        raise ValueError("memory_level must be 'project' or 'longterm'")

    return {
        "memory_level": memory_level,
        "key": arguments["key"],
        "value": arguments["value"],
        "reason": arguments["reason"],
        "replace": arguments.get("replace", True),
    }


def create_memory_write_tool(
    project_store: ProjectMemoryStore,
    longterm_store: LongTermMemoryStore,
    user_id: str,
    project_id: str,
) -> RegisteredTool:
    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments["memory_level"] == "project":
            await project_store.merge_project_memory(
                project_id,
                arguments["key"],
                arguments["value"],
                arguments["reason"],
            )
            return {
                "ok": True,
                "memory_level": "project",
                "key": arguments["key"],
                "action": "updated",
            }

        await longterm_store.set_memory(
            user_id,
            arguments["key"],
            arguments["value"],
            arguments["reason"],
        )
        return {
            "ok": True,
            "memory_level": "longterm",
            "key": arguments["key"],
            "action": "updated",
        }

    return RegisteredTool(
        name="memory_write",
        description="Persist important project context or stable user preferences.",
        parameters={
            "type": "object",
            "properties": {
                "memory_level": {"type": "string", "enum": ["project", "longterm"]},
                "key": {"type": "string"},
                "value": {},
                "reason": {"type": "string"},
                "replace": {"type": "boolean"},
            },
            "required": ["memory_level", "key", "value", "reason"],
        },
        validator=validate_memory_write_args,
        handler=handler,
    )
