from datetime import datetime, timezone
import json
from typing import Any

from memory.longterm import LongTermMemoryStore
from memory.project import ProjectMemoryStore
from tools.registry import RegisteredTool

ALLOWED_LONGTERM_KEYS = {
    "user_name",
    "user_role",
    "response_style",
    "working_style",
    "language_preference",
}

ALLOWED_PROJECT_KEYS = {
    "project_goal",
    "tech_stack",
    "constraints",
    "decisions",
    "current_status",
    "open_questions",
    "major_milestones",
}


def validate_memory_write_args(arguments: dict[str, Any]) -> dict[str, Any]:
    required_fields = {"memory_level", "key", "value", "reason"}
    missing_fields = required_fields - arguments.keys()
    if missing_fields:
        raise ValueError(f"Missing required fields: {sorted(missing_fields)}")

    memory_level = arguments["memory_level"]
    if memory_level not in {"project", "longterm"}:
        raise ValueError("memory_level must be 'project' or 'longterm'")

    key = str(arguments["key"]).strip()
    if not key:
        raise ValueError("key must be a non-empty string")

    if memory_level == "longterm" and key not in ALLOWED_LONGTERM_KEYS:
        allowed = ", ".join(sorted(ALLOWED_LONGTERM_KEYS))
        raise ValueError(f"Unsupported longterm key: {key}. Allowed longterm keys: {allowed}")
    if memory_level == "project" and key not in ALLOWED_PROJECT_KEYS:
        allowed = ", ".join(sorted(ALLOWED_PROJECT_KEYS))
        raise ValueError(f"Unsupported project key: {key}. Allowed project keys: {allowed}")

    return {
        "memory_level": memory_level,
        "key": key,
        "value": arguments["value"],
        "reason": arguments["reason"],
        "replace": arguments.get("replace", True),
    }


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _log_memory_event(
    database: Any,
    *,
    user_id: str,
    project_id: str,
    memory_level: str,
    key: str,
    value: Any,
    reason: str,
) -> None:
    await database.execute(
        """
        INSERT INTO memory_events (
            user_id,
            project_id,
            memory_level,
            key,
            value_json,
            reason,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            project_id,
            memory_level,
            key,
            json.dumps(value, ensure_ascii=False),
            reason,
            _utc_timestamp(),
        ),
    )


def create_memory_write_tool(
    project_store: ProjectMemoryStore,
    longterm_store: LongTermMemoryStore,
    user_id: str,
    project_id: str,
) -> RegisteredTool:
    database = project_store._database

    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments["memory_level"] == "project":
            await project_store.merge_project_memory(
                project_id,
                arguments["key"],
                arguments["value"],
                arguments["reason"],
            )
            await _log_memory_event(
                database,
                user_id=user_id,
                project_id=project_id,
                memory_level="project",
                key=arguments["key"],
                value=arguments["value"],
                reason=arguments["reason"],
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
        await _log_memory_event(
            database,
            user_id=user_id,
            project_id=project_id,
            memory_level="longterm",
            key=arguments["key"],
            value=arguments["value"],
            reason=arguments["reason"],
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
