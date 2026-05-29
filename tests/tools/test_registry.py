import pytest

from memory.longterm import LongTermMemoryStore
from memory.project import ProjectMemoryStore
from storage.db import Database
from tools.builtin.memory_write import (
    ALLOWED_PROJECT_KEYS,
    ALLOWED_LONGTERM_KEYS,
    create_memory_write_tool,
    validate_memory_write_args,
)
from tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_registry_dispatches_longterm_memory_write(db_path):
    database = Database(db_path)
    await database.connect()
    await database.init_schema()

    project_store = ProjectMemoryStore(database)
    longterm_store = LongTermMemoryStore(database)
    registry = ToolRegistry()
    registry.register(
        create_memory_write_tool(
            project_store,
            longterm_store,
            "demo-user",
            "demo-project",
        )
    )

    result = await registry.dispatch(
        "memory_write",
        {
            "memory_level": "longterm",
            "key": "response_style",
            "value": "Prefer concise Chinese",
            "reason": "explicit user preference",
        },
    )

    assert result["ok"] is True
    stored = await longterm_store.get_memory("demo-user", "response_style")
    assert stored["value"] == "Prefer concise Chinese"
    event_rows = await database.fetchall(
        """
        SELECT memory_level, key, value_json, reason
        FROM memory_events
        ORDER BY id
        """
    )
    assert event_rows == [
        {
            "memory_level": "longterm",
            "key": "response_style",
            "value_json": '"Prefer concise Chinese"',
            "reason": "explicit user preference",
        }
    ]

    await database.close()


@pytest.mark.asyncio
async def test_registry_rejects_invalid_memory_level(db_path):
    database = Database(db_path)
    await database.connect()
    await database.init_schema()

    project_store = ProjectMemoryStore(database)
    longterm_store = LongTermMemoryStore(database)
    registry = ToolRegistry()
    registry.register(
        create_memory_write_tool(
            project_store,
            longterm_store,
            "demo-user",
            "demo-project",
        )
    )

    with pytest.raises(ValueError, match="memory_level"):
        await registry.dispatch(
            "memory_write",
            {
                "memory_level": "session",
                "key": "response_style",
                "value": "Prefer concise Chinese",
                "reason": "invalid level",
            },
        )

    await database.close()


def test_validate_memory_write_args_allows_standard_longterm_keys():
    for key in ALLOWED_LONGTERM_KEYS:
        validated = validate_memory_write_args(
            {
                "memory_level": "longterm",
                "key": key,
                "value": "demo",
                "reason": "standard key",
            }
        )

        assert validated["key"] == key


def test_validate_memory_write_args_rejects_unknown_longterm_keys():
    with pytest.raises(ValueError, match="longterm key"):
        validate_memory_write_args(
            {
                "memory_level": "longterm",
                "key": "user_info",
                "value": "用户叫张雨，是一名Python开发工程师。",
                "reason": "non-standard shape",
            }
        )


def test_validate_memory_write_args_allows_standard_project_keys():
    for key in ALLOWED_PROJECT_KEYS:
        validated = validate_memory_write_args(
            {
                "memory_level": "project",
                "key": key,
                "value": "demo",
                "reason": "standard key",
            }
        )

        assert validated["key"] == key


def test_validate_memory_write_args_rejects_unknown_project_keys():
    with pytest.raises(ValueError, match="project key"):
        validate_memory_write_args(
            {
                "memory_level": "project",
                "key": "project_summary",
                "value": "free-form summary",
                "reason": "non-standard shape",
            }
        )


@pytest.mark.asyncio
async def test_registry_dispatches_project_memory_write_and_logs_event(db_path):
    database = Database(db_path)
    await database.connect()
    await database.init_schema()

    project_store = ProjectMemoryStore(database)
    longterm_store = LongTermMemoryStore(database)
    registry = ToolRegistry()
    registry.register(
        create_memory_write_tool(
            project_store,
            longterm_store,
            "demo-user",
            "demo-project",
        )
    )

    result = await registry.dispatch(
        "memory_write",
        {
            "memory_level": "project",
            "key": "tech_stack",
            "value": "LangGraph + LiteLLM + FastAPI",
            "reason": "confirmed technical direction",
        },
    )

    assert result["ok"] is True
    stored = await project_store.get_project_context("demo-project")
    assert stored["tech_stack"] == "LangGraph + LiteLLM + FastAPI"
    event_rows = await database.fetchall(
        """
        SELECT project_id, memory_level, key, value_json, reason
        FROM memory_events
        ORDER BY id
        """
    )
    assert event_rows == [
        {
            "project_id": "demo-project",
            "memory_level": "project",
            "key": "tech_stack",
            "value_json": '"LangGraph + LiteLLM + FastAPI"',
            "reason": "confirmed technical direction",
        }
    ]

    await database.close()
