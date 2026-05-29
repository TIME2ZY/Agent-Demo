import pytest

from memory.longterm import LongTermMemoryStore
from memory.project import ProjectMemoryStore
from storage.db import Database
from tools.builtin.memory_write import create_memory_write_tool
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
