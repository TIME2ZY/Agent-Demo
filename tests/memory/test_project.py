import pytest

from memory.project import ProjectMemoryStore
from storage.db import Database


@pytest.mark.asyncio
async def test_project_memory_store_saves_and_loads_context(db_path):
    database = Database(db_path)
    await database.connect()
    await database.init_schema()

    store = ProjectMemoryStore(database)
    context = {
        "project_goal": "Build a memory-first agent",
        "tech_stack": "LangGraph + LiteLLM",
        "constraints": ["CLI only"],
        "decisions": [],
        "current_status": [],
        "open_questions": [],
    }

    await store.save_project_context("demo-project", context)
    loaded = await store.get_project_context("demo-project")

    assert loaded["project_goal"] == "Build a memory-first agent"
    assert loaded["tech_stack"] == "LangGraph + LiteLLM"
    assert loaded["constraints"] == ["CLI only"]

    await database.close()


@pytest.mark.asyncio
async def test_project_memory_store_merges_list_based_fields(db_path):
    database = Database(db_path)
    await database.connect()
    await database.init_schema()

    store = ProjectMemoryStore(database)
    await store.merge_project_memory(
        "demo-project",
        "constraints",
        "No bash tool",
        "confirmed by user",
    )
    await store.merge_project_memory(
        "demo-project",
        "constraints",
        "CLI only",
        "confirmed by user",
    )

    loaded = await store.get_project_context("demo-project")
    assert loaded["constraints"] == ["No bash tool", "CLI only"]

    await database.close()
