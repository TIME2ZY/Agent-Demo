import pytest

from storage.db import Database


@pytest.mark.asyncio
async def test_database_initializes_required_tables(db_path):
    database = Database(db_path)
    await database.connect()
    await database.init_schema()

    rows = await database.fetchall(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    )

    table_names = {row["name"] for row in rows}
    assert "project_memory" in table_names
    assert "longterm_memory" in table_names
    assert "memory_events" in table_names

    await database.close()


@pytest.mark.asyncio
async def test_database_execute_and_fetch_helpers_work(db_path):
    database = Database(db_path)
    await database.connect()
    await database.init_schema()

    await database.execute(
        "INSERT INTO project_memory (project_id, context_json, updated_at) VALUES (?, ?, ?)",
        ("demo", "{}", "2026-05-28T21:00:00+08:00"),
    )

    row = await database.fetchone(
        "SELECT project_id, context_json FROM project_memory WHERE project_id = ?",
        ("demo",),
    )

    assert row["project_id"] == "demo"
    assert row["context_json"] == "{}"

    await database.close()
