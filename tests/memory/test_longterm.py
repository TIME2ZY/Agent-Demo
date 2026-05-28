import pytest

from memory.longterm import LongTermMemoryStore
from storage.db import Database


@pytest.mark.asyncio
async def test_longterm_store_sets_and_lists_memories(db_path):
    database = Database(db_path)
    await database.connect()
    await database.init_schema()

    store = LongTermMemoryStore(database)
    await store.set_memory(
        "demo-user",
        "response_style",
        "Prefer concise Chinese",
        "explicit user preference",
    )

    items = await store.list_memories("demo-user")

    assert items == [
        {
            "key": "response_style",
            "value": "Prefer concise Chinese",
            "updated_at": items[0]["updated_at"],
        }
    ]

    await database.close()


@pytest.mark.asyncio
async def test_longterm_store_replaces_existing_value_by_key(db_path):
    database = Database(db_path)
    await database.connect()
    await database.init_schema()

    store = LongTermMemoryStore(database)
    await store.set_memory("demo-user", "response_style", "Prefer concise Chinese", "first")
    await store.set_memory("demo-user", "response_style", "Prefer direct Chinese", "second")

    item = await store.get_memory("demo-user", "response_style")
    assert item["value"] == "Prefer direct Chinese"

    await database.close()
