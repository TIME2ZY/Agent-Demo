from datetime import datetime, timezone
import json
from typing import Any

from storage.db import Database


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class LongTermMemoryStore:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def list_memories(self, user_id: str) -> list[dict[str, Any]]:
        rows = await self._database.fetchall(
            """
            SELECT key, value_json, updated_at
            FROM longterm_memory
            WHERE user_id = ?
            ORDER BY key
            """,
            (user_id,),
        )
        return [
            {
                "key": row["key"],
                "value": json.loads(row["value_json"]),
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    async def get_memory(self, user_id: str, key: str) -> dict[str, Any] | None:
        row = await self._database.fetchone(
            """
            SELECT key, value_json, updated_at
            FROM longterm_memory
            WHERE user_id = ? AND key = ?
            """,
            (user_id, key),
        )
        if row is None:
            return None

        return {
            "key": row["key"],
            "value": json.loads(row["value_json"]),
            "updated_at": row["updated_at"],
        }

    async def set_memory(
        self,
        user_id: str,
        key: str,
        value: Any,
        reason: str | None = None,
    ) -> None:
        del reason
        updated_at = _utc_timestamp()
        await self._database.execute(
            """
            INSERT INTO longterm_memory (user_id, key, value_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = excluded.updated_at
            """,
            (user_id, key, json.dumps(value, ensure_ascii=False), updated_at),
        )
