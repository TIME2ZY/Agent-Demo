from pathlib import Path
from typing import Any, Iterable

import aiosqlite


class Database:
    def __init__(self, db_path: Path):
        self._db_path = Path(db_path)
        self._connection: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        if self._connection is None:
            self._connection = await aiosqlite.connect(self._db_path)
            self._connection.row_factory = aiosqlite.Row

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def init_schema(self) -> None:
        await self.execute(
            """
            CREATE TABLE IF NOT EXISTS project_memory (
                project_id TEXT PRIMARY KEY,
                context_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        await self.execute(
            """
            CREATE TABLE IF NOT EXISTS longterm_memory (
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, key)
            )
            """
        )
        await self.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                project_id TEXT,
                memory_level TEXT NOT NULL,
                key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                reason TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

    async def execute(self, sql: str, params: Iterable[Any] = ()) -> None:
        connection = self._require_connection()
        await connection.execute(sql, tuple(params))
        await connection.commit()

    async def fetchone(
        self, sql: str, params: Iterable[Any] = ()
    ) -> dict[str, Any] | None:
        connection = self._require_connection()
        async with connection.execute(sql, tuple(params)) as cursor:
            row = await cursor.fetchone()
        return dict(row) if row is not None else None

    async def fetchall(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        connection = self._require_connection()
        async with connection.execute(sql, tuple(params)) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    def _require_connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("Database connection has not been initialized")
        return self._connection
