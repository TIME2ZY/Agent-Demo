from datetime import datetime, timezone
import json
from typing import Any

from storage.db import Database


LIST_FIELDS = {
    "constraints",
    "decisions",
    "current_status",
    "open_questions",
    "major_milestones",
}


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_context() -> dict[str, Any]:
    return {
        "project_goal": "",
        "tech_stack": "",
        "constraints": [],
        "decisions": [],
        "current_status": [],
        "open_questions": [],
        "major_milestones": [],
        "updated_at": _utc_timestamp(),
    }


class ProjectMemoryStore:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def get_project_context(self, project_id: str) -> dict[str, Any] | None:
        row = await self._database.fetchone(
            "SELECT context_json FROM project_memory WHERE project_id = ?",
            (project_id,),
        )
        if row is None:
            return None

        return json.loads(row["context_json"])

    async def save_project_context(self, project_id: str, context: dict[str, Any]) -> None:
        context["updated_at"] = _utc_timestamp()
        await self._database.execute(
            """
            INSERT INTO project_memory (project_id, context_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(project_id) DO UPDATE SET
                context_json = excluded.context_json,
                updated_at = excluded.updated_at
            """,
            (project_id, json.dumps(context, ensure_ascii=False), context["updated_at"]),
        )

    async def merge_project_memory(
        self,
        project_id: str,
        key: str,
        value: Any,
        reason: str | None = None,
        replace: bool = False,
    ) -> dict[str, Any]:
        del reason
        context = await self.get_project_context(project_id) or _empty_context()

        if key in LIST_FIELDS:
            values = value if isinstance(value, list) else [value]
            if replace:
                context[key] = list(values)
            else:
                for item in values:
                    if item not in context[key]:
                        context[key].append(item)
        else:
            context[key] = value

        await self.save_project_context(project_id, context)
        return context
