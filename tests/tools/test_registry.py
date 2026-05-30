import sys

import pytest

from memory.longterm import LongTermMemoryStore
from memory.project import ProjectMemoryStore
from storage.db import Database
from tools.builtin.file_tools import create_read_file_tool, create_write_file_tool
from tools.builtin.memory_write import (
    ALLOWED_PROJECT_KEYS,
    ALLOWED_LONGTERM_KEYS,
    create_memory_write_tool,
    validate_memory_write_args,
)
from tools.builtin.shell_tool import create_run_shell_tool
from tools.builtin.web_search import create_web_search_tool
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


def test_registry_exposes_new_tool_schemas():
    async def fake_fetcher(query: str, limit: int) -> list[dict[str, str]]:
        return [{"title": query, "url": "https://example.com", "snippet": f"top {limit}"}]

    registry = ToolRegistry()
    registry.register(create_read_file_tool())
    registry.register(create_write_file_tool())
    registry.register(create_run_shell_tool())
    registry.register(create_web_search_tool(fetcher=fake_fetcher))

    schema_names = {
        item["function"]["name"]
        for item in registry.schemas()
    }

    assert {"read_file", "write_file", "run_shell", "web_search"} <= schema_names


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


@pytest.mark.asyncio
async def test_read_file_tool_returns_file_contents(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("hello", encoding="utf-8")

    registry = ToolRegistry()
    registry.register(create_read_file_tool())

    result = await registry.dispatch("read_file", {"path": str(target)})

    assert result["ok"] is True
    assert result["content"] == "hello"
    assert result["path"] == str(target.resolve())


@pytest.mark.asyncio
async def test_read_file_tool_falls_back_to_gbk_on_windows_text_files(tmp_path):
    target = tmp_path / "notes-gbk.txt"
    target.write_bytes("第一行：GBK".encode("gb18030"))

    registry = ToolRegistry()
    registry.register(create_read_file_tool())

    result = await registry.dispatch("read_file", {"path": str(target)})

    assert result["ok"] is True
    assert result["content"] == "第一行：GBK"


@pytest.mark.asyncio
async def test_write_file_tool_writes_and_appends(tmp_path):
    target = tmp_path / "nested" / "notes.txt"

    registry = ToolRegistry()
    registry.register(create_write_file_tool())

    first_result = await registry.dispatch(
        "write_file",
        {"path": str(target), "content": "hello"},
    )
    second_result = await registry.dispatch(
        "write_file",
        {"path": str(target), "content": " world", "append": True},
    )

    assert first_result["ok"] is True
    assert second_result["ok"] is True
    assert target.read_text(encoding="utf-8") == "hello world"


@pytest.mark.asyncio
async def test_run_shell_tool_returns_stdout():
    registry = ToolRegistry()
    registry.register(create_run_shell_tool())

    command = f'"{sys.executable}" -c "print(\'ok\')"'
    result = await registry.dispatch("run_shell", {"command": command})

    assert result["ok"] is True
    assert result["returncode"] == 0
    assert result["stdout"].strip() == "ok"


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell-specific behavior")
async def test_run_shell_tool_executes_powershell_commands_on_windows():
    registry = ToolRegistry()
    registry.register(create_run_shell_tool())

    result = await registry.dispatch("run_shell", {"command": "Write-Output hello"})

    assert result["ok"] is True
    assert result["returncode"] == 0
    assert result["stdout"].strip() == "hello"


@pytest.mark.asyncio
async def test_run_shell_tool_returns_nonzero_exit_code():
    registry = ToolRegistry()
    registry.register(create_run_shell_tool())

    command = f'"{sys.executable}" -c "import sys; sys.exit(3)"'
    result = await registry.dispatch("run_shell", {"command": command})

    assert result["ok"] is False
    assert result["returncode"] == 3


@pytest.mark.asyncio
async def test_web_search_tool_normalizes_fetcher_results():
    async def fake_fetcher(query: str, limit: int) -> list[dict[str, str]]:
        return [
            {
                "title": f"{query} title",
                "url": "https://example.com",
                "snippet": f"limit={limit}",
            }
        ]

    registry = ToolRegistry()
    registry.register(create_web_search_tool(fetcher=fake_fetcher))

    result = await registry.dispatch("web_search", {"query": "example"})

    assert result["ok"] is True
    assert result["query"] == "example"
    assert result["results"] == [
        {
            "title": "example title",
            "url": "https://example.com",
            "snippet": "limit=5",
        }
    ]


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
