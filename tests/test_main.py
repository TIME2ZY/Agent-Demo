import logging
from io import BytesIO, TextIOWrapper

import pytest

from llm.client import LLMResult
from memory.longterm import LongTermMemoryStore
from memory.project import ProjectMemoryStore
from storage.db import Database
from main import (
    configure_logging,
    format_turn_error_message,
    handle_local_command,
    print_turn_output,
    write_output_line,
)


def test_print_turn_output_shows_reasoning_and_reply_by_default(capsys):
    result = LLMResult(
        type="message",
        content="Final answer",
        reasoning_content="Internal reasoning",
        tool_call_id=None,
        tool_name=None,
        tool_args=None,
        raw_response={},
    )

    print_turn_output(result)

    captured = capsys.readouterr()
    assert captured.out == "Thinking: Internal reasoning\nAssistant: Final answer\n"
    assert captured.err == ""


def test_print_turn_output_skips_empty_reasoning(capsys):
    result = LLMResult(
        type="message",
        content="Final answer",
        reasoning_content=None,
        tool_call_id=None,
        tool_name=None,
        tool_args=None,
        raw_response={},
    )

    print_turn_output(result)

    captured = capsys.readouterr()
    assert captured.out == "Assistant: Final answer\n"
    assert captured.err == ""


def test_configure_logging_defaults_to_quiet_mode():
    configure_logging(debug=False)

    assert logging.getLogger().getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING


def test_write_output_line_replaces_unencodable_characters():
    buffer = BytesIO()
    stream = TextIOWrapper(buffer, encoding="gbk", errors="strict")

    write_output_line("Assistant: hello 😊", stream=stream)
    stream.flush()

    assert buffer.getvalue().decode("gbk") == "Assistant: hello ?\n"


def test_format_turn_error_message_points_users_to_debug_mode():
    assert format_turn_error_message(debug=False) == (
        "Assistant: 抱歉，这一轮处理失败了。请重试；如果需要排查，请使用 --debug 重新启动。"
    )


@pytest.mark.asyncio
async def test_handle_local_command_returns_memory_snapshot(db_path):
    database = Database(db_path)
    await database.connect()
    await database.init_schema()

    project_store = ProjectMemoryStore(database)
    longterm_store = LongTermMemoryStore(database)
    await longterm_store.set_memory("demo-user", "user_name", "张雨", "seed")
    await project_store.save_project_context(
        "demo-project",
        {
            "project_goal": "Build a memory-first agent",
            "tech_stack": "LangGraph + LiteLLM",
            "constraints": ["CLI only"],
            "decisions": [],
            "current_status": [],
            "open_questions": [],
            "major_milestones": [],
        },
    )

    output = await handle_local_command(
        "/memory",
        user_id="demo-user",
        project_id="demo-project",
        database=database,
        project_store=project_store,
        longterm_store=longterm_store,
    )

    assert output is not None
    assert "Assistant: Current memory snapshot" in output
    assert "user_name: 张雨" in output
    assert "project_goal: Build a memory-first agent" in output
    assert "tech_stack: LangGraph + LiteLLM" in output

    await database.close()


@pytest.mark.asyncio
async def test_handle_local_command_returns_recent_events(db_path):
    database = Database(db_path)
    await database.connect()
    await database.init_schema()
    await database.execute(
        """
        INSERT INTO memory_events (
            user_id, project_id, memory_level, key, value_json, reason, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "demo-user",
            "demo-project",
            "project",
            "tech_stack",
            '"LangGraph + LiteLLM"',
            "confirmed stack",
            "2026-05-29T04:45:36.932766+00:00",
        ),
    )

    output = await handle_local_command(
        "/events",
        user_id="demo-user",
        project_id="demo-project",
        database=database,
        project_store=ProjectMemoryStore(database),
        longterm_store=LongTermMemoryStore(database),
    )

    assert output is not None
    assert "Assistant: Recent memory events" in output
    assert "project | tech_stack | confirmed stack" in output

    await database.close()
