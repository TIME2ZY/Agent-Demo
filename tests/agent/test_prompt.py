from types import SimpleNamespace

from agent.prompt import (
    build_messages,
    build_system_prompt,
    format_longterm_memories,
    format_project_memory,
)


def test_format_project_memory_creates_readable_sections():
    context = {
        "project_goal": "Build a memory-first agent",
        "tech_stack": "LangGraph + LiteLLM + FastAPI",
        "constraints": ["CLI only", "No bash tool"],
        "decisions": ["Use SQLite"],
        "current_status": ["Prompt design approved"],
        "open_questions": [],
    }

    formatted = format_project_memory(context)

    assert "Project goal:" in formatted
    assert "Selected tech stack:" in formatted
    assert "LangGraph + LiteLLM + FastAPI" in formatted
    assert "- CLI only" in formatted
    assert "- Use SQLite" in formatted
    assert "Open questions:" in formatted


def test_build_system_prompt_includes_identity_and_memory_schema_rules():
    prompt = build_system_prompt()

    assert "not the official DeepSeek product" in prompt
    assert "project_goal" in prompt
    assert "tech_stack" in prompt
    assert "current_status" in prompt
    assert "user_name" in prompt
    assert "user_role" in prompt
    assert "language_preference" in prompt
    assert "read_file" in prompt
    assert "write_file" in prompt
    assert "run_shell" in prompt
    assert "web_search" in prompt
    assert "must use the corresponding tool" in prompt
    assert "Never claim that you read" in prompt


def test_format_longterm_memories_respects_item_limit():
    formatted = format_longterm_memories(
        [
            {"key": "response_style", "value": "Prefer concise Chinese"},
            {"key": "working_style", "value": "Architecture first"},
        ],
        max_items=1,
    )

    assert "response_style" in formatted
    assert "working_style" not in formatted


def test_build_messages_includes_memory_context_and_trimmed_session_history():
    settings = SimpleNamespace(
        max_longterm_items_in_prompt=1,
        max_session_messages_in_prompt=2,
    )

    messages = build_messages(
        settings=settings,
        longterm_memories=[
            {"key": "response_style", "value": "Prefer concise Chinese"},
            {"key": "working_style", "value": "Architecture first"},
        ],
        project_context={
            "project_goal": "Build a memory-first agent",
            "constraints": [],
            "decisions": [],
            "current_status": [],
            "open_questions": [],
        },
        session_messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ],
        user_message="Current question",
    )

    assert messages[0]["role"] == "system"
    assert "response_style" in messages[0]["content"]
    assert "- response_style: Prefer concise Chinese" in messages[0]["content"]
    assert "- working_style: Architecture first" not in messages[0]["content"]
    assert messages[1:] == [
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
        {"role": "user", "content": "Current question"},
    ]


def test_build_messages_does_not_duplicate_current_user_message_when_already_in_session():
    settings = SimpleNamespace(
        max_longterm_items_in_prompt=1,
        max_session_messages_in_prompt=4,
    )

    messages = build_messages(
        settings=settings,
        longterm_memories=[],
        project_context=None,
        session_messages=[
            {"role": "user", "content": "Current question"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"a.txt"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "name": "read_file",
                "tool_call_id": "call-1",
                "content": '{"ok": true}',
            },
        ],
        user_message="Current question",
    )

    user_messages = [message for message in messages if message.get("role") == "user"]
    assert user_messages == [{"role": "user", "content": "Current question"}]


def test_build_messages_keeps_tool_result_paired_with_preceding_tool_call_when_trimming():
    settings = SimpleNamespace(
        max_longterm_items_in_prompt=1,
        max_session_messages_in_prompt=4,
    )

    messages = build_messages(
        settings=settings,
        longterm_memories=[],
        project_context=None,
        session_messages=[
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer", "reasoning_content": "done"},
            {"role": "user", "content": "Current question"},
            {
                "role": "assistant",
                "content": "Reading file",
                "reasoning_content": "Need file contents",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"a.txt"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "name": "read_file",
                "tool_call_id": "call-1",
                "content": '{"ok": true}',
            },
        ],
        user_message="Current question",
    )

    assistant_index = next(
        index for index, message in enumerate(messages) if message.get("tool_calls")
    )
    assert messages[assistant_index]["content"] == "Reading file"
    assert messages[assistant_index]["reasoning_content"] == "Need file contents"
    assert messages[assistant_index + 1]["role"] == "tool"
    assert messages[assistant_index + 1]["tool_call_id"] == "call-1"
