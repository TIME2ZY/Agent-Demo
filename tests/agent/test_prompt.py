from types import SimpleNamespace

from agent.prompt import build_messages, format_longterm_memories, format_project_memory


def test_format_project_memory_creates_readable_sections():
    context = {
        "project_goal": "Build a memory-first agent",
        "constraints": ["CLI only", "No bash tool"],
        "decisions": ["Use SQLite"],
        "current_status": ["Prompt design approved"],
        "open_questions": [],
    }

    formatted = format_project_memory(context)

    assert "Project goal:" in formatted
    assert "- CLI only" in formatted
    assert "- Use SQLite" in formatted
    assert "Open questions:" in formatted


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
    assert "working_style" not in messages[0]["content"]
    assert messages[1:] == [
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
        {"role": "user", "content": "Current question"},
    ]
