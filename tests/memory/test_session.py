from memory.session import SessionMemory


def test_session_memory_tracks_recent_messages():
    session = SessionMemory()
    session.add_user_message("Hello")
    session.add_assistant_message("Hi there")

    assert session.recent_messages() == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]


def test_session_memory_can_store_tool_messages():
    session = SessionMemory()
    session.add_tool_call("call-1", "memory_write", {"key": "response_style"})
    session.add_tool_result("call-1", "memory_write", {"ok": True})

    messages = session.recent_messages()
    assert messages[0]["role"] == "assistant"
    assert messages[0]["tool_calls"][0]["id"] == "call-1"
    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_call_id"] == "call-1"


def test_session_memory_preserves_assistant_reasoning_fields():
    session = SessionMemory()
    session.add_assistant_message("Hi there", reasoning_content="Simple greeting")
    session.add_tool_call(
        "call-1",
        "read_file",
        {"path": "notes.txt"},
        content="先读文件",
        reasoning_content="需要文件内容",
    )

    messages = session.recent_messages()
    assert messages[0]["reasoning_content"] == "Simple greeting"
    assert messages[1]["content"] == "先读文件"
    assert messages[1]["reasoning_content"] == "需要文件内容"
