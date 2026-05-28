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
    session.add_tool_call("memory_write", {"key": "response_style"})
    session.add_tool_result("memory_write", {"ok": True})

    messages = session.recent_messages()
    assert messages[0]["role"] == "assistant"
    assert messages[1]["role"] == "tool"
