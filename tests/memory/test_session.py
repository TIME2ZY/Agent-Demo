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
        content="Read the file first",
        reasoning_content="Need the file contents",
    )

    messages = session.recent_messages()
    assert messages[0]["reasoning_content"] == "Simple greeting"
    assert messages[1]["content"] == "Read the file first"
    assert messages[1]["reasoning_content"] == "Need the file contents"


def test_session_memory_can_store_multiple_tool_calls_in_one_message():
    session = SessionMemory()
    session.add_tool_calls(
        [
            {"id": "call-1", "name": "read_file", "arguments": {"path": "notes.txt"}},
            {"id": "call-2", "name": "write_file", "arguments": {"path": "out.txt"}},
        ],
        content="Read and write",
        reasoning_content="Need both operations",
    )

    messages = session.recent_messages()
    assert messages == [
        {
            "role": "assistant",
            "content": "Read and write",
            "reasoning_content": "Need both operations",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "notes.txt"}',
                    },
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path": "out.txt"}',
                    },
                },
            ],
        }
    ]


def test_session_memory_drops_old_messages_when_limit_is_reached():
    session = SessionMemory(max_messages=3)
    session.add_user_message("u1")
    session.add_assistant_message("a1")
    session.add_user_message("u2")
    session.add_assistant_message("a2")

    assert session.recent_messages() == [
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
