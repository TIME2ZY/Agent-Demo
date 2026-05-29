import pytest

from memory.session import SessionMemory


class FakeClient:
    def __init__(self, results, stream_results=None):
        self._results = list(results)
        self._stream_results = list(stream_results or [])
        self.calls = []
        self.stream_calls = []

    async def chat(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        return self._results.pop(0)

    async def chat_stream(self, messages, tools=None, on_content_chunk=None):
        self.stream_calls.append({"messages": messages, "tools": tools})
        result = self._stream_results.pop(0)
        if result.type == "message" and result.content and on_content_chunk is not None:
            on_content_chunk(result.content)
        return result


class FakeProjectStore:
    def __init__(self, context=None):
        self._context = context

    async def get_project_context(self, project_id):
        return self._context


class FakeLongTermStore:
    def __init__(self, items=None):
        self._items = items or []

    async def list_memories(self, user_id):
        return self._items


class FakeRegistry:
    def __init__(self, tool_result):
        self._tool_result = tool_result
        self.calls = []

    def schemas(self):
        return [{"type": "function", "function": {"name": "memory_write"}}]

    async def dispatch(self, name, arguments):
        self.calls.append((name, arguments))
        if isinstance(self._tool_result, list):
            return self._tool_result.pop(0)
        return self._tool_result


class FailingThenSucceedingRegistry:
    def __init__(self):
        self.calls = []
        self._failed = False

    def schemas(self):
        return [{"type": "function", "function": {"name": "read_file"}}]

    async def dispatch(self, name, arguments):
        self.calls.append((name, arguments))
        if not self._failed:
            self._failed = True
            raise UnicodeDecodeError("utf-8", b"\xb5", 0, 1, "invalid start byte")
        return {"ok": True, "content": "safe now"}


class FakeResult:
    def __init__(
        self,
        result_type,
        content="",
        tool_call_id=None,
        tool_name=None,
        tool_args=None,
        reasoning_content=None,
    ):
        self.type = result_type
        self.content = content
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.reasoning_content = reasoning_content


@pytest.mark.asyncio
async def test_run_turn_returns_normal_message_when_no_tool_is_called():
    from agent.loop import AgentLoop

    session_memory = SessionMemory()
    loop = AgentLoop(
        settings=type(
            "Settings",
            (),
            {
                "max_tool_steps_per_turn": 1,
                "max_session_messages_in_prompt": 8,
                "max_longterm_items_in_prompt": 5,
            },
        )(),
        client=FakeClient(
            [FakeResult("message", content="Hello user", reasoning_content="Answer directly")]
        ),
        session_memory=session_memory,
        project_store=FakeProjectStore(),
        longterm_store=FakeLongTermStore(),
        tool_registry=FakeRegistry({"ok": True}),
    )

    result = await loop.run_turn("demo-user", "demo-project", "Hi")

    assert result.content == "Hello user"
    assert result.reasoning_content == "Answer directly"
    assert session_memory.recent_messages()[-1] == {
        "role": "assistant",
        "content": "Hello user",
        "reasoning_content": "Answer directly",
    }


@pytest.mark.asyncio
async def test_run_turn_handles_single_memory_write_then_returns_final_reply():
    from agent.loop import AgentLoop

    session_memory = SessionMemory()
    loop = AgentLoop(
        settings=type(
            "Settings",
            (),
            {
                "max_tool_steps_per_turn": 1,
                "max_session_messages_in_prompt": 8,
                "max_longterm_items_in_prompt": 5,
            },
        )(),
        client=FakeClient(
            [
                FakeResult(
                    "tool_call",
                    tool_call_id="call-1",
                    tool_name="memory_write",
                    tool_args={
                        "memory_level": "longterm",
                        "key": "response_style",
                        "value": "Prefer concise Chinese",
                        "reason": "explicit preference",
                    },
                ),
                FakeResult(
                    "message",
                    content="I will remember that preference.",
                    reasoning_content="Preference stored successfully",
                ),
            ]
        ),
        session_memory=session_memory,
        project_store=FakeProjectStore(),
        longterm_store=FakeLongTermStore(),
        tool_registry=FakeRegistry({"ok": True, "key": "response_style"}),
    )

    result = await loop.run_turn("demo-user", "demo-project", "Answer concisely")

    assert result.content == "I will remember that preference."
    assert result.reasoning_content == "Preference stored successfully"
    assert session_memory.recent_messages()[1]["tool_calls"][0]["id"] == "call-1"
    assert session_memory.recent_messages()[2]["tool_call_id"] == "call-1"
    assert session_memory.recent_messages()[-1] == {
        "role": "assistant",
        "content": "I will remember that preference.",
        "reasoning_content": "Preference stored successfully",
    }


@pytest.mark.asyncio
async def test_run_turn_handles_multiple_tool_calls_before_final_message():
    from agent.loop import AgentLoop

    session_memory = SessionMemory()
    registry = FakeRegistry(
        [
            {"ok": True, "key": "response_style"},
            {"ok": True, "path": "notes.txt"},
        ]
    )
    loop = AgentLoop(
        settings=type(
            "Settings",
            (),
            {
                "max_tool_steps_per_turn": 3,
                "max_session_messages_in_prompt": 8,
                "max_longterm_items_in_prompt": 5,
            },
        )(),
        client=FakeClient(
            [
                FakeResult(
                    "tool_call",
                    tool_call_id="call-1",
                    tool_name="memory_write",
                    tool_args={
                        "memory_level": "longterm",
                        "key": "response_style",
                        "value": "Prefer concise Chinese",
                        "reason": "explicit preference",
                    },
                ),
                FakeResult(
                    "tool_call",
                    tool_call_id="call-2",
                    tool_name="read_file",
                    tool_args={"path": "notes.txt"},
                ),
                FakeResult(
                    "message",
                    content="I checked the saved preference and the file.",
                    reasoning_content="Two tool calls completed",
                ),
            ]
        ),
        session_memory=session_memory,
        project_store=FakeProjectStore(),
        longterm_store=FakeLongTermStore(),
        tool_registry=registry,
    )

    result = await loop.run_turn("demo-user", "demo-project", "Check memory and file")

    assert result.type == "message"
    assert result.content == "I checked the saved preference and the file."
    assert registry.calls == [
        (
            "memory_write",
            {
                "memory_level": "longterm",
                "key": "response_style",
                "value": "Prefer concise Chinese",
                "reason": "explicit preference",
            },
        ),
        ("read_file", {"path": "notes.txt"}),
    ]
    assert session_memory.recent_messages()[-1] == {
        "role": "assistant",
        "content": "I checked the saved preference and the file.",
        "reasoning_content": "Two tool calls completed",
    }


@pytest.mark.asyncio
async def test_run_turn_raises_when_tool_call_limit_is_exceeded():
    from agent.loop import AgentLoop

    session_memory = SessionMemory()
    loop = AgentLoop(
        settings=type(
            "Settings",
            (),
            {
                "max_tool_steps_per_turn": 1,
                "max_session_messages_in_prompt": 8,
                "max_longterm_items_in_prompt": 5,
            },
        )(),
        client=FakeClient(
            [
                FakeResult(
                    "tool_call",
                    tool_call_id="call-1",
                    tool_name="memory_write",
                    tool_args={
                        "memory_level": "longterm",
                        "key": "response_style",
                        "value": "Prefer concise Chinese",
                        "reason": "explicit preference",
                    },
                ),
                FakeResult(
                    "tool_call",
                    tool_call_id="call-2",
                    tool_name="read_file",
                    tool_args={"path": "notes.txt"},
                ),
            ]
        ),
        session_memory=session_memory,
        project_store=FakeProjectStore(),
        longterm_store=FakeLongTermStore(),
        tool_registry=FakeRegistry({"ok": True}),
    )

    with pytest.raises(RuntimeError, match="Exceeded max tool steps per turn"):
        await loop.run_turn("demo-user", "demo-project", "Keep going")


@pytest.mark.asyncio
async def test_run_turn_passes_reasoning_content_back_after_tool_call():
    from agent.loop import AgentLoop

    session_memory = SessionMemory()
    client = FakeClient(
        [
            FakeResult(
                "tool_call",
                content="Remembering the preference first",
                tool_call_id="call-1",
                tool_name="memory_write",
                tool_args={
                    "memory_level": "longterm",
                    "key": "response_style",
                    "value": "concise Chinese",
                    "reason": "explicit preference",
                },
                reasoning_content="User asked me to save a stable preference",
            ),
            FakeResult(
                "message",
                content="I have saved it.",
                reasoning_content="Preference stored",
            ),
        ]
    )
    loop = AgentLoop(
        settings=type(
            "Settings",
            (),
            {
                "max_tool_steps_per_turn": 4,
                "max_session_messages_in_prompt": 8,
                "max_longterm_items_in_prompt": 5,
            },
        )(),
        client=client,
        session_memory=session_memory,
        project_store=FakeProjectStore(),
        longterm_store=FakeLongTermStore(),
        tool_registry=FakeRegistry({"ok": True}),
    )

    await loop.run_turn("demo-user", "demo-project", "Remember my preference")

    second_call_messages = client.calls[1]["messages"]
    assistant_tool_call_message = next(
        message for message in second_call_messages if message.get("role") == "assistant"
    )
    assert assistant_tool_call_message["content"] == "Remembering the preference first"
    assert (
        assistant_tool_call_message["reasoning_content"]
        == "User asked me to save a stable preference"
    )


@pytest.mark.asyncio
async def test_run_turn_streams_follow_up_message_after_tool_call():
    from agent.loop import AgentLoop

    streamed = []
    client = FakeClient(
        [
            FakeResult(
                "tool_call",
                tool_call_id="call-1",
                tool_name="memory_write",
                tool_args={
                    "memory_level": "longterm",
                    "key": "response_style",
                    "value": "Prefer concise Chinese",
                    "reason": "explicit preference",
                },
            ),
        ],
        stream_results=[
            FakeResult(
                "message",
                content="Streamed final reply",
                reasoning_content="Done after tool",
            )
        ],
    )
    loop = AgentLoop(
        settings=type(
            "Settings",
            (),
            {
                "max_tool_steps_per_turn": 4,
                "max_session_messages_in_prompt": 8,
                "max_longterm_items_in_prompt": 5,
            },
        )(),
        client=client,
        session_memory=SessionMemory(),
        project_store=FakeProjectStore(),
        longterm_store=FakeLongTermStore(),
        tool_registry=FakeRegistry({"ok": True}),
    )

    result = await loop.run_turn(
        "demo-user",
        "demo-project",
        "Remember the preference",
        on_stream=streamed.append,
    )

    assert result.content == "Streamed final reply"
    assert streamed == ["Streamed final reply"]
    assert len(client.calls) == 1
    assert len(client.stream_calls) == 1


@pytest.mark.asyncio
async def test_run_turn_does_not_stream_direct_first_reply():
    from agent.loop import AgentLoop

    streamed = []
    client = FakeClient(
        [
            FakeResult(
                "message",
                content="Direct answer",
                reasoning_content="No tools needed",
            )
        ]
    )
    loop = AgentLoop(
        settings=type(
            "Settings",
            (),
            {
                "max_tool_steps_per_turn": 4,
                "max_session_messages_in_prompt": 8,
                "max_longterm_items_in_prompt": 5,
            },
        )(),
        client=client,
        session_memory=SessionMemory(),
        project_store=FakeProjectStore(),
        longterm_store=FakeLongTermStore(),
        tool_registry=FakeRegistry({"ok": True}),
    )

    result = await loop.run_turn(
        "demo-user",
        "demo-project",
        "Say hello",
        on_stream=streamed.append,
    )

    assert result.content == "Direct answer"
    assert streamed == []
    assert len(client.calls) == 1
    assert len(client.stream_calls) == 0


@pytest.mark.asyncio
async def test_run_turn_streams_only_final_message_content_across_multiple_tool_calls():
    from agent.loop import AgentLoop

    streamed = []
    client = FakeClient(
        [
            FakeResult(
                "tool_call",
                tool_call_id="call-1",
                tool_name="memory_write",
                tool_args={
                    "memory_level": "longterm",
                    "key": "response_style",
                    "value": "Prefer concise Chinese",
                    "reason": "explicit preference",
                },
            ),
        ],
        stream_results=[
            FakeResult(
                "tool_call",
                tool_call_id="call-2",
                tool_name="read_file",
                tool_args={"path": "notes.txt"},
            ),
            FakeResult(
                "message",
                content="Final streamed answer",
                reasoning_content="All done",
            ),
        ],
    )
    loop = AgentLoop(
        settings=type(
            "Settings",
            (),
            {
                "max_tool_steps_per_turn": 4,
                "max_session_messages_in_prompt": 8,
                "max_longterm_items_in_prompt": 5,
            },
        )(),
        client=client,
        session_memory=SessionMemory(),
        project_store=FakeProjectStore(),
        longterm_store=FakeLongTermStore(),
        tool_registry=FakeRegistry([{"ok": True}, {"ok": True, "path": "notes.txt"}]),
    )

    result = await loop.run_turn(
        "demo-user",
        "demo-project",
        "Do two tools and then answer",
        on_stream=streamed.append,
    )

    assert result.content == "Final streamed answer"
    assert streamed == ["Final streamed answer"]
    assert len(client.calls) == 1
    assert len(client.stream_calls) == 2


@pytest.mark.asyncio
async def test_run_turn_does_not_leave_orphan_tool_call_after_tool_failure():
    from agent.loop import AgentLoop

    session_memory = SessionMemory()
    client = FakeClient(
        [
            FakeResult(
                "tool_call",
                tool_call_id="call-1",
                tool_name="read_file",
                tool_args={"path": "bad-gbk.txt"},
            ),
            FakeResult(
                "message",
                content="Recovered cleanly",
                reasoning_content="No orphan tool calls remained",
            ),
        ]
    )
    registry = FailingThenSucceedingRegistry()
    loop = AgentLoop(
        settings=type(
            "Settings",
            (),
            {
                "max_tool_steps_per_turn": 4,
                "max_session_messages_in_prompt": 8,
                "max_longterm_items_in_prompt": 5,
            },
        )(),
        client=client,
        session_memory=session_memory,
        project_store=FakeProjectStore(),
        longterm_store=FakeLongTermStore(),
        tool_registry=registry,
    )

    with pytest.raises(UnicodeDecodeError):
        await loop.run_turn("demo-user", "demo-project", "Read the bad file")

    assert session_memory.recent_messages() == [
        {"role": "user", "content": "Read the bad file"}
    ]

    result = await loop.run_turn("demo-user", "demo-project", "Say hello instead")

    assert result.content == "Recovered cleanly"
    assert client.calls[1]["messages"] == [
        {"role": "system", "content": client.calls[1]["messages"][0]["content"]},
        {"role": "user", "content": "Read the bad file"},
        {"role": "user", "content": "Say hello instead"},
    ]
