import pytest

from memory.session import SessionMemory


class FakeClient:
    def __init__(self, results):
        self._results = list(results)

    async def chat(self, messages, tools=None):
        return self._results.pop(0)


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

    def schemas(self):
        return [{"type": "function", "function": {"name": "memory_write"}}]

    async def dispatch(self, name, arguments):
        return self._tool_result


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
    }
