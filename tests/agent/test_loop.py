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
    def __init__(self, result_type, content="", tool_name=None, tool_args=None):
        self.type = result_type
        self.content = content
        self.tool_name = tool_name
        self.tool_args = tool_args


@pytest.mark.asyncio
async def test_run_turn_returns_normal_message_when_no_tool_is_called():
    from agent.loop import AgentLoop

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
        client=FakeClient([FakeResult("message", content="Hello user")]),
        session_memory=SessionMemory(),
        project_store=FakeProjectStore(),
        longterm_store=FakeLongTermStore(),
        tool_registry=FakeRegistry({"ok": True}),
    )

    reply = await loop.run_turn("demo-user", "demo-project", "Hi")

    assert reply == "Hello user"


@pytest.mark.asyncio
async def test_run_turn_handles_single_memory_write_then_returns_final_reply():
    from agent.loop import AgentLoop

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
                    tool_name="memory_write",
                    tool_args={
                        "memory_level": "longterm",
                        "key": "response_style",
                        "value": "Prefer concise Chinese",
                        "reason": "explicit preference",
                    },
                ),
                FakeResult("message", content="I will remember that preference."),
            ]
        ),
        session_memory=SessionMemory(),
        project_store=FakeProjectStore(),
        longterm_store=FakeLongTermStore(),
        tool_registry=FakeRegistry({"ok": True, "key": "response_style"}),
    )

    reply = await loop.run_turn("demo-user", "demo-project", "Answer concisely")

    assert reply == "I will remember that preference."
