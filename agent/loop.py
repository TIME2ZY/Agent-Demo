from typing import Any, Callable

from agent.prompt import build_messages
from llm.client import LLMResult


class AgentLoop:
    def __init__(
        self,
        settings: Any,
        client: Any,
        session_memory: Any,
        project_store: Any,
        longterm_store: Any,
        tool_registry: Any,
    ) -> None:
        self._settings = settings
        self._client = client
        self._session_memory = session_memory
        self._project_store = project_store
        self._longterm_store = longterm_store
        self._tool_registry = tool_registry

    async def run_turn(
        self,
        user_id: str,
        project_id: str,
        user_message: str,
        on_stream: Callable[[str], None] | None = None,
    ) -> LLMResult:
        self._session_memory.add_user_message(user_message)

        tool_steps = 0
        use_streaming = False

        while True:
            session_history_limit = getattr(self._settings, "session_history_limit", None)
            if session_history_limit is not None:
                session_history_limit += 1

            project_context = await self._project_store.get_project_context(project_id)
            longterm_memories = await self._longterm_store.list_memories(user_id)
            prompt_messages = build_messages(
                settings=self._settings,
                longterm_memories=longterm_memories,
                project_context=project_context,
                session_messages=self._session_memory.recent_messages(session_history_limit),
                user_message=user_message,
            )

            if use_streaming and on_stream is not None:
                result = await self._client.chat_stream(
                    prompt_messages,
                    tools=self._tool_registry.schemas(),
                    on_content_chunk=on_stream,
                )
            else:
                result = await self._client.chat(
                    prompt_messages,
                    tools=self._tool_registry.schemas(),
                )

            if result.type == "message":
                self._session_memory.add_assistant_message(
                    result.content,
                    reasoning_content=result.reasoning_content,
                )
                return result

            if self._settings.max_tool_steps_per_turn < 1:
                raise RuntimeError("Tool calls are disabled for this run")
            tool_calls = getattr(result, "tool_calls", None) or [
                {
                    "id": result.tool_call_id,
                    "name": result.tool_name,
                    "arguments": result.tool_args,
                }
            ]
            if any(not tool_call["id"] for tool_call in tool_calls):
                raise RuntimeError("Tool call result is missing tool_call_id")
            if tool_steps + len(tool_calls) > self._settings.max_tool_steps_per_turn:
                raise RuntimeError("Exceeded max tool steps per turn")

            tool_results = []
            for tool_call in tool_calls:
                tool_results.append(
                    await self._tool_registry.dispatch(
                        tool_call["name"],
                        tool_call["arguments"],
                    )
                )

            self._session_memory.add_tool_calls(
                tool_calls,
                content=result.content,
                reasoning_content=result.reasoning_content,
            )
            for tool_call, tool_result in zip(tool_calls, tool_results):
                self._session_memory.add_tool_result(
                    tool_call["id"],
                    tool_call["name"],
                    tool_result,
                )
            tool_steps += len(tool_calls)
            use_streaming = True
