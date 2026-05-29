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
            project_context = await self._project_store.get_project_context(project_id)
            longterm_memories = await self._longterm_store.list_memories(user_id)
            prompt_messages = build_messages(
                settings=self._settings,
                longterm_memories=longterm_memories,
                project_context=project_context,
                session_messages=self._session_memory.recent_messages(),
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
            if tool_steps >= self._settings.max_tool_steps_per_turn:
                raise RuntimeError("Exceeded max tool steps per turn")
            if not result.tool_call_id:
                raise RuntimeError("Tool call result is missing tool_call_id")

            tool_result = await self._tool_registry.dispatch(
                result.tool_name,
                result.tool_args,
            )
            self._session_memory.add_tool_call(
                result.tool_call_id,
                result.tool_name,
                result.tool_args,
                content=result.content,
                reasoning_content=result.reasoning_content,
            )
            self._session_memory.add_tool_result(
                result.tool_call_id,
                result.tool_name,
                tool_result,
            )
            tool_steps += 1
            use_streaming = True
