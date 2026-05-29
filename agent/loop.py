from typing import Any

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

    async def run_turn(self, user_id: str, project_id: str, user_message: str) -> LLMResult:
        self._session_memory.add_user_message(user_message)

        project_context = await self._project_store.get_project_context(project_id)
        longterm_memories = await self._longterm_store.list_memories(user_id)
        prompt_messages = build_messages(
            settings=self._settings,
            longterm_memories=longterm_memories,
            project_context=project_context,
            session_messages=self._session_memory.recent_messages(
                self._settings.max_session_messages_in_prompt
            ),
            user_message=user_message,
        )

        result = await self._client.chat(
            prompt_messages,
            tools=self._tool_registry.schemas(),
        )

        if result.type == "message":
            self._session_memory.add_assistant_message(result.content)
            return result

        if self._settings.max_tool_steps_per_turn < 1:
            raise RuntimeError("Tool calls are disabled for this run")

        self._session_memory.add_tool_call(result.tool_name, result.tool_args)
        tool_result = await self._tool_registry.dispatch(result.tool_name, result.tool_args)
        self._session_memory.add_tool_result(result.tool_name, tool_result)

        follow_up_messages = build_messages(
            settings=self._settings,
            longterm_memories=await self._longterm_store.list_memories(user_id),
            project_context=await self._project_store.get_project_context(project_id),
            session_messages=self._session_memory.recent_messages(
                self._settings.max_session_messages_in_prompt
            ),
            user_message=user_message,
        )
        final_result = await self._client.chat(
            follow_up_messages,
            tools=self._tool_registry.schemas(),
        )
        self._session_memory.add_assistant_message(final_result.content)
        return final_result
