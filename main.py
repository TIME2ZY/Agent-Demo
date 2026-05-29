import argparse
import asyncio
import logging

from agent.loop import AgentLoop
from config import load_settings
from llm.client import DeepSeekClient, LLMResult
from memory.longterm import LongTermMemoryStore
from memory.project import ProjectMemoryStore
from memory.session import SessionMemory
from storage.db import Database
from tools.builtin.memory_write import create_memory_write_tool
from tools.registry import ToolRegistry


def configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(level=level, force=True)
    logging.getLogger("httpx").setLevel(logging.INFO if debug else logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.INFO if debug else logging.WARNING)


def print_turn_output(result: LLMResult) -> None:
    if result.reasoning_content:
        print(f"Thinking: {result.reasoning_content}")
    print(f"Assistant: {result.content}")


async def run_cli(user_id: str, project_id: str, debug: bool = False) -> None:
    configure_logging(debug)

    settings = load_settings()
    database = Database(settings.db_path)
    await database.connect()
    await database.init_schema()

    project_store = ProjectMemoryStore(database)
    longterm_store = LongTermMemoryStore(database)
    session_memory = SessionMemory()
    client = DeepSeekClient(settings)
    registry = ToolRegistry()
    registry.register(
        create_memory_write_tool(project_store, longterm_store, user_id, project_id)
    )

    agent_loop = AgentLoop(
        settings=settings,
        client=client,
        session_memory=session_memory,
        project_store=project_store,
        longterm_store=longterm_store,
        tool_registry=registry,
    )

    try:
        while True:
            try:
                user_message = input("You: ").strip()
            except EOFError:
                break

            if user_message.lower() in {"exit", "quit"}:
                break
            if not user_message:
                continue

            result = await agent_loop.run_turn(user_id, project_id, user_message)
            print_turn_output(result)
    finally:
        await database.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory-first personal agent")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    asyncio.run(run_cli(args.user_id, args.project_id, debug=args.debug))


if __name__ == "__main__":
    main()
