import argparse
import asyncio
import logging

from agent.loop import AgentLoop
from config import load_settings
from llm.client import DeepSeekClient
from memory.longterm import LongTermMemoryStore
from memory.project import ProjectMemoryStore
from memory.session import SessionMemory
from storage.db import Database
from tools.builtin.memory_write import create_memory_write_tool
from tools.registry import ToolRegistry


async def run_cli(user_id: str, project_id: str) -> None:
    logging.basicConfig(level=logging.INFO)

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

            reply = await agent_loop.run_turn(user_id, project_id, user_message)
            print(f"Assistant: {reply}")
    finally:
        await database.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory-first personal agent")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--project-id", required=True)
    args = parser.parse_args()

    asyncio.run(run_cli(args.user_id, args.project_id))


if __name__ == "__main__":
    main()
