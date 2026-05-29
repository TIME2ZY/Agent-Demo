import argparse
import asyncio
import logging
import sys
from typing import Any

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


def write_output_line(text: str, stream=None) -> None:
    target = stream or sys.stdout

    try:
        print(text, file=target)
    except UnicodeEncodeError:
        encoding = target.encoding or "utf-8"
        target.buffer.write((text + "\n").encode(encoding, errors="replace"))
        target.flush()


def format_turn_error_message(debug: bool) -> str:
    if debug:
        return (
            "Assistant: 抱歉，这一轮处理失败了。"
            "详细错误已输出，请查看调试日志。"
        )

    return (
        "Assistant: 抱歉，这一轮处理失败了。"
        "请重试；如果需要排查，请使用 --debug 重新启动。"
    )


def _format_scalar(value: Any) -> str:
    if value in (None, "", [], {}):
        return "None"
    return str(value)


def format_memory_snapshot(
    longterm_memories: list[dict[str, Any]],
    project_context: dict[str, Any] | None,
) -> str:
    lines = ["Assistant: Current memory snapshot", "", "Long-term memory:"]
    if longterm_memories:
        for item in longterm_memories:
            lines.append(f"- {item['key']}: {item['value']}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("Project memory:")
    if not project_context:
        lines.append("- None")
        return "\n".join(lines)

    ordered_keys = [
        "project_goal",
        "tech_stack",
        "constraints",
        "decisions",
        "current_status",
        "open_questions",
        "major_milestones",
    ]
    for key in ordered_keys:
        value = project_context.get(key)
        if isinstance(value, list):
            if value:
                lines.append(f"- {key}: {', '.join(str(item) for item in value)}")
            else:
                lines.append(f"- {key}: None")
        else:
            lines.append(f"- {key}: {_format_scalar(value)}")

    return "\n".join(lines)


def format_events_snapshot(event_rows: list[dict[str, Any]]) -> str:
    lines = ["Assistant: Recent memory events"]
    if not event_rows:
        lines.append("- None")
        return "\n".join(lines)

    for row in event_rows:
        lines.append(
            f"- {row['memory_level']} | {row['key']} | {row['reason']} | {row['created_at']}"
        )
    return "\n".join(lines)


async def handle_local_command(
    command: str,
    *,
    user_id: str,
    project_id: str,
    database: Database,
    project_store: ProjectMemoryStore,
    longterm_store: LongTermMemoryStore,
) -> str | None:
    normalized = command.strip().lower()
    if normalized == "/memory":
        longterm_memories = await longterm_store.list_memories(user_id)
        project_context = await project_store.get_project_context(project_id)
        return format_memory_snapshot(longterm_memories, project_context)

    if normalized == "/events":
        event_rows = await database.fetchall(
            """
            SELECT memory_level, key, reason, created_at
            FROM memory_events
            WHERE user_id = ? AND project_id = ?
            ORDER BY id DESC
            LIMIT 20
            """,
            (user_id, project_id),
        )
        return format_events_snapshot(event_rows)

    return None


def print_turn_output(result: LLMResult) -> None:
    if result.reasoning_content:
        write_output_line(f"Thinking: {result.reasoning_content}")
    write_output_line(f"Assistant: {result.content}")


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

            command_output = await handle_local_command(
                user_message,
                user_id=user_id,
                project_id=project_id,
                database=database,
                project_store=project_store,
                longterm_store=longterm_store,
            )
            if command_output is not None:
                write_output_line(command_output)
                continue

            try:
                result = await agent_loop.run_turn(user_id, project_id, user_message)
            except Exception:
                logging.getLogger(__name__).exception("turn_processing_failed")
                write_output_line(format_turn_error_message(debug))
                continue

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
