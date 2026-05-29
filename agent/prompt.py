from typing import Any


def build_system_prompt() -> str:
    return """You are a personal assistant focused on conversation and memory.

Use memory_write when:
- the user states a stable long-term preference,
- the user confirms a project goal, project constraint, project decision, or project status worth preserving.

Do not use memory_write for temporary chatter, low-confidence guesses, or one-off details.
"""


def format_longterm_memories(memories: list[dict[str, Any]], max_items: int) -> str:
    if not memories:
        return "Long-term memory:\n- None recorded yet"

    lines = ["Long-term memory:"]
    for item in memories[:max_items]:
        lines.append(f"- {item['key']}: {item['value']}")
    return "\n".join(lines)


def format_project_memory(context: dict[str, Any] | None) -> str:
    if not context:
        return "Project memory:\n- None recorded yet"

    lines = ["Project goal:"]
    lines.append(f"- {context.get('project_goal') or 'Not recorded yet'}")

    sections = [
        ("Constraints", context.get("constraints", [])),
        ("Confirmed decisions", context.get("decisions", [])),
        ("Current status", context.get("current_status", [])),
        ("Open questions", context.get("open_questions", [])),
    ]
    for label, items in sections:
        lines.append("")
        lines.append(f"{label}:")
        if items:
            lines.extend(f"- {item}" for item in items)
        else:
            lines.append("- None")

    return "\n".join(lines)


def build_messages(
    settings: Any,
    longterm_memories: list[dict[str, Any]],
    project_context: dict[str, Any] | None,
    session_messages: list[dict[str, Any]],
    user_message: str,
) -> list[dict[str, Any]]:
    system_content = "\n\n".join(
        [
            build_system_prompt(),
            format_longterm_memories(
                longterm_memories,
                settings.max_longterm_items_in_prompt,
            ),
            format_project_memory(project_context),
        ]
    )
    trimmed_session = session_messages[-settings.max_session_messages_in_prompt :]

    return [
        {"role": "system", "content": system_content},
        *trimmed_session,
        {"role": "user", "content": user_message},
    ]
