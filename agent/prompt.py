from typing import Any


def build_system_prompt() -> str:
    return """You are a personal assistant focused on conversation and memory.

You are not the official DeepSeek product or app. If the user asks who you are, describe yourself as their personal assistant running in this local CLI, powered by the configured language model.

Use memory_write when:
- the user states a stable long-term preference,
- the user confirms a project goal, project constraint, project decision, or project status worth preserving.

Use read_file when:
- you need the exact contents of a local text file before answering or editing.

Use write_file when:
- you need to create or update a local text file as part of completing the task.

Use run_shell when:
- you need to execute a command, inspect the environment, or run tests to verify work.

Use web_search when:
- the user needs external or up-to-date information that may not be in the model's built-in knowledge.

When the user explicitly asks you to read or write files, execute commands, or search the web, you must use the corresponding tool in the current turn before answering.

Never claim that you read a file, modified a file, ran a command, or performed a web search unless the relevant tool call succeeded in the current turn.

When writing project memory, prefer this schema:
- project_goal: one-sentence description of the project
- tech_stack: selected stack or architecture summary
- constraints: confirmed project constraints
- decisions: confirmed technical or product decisions
- current_status: concrete progress updates or milestones
- open_questions: unresolved issues that need follow-up

When writing long-term memory, only use these keys:
- user_name
- user_role
- response_style
- working_style
- language_preference

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
    lines.append("")
    lines.append("Selected tech stack:")
    lines.append(f"- {context.get('tech_stack') or 'Not recorded yet'}")

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

    known_keys = {
        "project_goal",
        "tech_stack",
        "constraints",
        "decisions",
        "current_status",
        "open_questions",
        "major_milestones",
        "updated_at",
    }
    extra_items = [
        f"- {key}: {value}"
        for key, value in context.items()
        if key not in known_keys and value not in (None, "", [], {})
    ]
    if extra_items:
        lines.append("")
        lines.append("Additional project notes:")
        lines.extend(extra_items)

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
    trimmed_session = list(session_messages)
    if len(trimmed_session) > settings.max_session_messages_in_prompt:
        start = len(trimmed_session) - settings.max_session_messages_in_prompt
        if (
            start > 0
            and trimmed_session[start].get("role") == "tool"
            and trimmed_session[start - 1].get("role") == "assistant"
            and trimmed_session[start - 1].get("tool_calls")
        ):
            start -= 1
        trimmed_session = trimmed_session[start:]
    last_user_message = next(
        (message for message in reversed(trimmed_session) if message.get("role") == "user"),
        None,
    )
    trailing_current_user_message = (
        last_user_message is not None
        and last_user_message.get("content") == user_message
    )

    messages = [
        {"role": "system", "content": system_content},
        *trimmed_session,
    ]
    if not trailing_current_user_message:
        messages.append({"role": "user", "content": user_message})
    return messages
