# Memory-First Personal Agent Design

## Summary

This document defines the first runnable version of `my_agent`: a CLI-based personal assistant focused on conversation and memory. The agent is not a coding agent and does not execute shell commands. Its core job is to:

- hold a useful conversation,
- remember stable user preferences,
- remember project-specific context,
- and decide when important information should be persisted.

The design favors a memory-first architecture where the host program controls memory retrieval and prompt assembly, while the model only decides whether to write new memory through a single tool: `memory_write`.

## Goals

- Build a single-user CLI assistant with multi-project memory isolation.
- Preserve long-term user preferences across sessions.
- Preserve project context per `project_id`.
- Keep the first version small, observable, and easy to debug.
- Use DeepSeek through the OpenAI-compatible SDK.
- Use SQLite as the only persistence layer for long-term and project memory.

## Non-Goals

- No shell execution.
- No coding-agent behavior.
- No file-writing tool in the first version.
- No multi-user product features in the first version.
- No vector database, embeddings, or retrieval pipelines in the first version.
- No complex plugin or tool ecosystem in the first version.

## Product Scope

The first version is a personal assistant with two kinds of durable memory:

- `L2 project memory`: facts and decisions tied to one project.
- `L3 long-term memory`: stable user preferences shared across projects.

It also has one short-lived memory layer:

- `L1 session memory`: recent conversation history stored only in memory for the current run.

The scope model is:

- one `user_id`,
- many `project_id`s,
- one active session at a time.

This means the same user can work across multiple projects while keeping shared long-term preferences and isolated project context.

## Architecture Overview

The architecture is intentionally host-driven:

- the host program retrieves memory,
- the host program decides what context to inject,
- the model responds to the injected context,
- the model may call `memory_write` when new durable information should be stored.

Memory retrieval is not exposed as a tool. This is a deliberate design decision. The host is better positioned than the model to load the right context every turn, which reduces missed memory lookups and keeps the tool surface small.

## Directory Structure

```text
my_agent/
├── .env
├── config.py
├── main.py
│
├── agent/
│   ├── loop.py
│   └── prompt.py
│
├── llm/
│   └── client.py
│
├── memory/
│   ├── session.py
│   ├── project.py
│   └── longterm.py
│
├── tools/
│   ├── registry.py
│   └── builtin/
│       └── memory_write.py
│
├── storage/
│   └── db.py
│
└── tests/
```

`read.py`, `write.py`, and `bash.py` are intentionally omitted from the MVP. They can be added later if the product evolves beyond a memory-first assistant.

## Module Responsibilities

### `config.py`

Defines global constants and runtime settings, including:

- DeepSeek model name,
- database file path,
- max tool steps per turn,
- session history limit,
- prompt memory injection limits,
- default `user_id` and `project_id` behavior if desired.

### `main.py`

The CLI entry point. Responsibilities:

- parse arguments such as `--user-id` and `--project-id`,
- initialize storage, memory services, tool registry, and LLM client,
- run the agent with `asyncio.run()`,
- keep the command-line loop alive until exit.

This should be the only top-level async entrypoint.

### `agent/loop.py`

Owns the turn-by-turn execution flow. Responsibilities:

- receive user input,
- update L1 session state,
- retrieve L2 and L3 memory,
- assemble prompt input,
- call the model,
- detect and execute tool calls,
- handle tool results,
- produce the final assistant reply for the turn.

This module is the orchestrator. It owns the control flow but not the business logic of memory storage or raw model response parsing.

### `agent/prompt.py`

Builds the final prompt payload for the model. Responsibilities:

- render system rules,
- render long-term memory in a compact format,
- render project memory as a concise summary,
- append recent session history,
- expose tool instructions, especially `memory_write` guidance.

This module controls what the model sees, but not what gets persisted.

### `llm/client.py`

Wraps the DeepSeek API and normalizes responses. Responsibilities:

- call the OpenAI-compatible SDK against DeepSeek `base_url`,
- preserve the raw response for debugging,
- detect whether a response is a normal assistant message or a tool call,
- normalize the result into a stable internal shape used by `agent/loop.py`.

This boundary is critical because DeepSeek tool-calling payloads may differ from OpenAI expectations. The rest of the codebase should never rely on raw provider response structure.

### `memory/session.py`

In-memory short-term session history. Responsibilities:

- append user and assistant messages,
- append tool call and tool result messages if needed,
- return the recent history for prompt injection.

It does not talk to SQLite.

### `memory/project.py`

Project-scoped memory service. Responsibilities:

- load project memory by `project_id`,
- save the full project memory blob,
- merge new project facts or decisions into the stored JSON.

### `memory/longterm.py`

User-scoped memory service. Responsibilities:

- load long-term preference entries by `user_id`,
- retrieve one preference by key,
- write or replace preference entries.

### `tools/registry.py`

Thin tool registry. Responsibilities:

- define the tool schema exposed to the model,
- validate tool arguments,
- dispatch tool execution,
- return structured results or structured errors.

The first version only needs to support one tool cleanly.

### `tools/builtin/memory_write.py`

The only model-callable tool in the MVP. Responsibilities:

- accept validated arguments,
- route writes to project memory or long-term memory,
- return a structured success or error payload.

### `storage/db.py`

SQLite access layer and single connection owner. Responsibilities:

- create and manage one shared async SQLite connection,
- create tables if needed,
- expose base query helpers,
- keep the rest of the system from opening ad hoc connections.

This module should not know anything about prompts or agent flow.

## Memory Model

### L1: Session Memory

L1 is the current in-memory conversation history. It is:

- transient,
- local to one process run,
- used to preserve immediate conversational continuity.

It should store recent messages only. The host controls how many are injected into each model call.

### L2: Project Memory

L2 captures facts specific to one project. Typical examples:

- project goal,
- confirmed constraints,
- selected technical decisions,
- current status,
- open questions,
- major milestones.

This memory is isolated by `project_id`.

The first version stores L2 as one JSON blob per project because project context is usually best retrieved as a whole. A project memory document may look like:

```json
{
  "project_goal": "Build a conversation and memory-focused personal assistant",
  "constraints": [
    "No bash tool",
    "CLI only in v1"
  ],
  "decisions": [
    "Use DeepSeek",
    "Use SQLite",
    "Use project_id to isolate project memory"
  ],
  "current_status": [
    "Three memory layers confirmed"
  ],
  "open_questions": [],
  "updated_at": "2026-05-28T20:30:00+08:00"
}
```

### L3: Long-Term Memory

L3 stores durable user preferences shared across projects. Typical examples:

- preferred language,
- preferred response style,
- stable working style,
- recurring formatting preferences.

This memory is keyed by `user_id` and stored as small independent entries rather than one large blob. Each item should be easy to overwrite without rewriting the full user profile.

Example logical shape:

```json
[
  {
    "key": "response_style",
    "value": "Prefer concise, direct Chinese answers",
    "confidence": "high"
  },
  {
    "key": "working_style",
    "value": "Prefers architecture first, implementation second",
    "confidence": "high"
  }
]
```

## Prompt Composition

The model input should be assembled in this order:

1. system rules,
2. L3 long-term memory,
3. L2 project memory,
4. recent L1 session history,
5. current user message.

This order is intentional:

- system rules define the agent behavior,
- long-term memory describes the person,
- project memory describes the current work context,
- session history provides immediate conversational continuity.

### Prompt Injection Rules

For the MVP, prompt injection should stay conservative:

- inject at most 5 long-term memory items,
- inject project memory as a summarized text block, not raw JSON,
- inject only the most recent 6 to 10 session messages,
- trim L1 before trimming L2 or L3 when the prompt gets large.

Project memory should be rendered into readable text, for example:

```text
Project goal:
- Build a memory-first personal assistant

Constraints:
- No bash tool
- CLI only in v1

Confirmed decisions:
- Use DeepSeek
- Use SQLite
- Isolate project memory by project_id
```

The stored representation and the prompt representation should be different if that improves clarity.

## `memory_write` Tool Design

The first version exposes exactly one tool:

`memory_write`

### Purpose

Allows the model to request durable memory writes when the conversation reveals information that should persist beyond the current turn or session.

### Tool Parameters

Recommended argument shape:

```json
{
  "memory_level": "project | longterm",
  "key": "string",
  "value": "string or json",
  "reason": "why this should be saved",
  "replace": true
}
```

`replace` is optional but recommended. It is especially useful for long-term preference updates where newer explicit preferences should often replace older values under the same key.

### Write Rules

The system prompt should instruct the model to call `memory_write` when:

- the user states a stable preference,
- the user confirms a project goal,
- the user confirms a project constraint,
- the user confirms a project decision,
- the user describes important project status that should survive across sessions.

The model should not call `memory_write` for:

- casual small talk,
- temporary details,
- low-confidence guesses,
- one-off wording that is unlikely to matter later.

### Execution Semantics

When `memory_write` is executed:

- `project` writes are merged into the current project JSON structure,
- `longterm` writes replace or create one key for the current user,
- a structured result is returned to the model,
- the model can then produce a final natural-language reply.

Example success result:

```json
{
  "ok": true,
  "memory_level": "longterm",
  "key": "response_style",
  "action": "updated"
}
```

## Main Loop Behavior

The turn loop should follow a fixed state machine:

1. read user input,
2. append the user message to L1,
3. load L2 project memory,
4. load L3 long-term memory,
5. build prompt input,
6. call the model,
7. inspect whether the result is a normal message or tool call,
8. if it is a normal message, output it and store it in L1,
9. if it is `memory_write`, execute the tool, append the tool interaction to L1, then call the model once more for the final assistant reply,
10. end the turn.

### Tool-Step Limit

The MVP should enforce a strict cap:

- `MAX_TOOL_STEPS_PER_TURN = 1` is recommended,
- `2` is acceptable if needed during debugging.

This avoids loops where the model keeps trying to write memory fragments repeatedly.

## LLM Response Normalization

`llm/client.py` should normalize all responses into a stable internal format such as:

```python
{
    "type": "message" | "tool_call",
    "content": "...",
    "tool_name": "memory_write",
    "tool_args": {...},
    "raw_response": ...
}
```

The host should always log or preserve the raw provider response for debugging. Parsing logic must be based on observed DeepSeek payloads rather than assumed OpenAI parity.

## SQLite Schema

The MVP should use two required business tables and one optional event log table.

### `project_memory`

```sql
CREATE TABLE IF NOT EXISTS project_memory (
    project_id TEXT PRIMARY KEY,
    context_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

Purpose:

- one row per project,
- one JSON blob containing current project memory,
- updated as a whole after merges.

### `longterm_memory`

```sql
CREATE TABLE IF NOT EXISTS longterm_memory (
    user_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, key)
);
```

Purpose:

- one row per user preference key,
- precise replacement by key,
- shared across projects for the same user.

### Optional: `memory_events`

```sql
CREATE TABLE IF NOT EXISTS memory_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    project_id TEXT,
    memory_level TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL
);
```

Purpose:

- debug memory writes,
- inspect what the model decided to persist,
- trace why certain memory state exists.

This table is optional for the strict MVP, but recommended for observability.

## Async and Storage Rules

The project should be fully async from the start.

- `main.py` should use `asyncio.run()` once.
- SQLite access should use one shared `aiosqlite` connection.
- memory services should depend on the shared connection provider from `storage/db.py`.
- no module should create ad hoc database connections inside helper methods.

This avoids nested event loop issues and keeps persistence behavior predictable.

## Error Handling

Errors should be recoverable at the turn level whenever possible.

### LLM Errors

- surface a clean user-facing failure message,
- log raw request and response context where possible,
- do not corrupt memory state.

### Tool Argument Errors

- return a structured tool error,
- allow the model to recover with a corrected or final natural-language reply.

### SQLite Write Errors

- return a structured tool failure,
- do not crash the full session if one memory write fails,
- preserve the conversation loop if possible.

In all cases, a failed `memory_write` should be treated as a failed tool action inside the current turn, not as a fatal process error by default.

## Logging and Observability

The MVP should include lightweight logging from the beginning.

At minimum, log:

- normalized LLM result type,
- raw DeepSeek response payload,
- tool name and tool args,
- tool result,
- memory write destination and key,
- turn-level errors.

This is essential because provider-specific tool-calling formats are easy to misread.

## Testing Strategy

The first version does not need large test coverage, but it does need focused tests for the fragile edges.

Recommended minimum test set:

- `llm/client.py`: normalize a plain message response,
- `llm/client.py`: normalize a tool-call response,
- `tools/registry.py`: dispatch `memory_write` with valid args,
- `tools/registry.py`: reject invalid args,
- `memory/project.py`: merge and persist project memory,
- `memory/longterm.py`: replace one preference by key.

These tests matter more than broad coverage because the highest-risk failures are format drift, bad tool schemas, and incorrect persistence semantics.

## Configuration

`config.py` should define or derive at least:

- `DEEPSEEK_API_KEY`,
- `DEEPSEEK_BASE_URL`,
- `MODEL_NAME`,
- `DB_PATH`,
- `MAX_TOOL_STEPS_PER_TURN`,
- `SESSION_HISTORY_LIMIT`,
- `MAX_LONGTERM_ITEMS_IN_PROMPT`,
- `MAX_SESSION_MESSAGES_IN_PROMPT`.

Environment variables should remain the source of secrets, while non-secret runtime defaults can live in code.

## MVP Implementation Order

The first implementation should be built incrementally. Each step must be independently runnable before moving on.

1. `config.py` and `.env` loading.
2. `llm/client.py` with one successful plain text call to DeepSeek.
3. `main.py` and `agent/loop.py` for one-turn plain conversation without tools.
4. `memory/session.py` to preserve recent chat history.
5. `storage/db.py` with shared connection and table creation.
6. `memory/project.py` and `memory/longterm.py` basic load and save behavior.
7. `tools/registry.py` and `tools/builtin/memory_write.py`.
8. prompt instructions for memory writing.
9. one end-to-end turn where the model calls `memory_write` successfully.
10. focused tests for normalization, dispatch, and persistence.

Do not wait until everything is implemented before running the agent. The system should stay runnable at each stage.

## Deferred Work

The following ideas are explicitly deferred until after the MVP works reliably:

- file-reading tools,
- file-writing tools,
- retrieval or search systems,
- richer project memory schema normalization,
- multi-user support,
- GUI or web interface,
- more than one tool in the registry,
- automatic summarization or compaction pipelines.

## Key Design Decisions

- The agent is memory-first, not execution-first.
- Memory retrieval is host-driven, not tool-driven.
- Project memory is isolated by `project_id`.
- Long-term memory is shared by `user_id`.
- Project memory is stored as a whole JSON blob.
- Long-term memory is stored as key-based entries.
- The only model-callable tool in the MVP is `memory_write`.
- The project is async end-to-end.
- SQLite is the single persistence backend.
- LLM provider response parsing is isolated inside `llm/client.py`.

## Acceptance Criteria for the MVP

The MVP is successful when all of the following are true:

- the CLI can start a session for a given `user_id` and `project_id`,
- the model can answer normally without tools,
- the host injects relevant L2 and L3 memory each turn,
- the model can call `memory_write` successfully,
- project memory persists across restarts for the same `project_id`,
- long-term memory persists across restarts for the same `user_id`,
- different projects do not share project memory accidentally,
- one failed memory write does not crash the whole conversation loop.

## Conclusion

This design keeps the first version intentionally narrow: a reliable personal conversation assistant with durable project and preference memory. The narrow tool surface is a feature, not a limitation. It reduces ambiguity, keeps debugging manageable, and creates a stable foundation for future capabilities only after the memory-first core works end to end.
