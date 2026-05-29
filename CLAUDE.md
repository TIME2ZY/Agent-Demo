# CLAUDE.md

## Commands

```bash
uv sync --extra dev          # Install all deps (prod + dev)
uv run pytest -q             # Run full test suite
uv run pytest tests/<path>   # Run a single test file
uv run python main.py --user-id <user> --project-id <project> [--debug]
```

No linter/type-checker is configured yet. Tests live under `tests/` and mirror the source layout (`tests/storage/test_db.py`, `tests/agent/test_loop.py`).

## Architecture

```
main.py ──► AgentLoop ──► PromptBuilder ──► DeepSeekClient
                │                                 │
                ▼                                 ▼
         SessionMemory                    ToolRegistry
                │                                 │
                ▼                                 ▼
     ProjectMemoryStore ◄── memory_write ──► LongTermMemoryStore
                │                                 │
                └──────────── Database ───────────┘
```

**Data flow per turn:**
1. `main.py` reads user input, checks for local commands (`/memory`, `/events`), else hands off to `AgentLoop.run_turn()`
2. `AgentLoop` appends user message to `SessionMemory`, fetches long-term + project memories, calls `build_messages()` to assemble the full prompt
3. `DeepSeekClient.chat()` sends the request via OpenAI-compatible SDK
4. If the response is a tool call → dispatch to `ToolRegistry`, append tool + result to session, call LLM again with updated context
5. Return final `LLMResult` (content + optional reasoning_content)

## Modules

| Package | Purpose |
|---|---|
| `agent/` | `AgentLoop` (turn orchestration) and `prompt.py` (system prompt + message assembly) |
| `llm/` | `DeepSeekClient` — AsyncOpenAI wrapper, response normalization (`normalize_response`) handles both message and tool_call responses |
| `memory/` | Three stores: `SessionMemory` (in-memory conversation), `ProjectMemoryStore` (SQLite, per project_id), `LongTermMemoryStore` (SQLite, per user_id) |
| `storage/` | `Database` — thin async SQLite wrapper with `execute`, `fetchone`, `fetchall`. Uses `aiosqlite.Row` row factory. |
| `tools/` | `ToolRegistry` — register tools, generate OpenAI tool schemas, dispatch by name |
| `tools/builtin/` | `memory_write` — the only built-in tool; validates keys against allowlists, writes to project/longterm stores, logs to `memory_events` |
| `config.py` | `load_settings()` from `.env` via `python-dotenv`, returns frozen `Settings` dataclass |

## Key conventions

- **Python 3.12** (`.python-version`)
- **`uv`** for all dependency/venv management — never use bare `pip` or `python`
- **SQLite** via `aiosqlite` — all DB access is async. Call `database.connect()` + `database.init_schema()` before use, `database.close()` when done
- **In-memory session** — `SessionMemory` is not persisted; only project/longterm stores hit SQLite
- **Memory events audit log** — every `memory_write` is logged to `memory_events` table regardless of level
- **Tool validation** — `validate_memory_write_args` enforces allowlists for both `longterm` and `project` keys before the handler runs
- **Project memory schema** — list-typed fields (`constraints`, `decisions`, `current_status`, `open_questions`, `major_milestones`) are dedup-appended on merge; scalar fields (`project_goal`, `tech_stack`) are overwritten
- **`LLMResult.type`** is `"message"` or `"tool_call"` — `AgentLoop` branches on this to decide whether to call tools or return immediately
- **`max_tool_steps_per_turn`** defaults to 4 — `AgentLoop` can execute multiple tool calls within one user turn before requiring a final assistant message
- **Tests use `tmp_path` fixture** for isolated SQLite databases — see `tests/conftest.py`

## Environment

Copy `.env.example` to `.env` — only `DEEPSEEK_API_KEY` is required.

| Variable | Default |
|---|---|
| `DEEPSEEK_API_KEY` | (required) |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` |
| `MODEL_NAME` | `deepseek-v4-flash` |
| `DB_PATH` | `agent_memory.db` |
