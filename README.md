# Agent-Demo

`Agent-Demo` is a memory-first CLI assistant prototype. The host process owns
prompt assembly, memory retrieval, and turn orchestration, while the model can
persist durable facts through a single `memory_write` tool.

## Project Layout

```text
Agent-Demo/
|-- agent/          # turn loop and prompt assembly
|-- docs/           # design notes and implementation plan
|-- llm/            # DeepSeek client wrapper
|-- memory/         # session, project, and long-term memory stores
|-- storage/        # shared SQLite access layer
|-- tests/          # test suite
|-- tools/          # model-callable tool registry and builtins
|-- config.py       # runtime settings loader
|-- main.py         # CLI entry point
|-- pyproject.toml  # project metadata and test configuration
`-- uv.lock         # locked dependency graph for uv
```

## Development Workflow

This repository uses `uv` for dependency management and command execution.

### Setup

```powershell
uv sync --extra dev
```

This creates or updates the local `.venv` from `pyproject.toml` and `uv.lock`.

### Run Tests

```powershell
uv run pytest -q
```

### Run One Test File

```powershell
uv run pytest tests/storage/test_db.py -q
```

## Run The CLI

```powershell
uv run python main.py --user-id demo-user --project-id demo-project
```

Type `exit` or `quit` to end the session.

## Environment Variables

Copy `.env.example` to `.env` and fill in the values you need:

```env
DEEPSEEK_API_KEY=your-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-v4-flash
DB_PATH=agent_memory.db
```

`DEEPSEEK_API_KEY` is required by `load_settings()`. The rest have defaults, but
keeping them explicit in `.env` makes local runs easier to reason about.

## Python Version

The repository is pinned through [`.python-version`](./.python-version). `uv`
will use a compatible local Python when available, or manage one for the
project when needed.
