# Agent-Demo

`Agent-Demo` is an early-stage Python project for a memory-first CLI assistant.
The repository currently contains the configuration layer, a shared SQLite
storage layer, and the first round of tests that define the baseline behavior.

## Current Status

The project is still in the scaffold phase.

- Implemented:
  - environment and runtime settings loading in [`config.py`](./config.py)
  - shared SQLite connection and schema bootstrap in [`storage/db.py`](./storage/db.py)
  - smoke, config, and storage tests under [`tests/`](./tests)
- Reserved for later tasks:
  - `agent/`
  - `llm/`
  - `memory/`
  - `tools/`

There is no runnable `main.py` entry point yet. For now, the repository is best
treated as a tested foundation rather than a complete assistant.

## Project Layout

```text
Agent-Demo/
|-- agent/          # reserved package for the turn loop and prompt assembly
|-- docs/           # planning and design notes
|-- llm/            # reserved package for provider clients
|-- memory/         # reserved package for memory services
|-- storage/        # implemented SQLite access layer
|-- tests/          # active test suite
|-- tools/          # reserved package for model-callable tools
|-- config.py       # implemented runtime settings loader
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
