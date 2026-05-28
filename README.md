# Agent-Demo

This is a demo repository for practicing Git and GitHub workflows.

## Development Environment

This project uses `uv` for dependency management and local execution.

### Quick Start

1. Install `uv`
2. Sync the project environment:
   `uv sync --extra dev`
3. Run tests:
   `uv run pytest -q`

### Python Version

The project is pinned via `.python-version`. `uv` will use a compatible local
Python if one is available, or you can let `uv` manage Python for you.
