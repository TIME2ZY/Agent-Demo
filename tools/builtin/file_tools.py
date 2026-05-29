import locale
from pathlib import Path
from typing import Any

from tools.registry import RegisteredTool


def validate_read_file_args(arguments: dict[str, Any]) -> dict[str, Any]:
    path = str(arguments.get("path", "")).strip()
    if not path:
        raise ValueError("path must be a non-empty string")

    return {"path": path}


def validate_write_file_args(arguments: dict[str, Any]) -> dict[str, Any]:
    path = str(arguments.get("path", "")).strip()
    if not path:
        raise ValueError("path must be a non-empty string")
    if "content" not in arguments:
        raise ValueError("content is required")

    content = arguments["content"]
    if not isinstance(content, str):
        raise ValueError("content must be a string")

    return {
        "path": path,
        "content": content,
        "append": bool(arguments.get("append", False)),
    }


def _resolve_path(path_value: str) -> Path:
    return Path(path_value).expanduser().resolve()


def _read_text_with_fallbacks(path: Path) -> str:
    encodings = ["utf-8"]
    preferred_encoding = locale.getpreferredencoding(False)
    if preferred_encoding and preferred_encoding.lower() not in {"utf-8", "utf8"}:
        encodings.append(preferred_encoding)
    if "gb18030" not in {encoding.lower() for encoding in encodings}:
        encodings.append("gb18030")

    last_error: UnicodeDecodeError | None = None
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    assert last_error is not None
    raise last_error


def create_read_file_tool() -> RegisteredTool:
    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        path = _resolve_path(arguments["path"])
        if not path.exists():
            raise ValueError(f"File does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        content = _read_text_with_fallbacks(path)
        return {
            "ok": True,
            "path": str(path),
            "content": content,
        }

    return RegisteredTool(
        name="read_file",
        description="Read UTF-8 text content from a local file path.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
        validator=validate_read_file_args,
        handler=handler,
    )


def create_write_file_tool() -> RegisteredTool:
    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        path = _resolve_path(arguments["path"])
        path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if arguments["append"] else "w"
        with path.open(mode, encoding="utf-8") as handle:
            handle.write(arguments["content"])

        return {
            "ok": True,
            "path": str(path),
            "bytes_written": len(arguments["content"].encode("utf-8")),
            "append": arguments["append"],
        }

    return RegisteredTool(
        name="write_file",
        description="Write UTF-8 text to a local file path, creating parent folders if needed.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "append": {"type": "boolean"},
            },
            "required": ["path", "content"],
        },
        validator=validate_write_file_args,
        handler=handler,
    )
