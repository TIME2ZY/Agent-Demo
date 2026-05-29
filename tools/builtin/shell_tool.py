import asyncio
from pathlib import Path
from typing import Any

from tools.registry import RegisteredTool


def validate_run_shell_args(arguments: dict[str, Any]) -> dict[str, Any]:
    command = str(arguments.get("command", "")).strip()
    if not command:
        raise ValueError("command must be a non-empty string")

    timeout_seconds = arguments.get("timeout_seconds", 30)
    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive number")

    cwd_value = arguments.get("cwd")
    cwd = None
    if cwd_value is not None:
        cwd = str(cwd_value).strip()
        if not cwd:
            raise ValueError("cwd must be a non-empty string when provided")

    return {
        "command": command,
        "timeout_seconds": float(timeout_seconds),
        "cwd": cwd,
    }


def create_run_shell_tool() -> RegisteredTool:
    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        cwd = None
        if arguments["cwd"] is not None:
            cwd_path = Path(arguments["cwd"]).expanduser().resolve()
            if not cwd_path.exists() or not cwd_path.is_dir():
                raise ValueError(f"cwd must point to an existing directory: {cwd_path}")
            cwd = str(cwd_path)

        process = await asyncio.create_subprocess_shell(
            arguments["command"],
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=arguments["timeout_seconds"],
            )
        except TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise TimeoutError(
                f"Command timed out after {arguments['timeout_seconds']} seconds"
            ) from exc

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        return {
            "ok": process.returncode == 0,
            "command": arguments["command"],
            "cwd": cwd,
            "returncode": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }

    return RegisteredTool(
        name="run_shell",
        description="Run a shell command and capture stdout, stderr, and return code.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "cwd": {"type": "string"},
                "timeout_seconds": {"type": "number"},
            },
            "required": ["command"],
        },
        validator=validate_run_shell_args,
        handler=handler,
    )
