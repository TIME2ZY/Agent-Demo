import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


_DOTENV_LOADED = False


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str
    deepseek_base_url: str
    model_name: str
    db_path: Path
    max_tool_steps_per_turn: int = 1
    session_history_limit: int = 8
    max_longterm_items_in_prompt: int = 5
    max_session_messages_in_prompt: int = 8


def load_settings() -> Settings:
    global _DOTENV_LOADED

    load_dotenv() if not _DOTENV_LOADED else None
    _DOTENV_LOADED = True

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    model_name = os.getenv("MODEL_NAME", "deepseek-v4-flash").strip()
    db_path_value = os.getenv("DB_PATH", "agent_memory.db").strip()

    if not deepseek_api_key:
        raise ValueError("DEEPSEEK_API_KEY is required")

    return Settings(
        deepseek_api_key=deepseek_api_key,
        deepseek_base_url=deepseek_base_url,
        model_name=model_name,
        db_path=Path(db_path_value),
    )
