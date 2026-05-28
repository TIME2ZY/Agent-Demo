from pathlib import Path

import pytest

from config import Settings, load_settings


def test_load_settings_reads_expected_values(monkeypatch, tmp_path):
    db_path = tmp_path / "memory.db"

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("MODEL_NAME", "deepseek-chat")
    monkeypatch.setenv("DB_PATH", str(db_path))

    settings = load_settings()

    assert isinstance(settings, Settings)
    assert settings.deepseek_api_key == "test-key"
    assert settings.deepseek_base_url == "https://example.invalid"
    assert settings.model_name == "deepseek-chat"
    assert settings.db_path == db_path
    assert settings.max_tool_steps_per_turn == 1
    assert settings.session_history_limit == 8


def test_load_settings_raises_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("MODEL_NAME", "deepseek-chat")
    monkeypatch.setenv("DB_PATH", "agent_memory.db")

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        load_settings()
