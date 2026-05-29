import logging

from llm.client import LLMResult
from main import configure_logging, print_turn_output


def test_print_turn_output_shows_reasoning_and_reply_by_default(capsys):
    result = LLMResult(
        type="message",
        content="Final answer",
        reasoning_content="Internal reasoning",
        tool_name=None,
        tool_args=None,
        raw_response={},
    )

    print_turn_output(result)

    captured = capsys.readouterr()
    assert captured.out == "Thinking: Internal reasoning\nAssistant: Final answer\n"
    assert captured.err == ""


def test_print_turn_output_skips_empty_reasoning(capsys):
    result = LLMResult(
        type="message",
        content="Final answer",
        reasoning_content=None,
        tool_name=None,
        tool_args=None,
        raw_response={},
    )

    print_turn_output(result)

    captured = capsys.readouterr()
    assert captured.out == "Assistant: Final answer\n"
    assert captured.err == ""


def test_configure_logging_defaults_to_quiet_mode():
    configure_logging(debug=False)

    assert logging.getLogger().getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING
