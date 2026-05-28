import json

from llm.client import DeepSeekClient, LLMResult


def test_normalize_response_returns_message_result():
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello from DeepSeek",
                },
                "finish_reason": "stop",
            }
        ]
    }

    result = DeepSeekClient.normalize_response(payload)

    assert isinstance(result, LLMResult)
    assert result.type == "message"
    assert result.content == "Hello from DeepSeek"
    assert result.tool_name is None


def test_normalize_response_returns_tool_call_result():
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "memory_write",
                                "arguments": json.dumps(
                                    {
                                        "memory_level": "longterm",
                                        "key": "response_style",
                                        "value": "Prefer concise Chinese",
                                        "reason": "explicit user preference",
                                    }
                                ),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }

    result = DeepSeekClient.normalize_response(payload)

    assert result.type == "tool_call"
    assert result.tool_name == "memory_write"
    assert result.tool_args["key"] == "response_style"
