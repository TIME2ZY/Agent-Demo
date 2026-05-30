import json
import logging
from types import SimpleNamespace

import pytest

from llm.client import DeepSeekClient, LLMResult


class MockChunk:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


async def async_iter(items):
    for item in items:
        yield item


def build_test_client(stream_items=None):
    async def fake_create(*, model, messages, tools=None, stream=False):
        if stream:
            return async_iter(stream_items or [])
        raise AssertionError("unexpected non-stream create() call in this test")

    client = DeepSeekClient.__new__(DeepSeekClient)
    client._settings = SimpleNamespace(model_name="deepseek-v4-flash")
    client._logger = logging.getLogger("tests.llm.client")
    client._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=fake_create),
        )
    )
    return client


def test_normalize_response_returns_message_result():
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello from DeepSeek",
                    "reasoning_conttent": "Simple greeting response",
                },
                "finish_reason": "stop",
            }
        ]
    }

    result = DeepSeekClient.normalize_response(payload)

    assert isinstance(result, LLMResult)
    assert result.type == "message"
    assert result.content == "Hello from DeepSeek"
    assert result.reasoning_content == "Simple greeting response"
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
    assert result.tool_call_id == "call-1"
    assert result.tool_name == "memory_write"
    assert result.tool_args["key"] == "response_style"
    assert result.reasoning_content is None


def test_normalize_response_preserves_multiple_tool_calls():
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
                                "name": "read_file",
                                "arguments": json.dumps({"path": "notes.txt"}),
                            },
                        },
                        {
                            "id": "call-2",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": json.dumps(
                                    {"path": "out.txt", "content": "done"}
                                ),
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }

    result = DeepSeekClient.normalize_response(payload)

    assert result.type == "tool_call"
    assert result.tool_call_id == "call-1"
    assert result.tool_name == "read_file"
    assert result.tool_args == {"path": "notes.txt"}
    assert result.tool_calls == [
        {
            "id": "call-1",
            "name": "read_file",
            "arguments": {"path": "notes.txt"},
        },
        {
            "id": "call-2",
            "name": "write_file",
            "arguments": {"path": "out.txt", "content": "done"},
        },
    ]


@pytest.mark.asyncio
async def test_chat_stream_accumulates_message_chunks_and_reasoning():
    chunks = [
        MockChunk(
            {
                "choices": [
                    {
                        "delta": {
                            "content": "Hel",
                            "reasoning_content": "step-1 ",
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        MockChunk(
            {
                "choices": [
                    {
                        "delta": {
                            "content": "lo",
                            "reasoning_conttent": "step-2",
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        ),
    ]
    client = build_test_client(stream_items=chunks)
    seen_chunks = []

    result = await client.chat_stream(
        messages=[{"role": "user", "content": "Say hello"}],
        on_content_chunk=seen_chunks.append,
    )

    assert result.type == "message"
    assert result.content == "Hello"
    assert result.reasoning_content == "step-1 step-2"
    assert seen_chunks == ["Hel", "lo"]


@pytest.mark.asyncio
async def test_chat_stream_suppresses_content_callback_for_tool_call_chunks():
    chunks = [
        MockChunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "memory_write",
                                        "arguments": '{"memory_level":"long',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        MockChunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": 'term","key":"response_style","value":"x","reason":"y"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        ),
    ]
    client = build_test_client(stream_items=chunks)
    seen_chunks = []

    result = await client.chat_stream(
        messages=[{"role": "user", "content": "Save preference"}],
        on_content_chunk=seen_chunks.append,
    )

    assert result.type == "tool_call"
    assert result.tool_call_id == "call-1"
    assert result.tool_name == "memory_write"
    assert result.tool_args["key"] == "response_style"
    assert seen_chunks == []


@pytest.mark.asyncio
async def test_chat_stream_matches_normalize_response_for_equivalent_payload():
    chunks = [
        MockChunk(
            {
                "choices": [
                    {
                        "delta": {
                            "content": "Hello from ",
                            "reasoning_content": "Simple ",
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        MockChunk(
            {
                "choices": [
                    {
                        "delta": {
                            "content": "DeepSeek",
                            "reasoning_content": "greeting response",
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        ),
    ]
    expected_payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello from DeepSeek",
                    "reasoning_content": "Simple greeting response",
                },
                "finish_reason": "stop",
            }
        ]
    }
    client = build_test_client(stream_items=chunks)

    streamed = await client.chat_stream(messages=[{"role": "user", "content": "hi"}])
    normalized = DeepSeekClient.normalize_response(expected_payload)

    assert streamed.type == normalized.type
    assert streamed.content == normalized.content
    assert streamed.reasoning_content == normalized.reasoning_content
    assert streamed.tool_name == normalized.tool_name
    assert streamed.tool_args == normalized.tool_args


@pytest.mark.asyncio
async def test_chat_stream_handles_empty_content_stream():
    chunks = [
        MockChunk(
            {
                "choices": [
                    {
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ]
            }
        ),
    ]
    client = build_test_client(stream_items=chunks)
    seen_chunks = []

    result = await client.chat_stream(
        messages=[{"role": "user", "content": "Say nothing"}],
        on_content_chunk=seen_chunks.append,
    )

    assert result.type == "message"
    assert result.content == ""
    assert result.reasoning_content is None
    assert seen_chunks == []
