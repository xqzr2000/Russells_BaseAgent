"""OpenAIClient streaming accumulation and unsupported-parameter fallback, mocked."""

from __future__ import annotations

import httpx
import openai
from openai.types.chat import ChatCompletionChunk

from baseagent.llm import ModelSettings, OpenAIClient


def chunk(delta: dict, usage: dict | None = None) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate({
        "id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
        "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": None}],
        "usage": usage,
    })


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self.chunks:
                yield c
        return gen()


def bad_request(message: str, param: str | None) -> openai.BadRequestError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    body = {"error": {"message": message, "param": param}}
    return openai.BadRequestError(message, response=httpx.Response(400, request=request, json=body), body=body["error"])


async def test_stream_accumulates_text_tool_calls_and_usage(monkeypatch):
    client = OpenAIClient(api_key="k")
    chunks = [
        chunk({"role": "assistant", "content": "Let me "}),
        chunk({"content": "check."}),
        chunk({"tool_calls": [{"index": 0, "id": "call_a", "type": "function",
                               "function": {"name": "calculator", "arguments": '{"expr'}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'ession": "1+1"}'}}]}),
        chunk({"tool_calls": [{"index": 1, "id": "call_b", "type": "function",
                               "function": {"name": "current_time", "arguments": "{}"}}]}),
        chunk(None, {"prompt_tokens": 50, "completion_tokens": 9, "total_tokens": 59,
                     "completion_tokens_details": {"reasoning_tokens": 4}}),
    ]
    seen = {}

    async def create(**params):
        seen.update(params)
        return FakeStream(chunks)

    monkeypatch.setattr(client._client.chat.completions, "create", create)
    deltas = []

    async def on_text(d):
        deltas.append(d)

    response = await client.chat([{"role": "user", "content": "hi"}],
                                 ModelSettings(model="gpt-5-mini", reasoning_effort="low"), tools=[{}], on_text=on_text)
    assert deltas == ["Let me ", "check."]
    message = response.message
    assert message["content"] == "Let me check."
    assert [c["function"]["name"] for c in message["tool_calls"]] == ["calculator", "current_time"]
    assert message["tool_calls"][0]["function"]["arguments"] == '{"expression": "1+1"}'
    assert response.usage.total_tokens == 59 and response.usage.reasoning_tokens == 4
    assert seen["reasoning_effort"] == "low" and seen["stream"] is True
    assert seen["max_completion_tokens"] == 4096 and "temperature" not in seen


async def test_unsupported_params_are_dropped_and_remembered(monkeypatch):
    client = OpenAIClient(api_key="k")
    calls = []

    async def create(**params):
        calls.append(dict(params))
        if "temperature" in params:
            raise bad_request("Unsupported value: 'temperature' does not support 0.2 with this model.", "temperature")
        if "reasoning_effort" in params:
            raise bad_request("Unrecognized request argument supplied: reasoning_effort", None)
        return FakeStream([chunk({"content": "ok"})])

    monkeypatch.setattr(client._client.chat.completions, "create", create)
    settings = ModelSettings(model="gpt-4.1", temperature=0.2, reasoning_effort="high")

    async def on_text(_):
        pass

    response = await client.chat([], settings, on_text=on_text)
    assert response.message["content"] == "ok"
    assert len(response.warnings) == 2 and len(calls) == 3
    calls.clear()
    second = await client.chat([], settings, on_text=on_text)
    assert len(calls) == 1 and not second.warnings  # remembered for this model


async def test_other_bad_requests_are_raised(monkeypatch):
    client = OpenAIClient(api_key="k")

    async def create(**params):
        raise bad_request("The model `nope` does not exist", "model")

    monkeypatch.setattr(client._client.chat.completions, "create", create)
    try:
        await client.chat([], ModelSettings(model="nope"))
    except openai.BadRequestError:
        pass
    else:
        raise AssertionError("expected BadRequestError")
