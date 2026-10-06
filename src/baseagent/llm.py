"""Model clients: one interface, a real OpenAI implementation, and fakes.

The agent only depends on ``LLMClient``. That keeps the loop testable offline
(``ScriptedLLM``) and gives the UI a no-key demo model (``FakeLLM``).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from pydantic import BaseModel

from baseagent.events import Usage

TextCallback = Callable[[str], Awaitable[None]]

FAKE_MODEL = "fake-echo"

# Parameters some models reject. When the API refuses one, the client drops it,
# remembers that for this model, and retries, so switching between e.g.
# gpt-4.1 (temperature, no reasoning_effort) and gpt-5 (the reverse) just works.
DROPPABLE_PARAMS = ("temperature", "reasoning_effort", "stream_options", "top_p")


class ModelSettings(BaseModel):
    model: str = "gpt-5-mini"
    reasoning_effort: str | None = None  # "minimal" | "low" | "medium" | "high"
    temperature: float | None = None
    max_output_tokens: int = 4096


@dataclass
class LLMResponse:
    message: dict[str, Any]  # {"role": "assistant", "content": str, "tool_calls"?: [...]}
    usage: Usage = field(default_factory=Usage)
    warnings: list[str] = field(default_factory=list)


class LLMClient(Protocol):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        settings: ModelSettings,
        tools: list[dict[str, Any]] | None = None,
        on_text: TextCallback | None = None,
    ) -> LLMResponse: ...

    async def list_models(self) -> list[str]: ...


def _usage_from(raw: Any) -> Usage:
    if raw is None:
        return Usage()
    data = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)
    prompt_details = data.get("prompt_tokens_details") or {}
    completion_details = data.get("completion_tokens_details") or {}
    return Usage(
        prompt_tokens=data.get("prompt_tokens") or 0,
        completion_tokens=data.get("completion_tokens") or 0,
        total_tokens=data.get("total_tokens") or 0,
        cached_tokens=prompt_details.get("cached_tokens") or 0,
        reasoning_tokens=completion_details.get("reasoning_tokens") or 0,
    )


class OpenAIClient:
    """Chat Completions over any OpenAI-compatible endpoint, with streaming."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
            max_retries=int(os.environ.get("OPENAI_MAX_RETRIES", "3")),
        )
        self._unsupported: dict[str, set[str]] = {}

    def _params(self, messages, settings: ModelSettings, tools, stream: bool) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": settings.model,
            "messages": messages,
            "max_completion_tokens": settings.max_output_tokens,
        }
        if tools:
            params["tools"] = tools
        if settings.reasoning_effort:
            params["reasoning_effort"] = settings.reasoning_effort
        if settings.temperature is not None:
            params["temperature"] = settings.temperature
        if stream:
            params["stream"] = True
            params["stream_options"] = {"include_usage": True}
        for name in self._unsupported.get(settings.model, ()):
            params.pop(name, None)
        return params

    def _rejected_param(self, exc: Exception, params: dict[str, Any]) -> str | None:
        named = getattr(exc, "param", None)
        if named in DROPPABLE_PARAMS and named in params:
            return named
        text = str(exc)
        for name in DROPPABLE_PARAMS:
            if name in params and re.search(rf"\b{name}\b", text):
                return name
        return None

    async def chat(self, messages, settings, tools=None, on_text=None) -> LLMResponse:
        from openai import BadRequestError

        stream = on_text is not None
        warnings: list[str] = []
        for _ in range(len(DROPPABLE_PARAMS) + 1):
            params = self._params(messages, settings, tools, stream)
            try:
                if stream:
                    return await self._stream(params, on_text, warnings)
                response = await self._client.chat.completions.create(**params)
                message = response.choices[0].message.model_dump(exclude_none=True)
                return LLMResponse(_clean(message), _usage_from(response.usage), warnings)
            except BadRequestError as exc:
                rejected = self._rejected_param(exc, params)
                if rejected is None:
                    raise
                self._unsupported.setdefault(settings.model, set()).add(rejected)
                warnings.append(f"{settings.model} does not support `{rejected}`; sent without it.")
        raise RuntimeError("Model request failed after dropping unsupported parameters.")

    async def _stream(self, params, on_text: TextCallback, warnings) -> LLMResponse:
        content: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        usage = Usage()
        stream = await self._client.chat.completions.create(**params)
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = _usage_from(chunk.usage)
            for choice in chunk.choices or []:
                delta = choice.delta
                if delta is None:
                    continue
                if delta.content:
                    content.append(delta.content)
                    await on_text(delta.content)
                for tc in delta.tool_calls or []:
                    slot = calls.setdefault(
                        tc.index,
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function is not None:
                        if tc.function.name:
                            slot["function"]["name"] += tc.function.name
                        if tc.function.arguments:
                            slot["function"]["arguments"] += tc.function.arguments
        message: dict[str, Any] = {"role": "assistant", "content": "".join(content)}
        if calls:
            message["tool_calls"] = [calls[i] for i in sorted(calls)]
        return LLMResponse(message, usage, warnings)

    async def list_models(self) -> list[str]:
        ids = [model.id async for model in self._client.models.list()]
        return sorted(ids)


def _clean(message: dict[str, Any]) -> dict[str, Any]:
    cleaned = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("tool_calls"):
        cleaned["tool_calls"] = [
            {"id": c["id"], "type": "function", "function": {
                "name": c["function"]["name"], "arguments": c["function"].get("arguments") or ""}}
            for c in message["tool_calls"]
        ]
    return cleaned


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class ScriptedLLM:
    """Replays a fixed list of assistant messages; records every request.

    For tests. A request with no ``tools`` (a compaction summary) consumes from
    ``summaries`` instead of the main script.
    """

    def __init__(self, script: list[dict[str, Any]], summaries: list[str] | None = None):
        self.script = list(script)
        self.summaries = list(summaries or [])
        self.requests: list[dict[str, Any]] = []

    async def chat(self, messages, settings, tools=None, on_text=None) -> LLMResponse:
        self.requests.append({"messages": json.loads(json.dumps(messages)), "tools": tools})
        if not tools and self.summaries:
            return LLMResponse({"role": "assistant", "content": self.summaries.pop(0)})
        if not self.script:
            raise RuntimeError("ScriptedLLM ran out of responses")
        message = dict(self.script.pop(0))
        message.setdefault("role", "assistant")
        message.setdefault("content", "")
        if on_text and message["content"]:
            await on_text(message["content"])
        return LLMResponse(message, Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    async def list_models(self) -> list[str]:
        return [FAKE_MODEL]


class FakeLLM:
    """A tiny rule-based "model" so the chat room works without an API key.

    It calls ``calculator`` for arithmetic, ``current_time`` when asked the
    time, and otherwise echoes. Good for checking the UI and the loop.
    """

    _ARITH = re.compile(r"[-+*/().\d\s^%]{3,}")

    async def chat(self, messages, settings, tools=None, on_text=None) -> LLMResponse:
        tool_names = {t["function"]["name"] for t in tools or []}
        last_user = next((i for i in range(len(messages) - 1, -1, -1)
                          if messages[i]["role"] == "user"), None)
        user_text = messages[last_user]["content"] if last_user is not None else ""
        if not isinstance(user_text, str):
            user_text = json.dumps(user_text)
        results = [m for m in messages[(last_user or 0) + 1:] if m["role"] == "tool"]

        if not tools:  # compaction request
            reply = "Summary: conversation so far handled by the fake model."
        elif results:
            reply = f"The tool returned: **{results[-1]['content']}**"
        else:
            expression = self._ARITH.search(user_text)
            if "calculator" in tool_names and expression and re.search(r"\d", expression.group()):
                return self._call("calculator", {"expression": expression.group().strip()})
            if "current_time" in tool_names and "time" in user_text.lower():
                return self._call("current_time", {})
            reply = (
                f"(fake model) You said: “{user_text.strip()}”. Try asking me to compute "
                "`12*(3+4)` or what time it is, or pick a real model in Settings."
            )
        if on_text:
            for word in re.findall(r"\S+\s*", reply):
                await on_text(word)
                await asyncio.sleep(0.02)
        return LLMResponse({"role": "assistant", "content": reply}, Usage(total_tokens=0))

    @staticmethod
    def _call(name: str, args: dict[str, Any]) -> LLMResponse:
        return LLMResponse({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": f"call_{name}", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}],
        })

    async def list_models(self) -> list[str]:
        return [FAKE_MODEL]


class RoutingLLM:
    """Sends the fake model id to ``FakeLLM`` and everything else to OpenAI."""

    def __init__(self):
        self.fake = FakeLLM()
        self._real: OpenAIClient | None = None

    @property
    def real(self) -> OpenAIClient:
        if self._real is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Add it as a Codespaces secret (or in .env), "
                    f"or choose the `{FAKE_MODEL}` model to try the UI offline."
                )
            self._real = OpenAIClient()
        return self._real

    async def chat(self, messages, settings, tools=None, on_text=None) -> LLMResponse:
        client = self.fake if settings.model == FAKE_MODEL else self.real
        return await client.chat(messages, settings, tools, on_text)

    async def list_models(self) -> list[str]:
        return await self.real.list_models()
