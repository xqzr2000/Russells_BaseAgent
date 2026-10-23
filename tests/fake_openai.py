"""A scripted, OpenAI-compatible chat model for testing a real OpenCode offline.

Point an OpenCode custom provider (``@ai-sdk/openai-compatible``) at
``server.base_url``. Each chat request gets the next scripted turn: a list of
tool calls, or text. Every request body is recorded in ``server.requests``,
so a test can check exactly which tools OpenCode offered the model.

    with FakeOpenAI([
        {"tool_calls": [("read", {"filePath": "/w/main.py"})]},
        {"text": "done"},
    ]) as fake:
        ... # run OpenCode against fake.base_url
        tools = fake.tool_names(0)
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MODEL = "scripted"


def provider_config(base_url: str, model: str = MODEL) -> dict:
    """The OpenCode ``provider`` entry for this fake (provider id ``fake``)."""
    return {
        "fake": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Fake",
            "options": {"baseURL": base_url, "apiKey": "not-a-key"},
            "models": {model: {"name": model, "tool_call": True,
                               "limit": {"context": 128000, "output": 8000}}},
        }
    }


class FakeOpenAI:
    def __init__(self, script: list[dict[str, Any]]):
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802 - http.server API
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
                with fake.lock:
                    fake.requests.append(body)
                    turn = fake.script.pop(0) if fake.script else {"text": "Done."}
                if body.get("stream"):
                    self._stream(turn)
                else:
                    self._json(turn)

            def _send(self, status: int, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.end_headers()

            def _json(self, turn: dict) -> None:
                message: dict[str, Any] = {"role": "assistant", "content": turn.get("text")}
                if turn.get("tool_calls"):
                    message["tool_calls"] = [
                        {"id": f"call_{i}", "type": "function",
                         "function": {"name": name, "arguments": json.dumps(args)}}
                        for i, (name, args) in enumerate(turn["tool_calls"])]
                finish = "tool_calls" if turn.get("tool_calls") else "stop"
                self._send(200, "application/json")
                self.wfile.write(json.dumps({
                    "id": "fake", "object": "chat.completion", "created": 0, "model": MODEL,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }).encode())

            def _stream(self, turn: dict) -> None:
                self._send(200, "text/event-stream")

                def chunk(delta: dict, finish: str | None = None, usage: dict | None = None) -> None:
                    data: dict[str, Any] = {"id": "fake", "object": "chat.completion.chunk", "created": 0,
                                            "model": MODEL,
                                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                    if usage:
                        data["usage"] = usage
                    self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
                    self.wfile.flush()

                chunk({"role": "assistant", "content": ""})
                if turn.get("text"):
                    chunk({"content": turn["text"]})
                for i, (name, args) in enumerate(turn.get("tool_calls") or []):
                    chunk({"tool_calls": [{"index": i, "id": f"call_{len(fake.requests)}_{i}", "type": "function",
                                           "function": {"name": name, "arguments": json.dumps(args)}}]})
                finish = "tool_calls" if turn.get("tool_calls") else "stop"
                chunk({}, finish, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        self.server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def tool_names(self, index: int) -> list[str]:
        """Names of the tools OpenCode sent with request ``index``."""
        return sorted(t["function"]["name"] for t in self.requests[index].get("tools") or [])

    def tool_results(self, index: int) -> list[str]:
        """The tool results OpenCode sent back to the model in request ``index``."""
        out = []
        for message in self.requests[index].get("messages", []):
            if message.get("role") == "tool":
                content = message.get("content")
                out.append(content if isinstance(content, str) else json.dumps(content))
        return out

    def __enter__(self) -> "FakeOpenAI":
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
