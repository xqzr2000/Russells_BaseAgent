"""HTTP API for the chat room.

    POST /api/sessions                 start a chat with an agent
    POST /api/sessions/{id}/messages   send a message; the response is an SSE
                                       stream of agent events until turn_end
    POST /api/sessions/{id}/stop       cancel the running turn
    GET  /api/agents | /api/models | /api/config

In production (``make serve``) it also serves the built web UI from web/dist.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from baseagent.agent import AgentSettings
from baseagent.agents import discover
from baseagent.events import event_to_dict
from baseagent.llm import FAKE_MODEL, LLMClient, RoutingLLM
from baseagent.server.sessions import Session, SessionManager

ROOT = Path(__file__).resolve().parents[3]
WEB_DIST = ROOT / "web" / "dist"
KEEPALIVE_SECONDS = 15

# Model ids from /v1/models that are not chat-completions text models.
_EXCLUDE = ("audio", "realtime", "tts", "transcribe", "image", "embedding", "moderation",
            "dall-e", "whisper", "search", "instruct", "codex", "computer-use", "deep-research")
_CHAT_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")
SUGGESTED_MODELS = ["gpt-5-mini", "gpt-5", "gpt-5-nano", "gpt-4.1", "gpt-4.1-mini", "o4-mini"]


class CreateSession(BaseModel):
    agent: str = "general"
    settings: dict[str, Any] | None = None


class SendMessage(BaseModel):
    content: str


def default_model() -> str:
    """The model for new chats: OPENAI_MODEL, but the offline model if there is no key."""
    if not os.environ.get("OPENAI_API_KEY"):
        return FAKE_MODEL
    return os.environ.get("OPENAI_MODEL") or "gpt-5-mini"


def create_app(llm: LLMClient | None = None, log_dir: Path | None = None) -> FastAPI:
    load_dotenv(ROOT / ".env")  # never overrides real env vars such as Codespaces secrets
    manager = SessionManager(llm or RoutingLLM(), log_dir or ROOT / "runs")
    model_cache: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        discover()
        yield
        await manager.close_all()

    app = FastAPI(title="Russell's BaseAgent", lifespan=lifespan)
    app.state.manager = manager

    def session_or_404(session_id: str) -> Session:
        try:
            return manager.get(session_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/health")
    async def health():
        return {"ok": True}

    @app.get("/api/config")
    async def config():
        base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        return {
            "openai_key_configured": bool(os.environ.get("OPENAI_API_KEY")),
            "base_url_host": urlparse(base_url).netloc,
            "default_model": default_model(),
            "fake_model": FAKE_MODEL,
            "suggested_models": SUGGESTED_MODELS,
        }

    @app.get("/api/models")
    async def models(refresh: bool = False):
        cached = model_cache.get("models")
        if cached and not refresh and time.time() - cached[0] < 600:
            return cached[1]
        result: dict[str, Any] = {"models": [], "error": None}
        if not os.environ.get("OPENAI_API_KEY"):
            result["error"] = "OPENAI_API_KEY is not set; only the offline fake model is available."
        else:
            try:
                ids = await manager.llm.list_models()
                result["models"] = [
                    i for i in ids
                    if i.startswith(_CHAT_PREFIXES) and not any(x in i for x in _EXCLUDE)
                ]
            except Exception as exc:  # noqa: BLE001
                result["error"] = f"Could not list models: {type(exc).__name__}: {exc}"
        result["models"] = [FAKE_MODEL, *result["models"]]
        model_cache["models"] = (time.time(), result)
        return result

    @app.get("/api/agents")
    async def agents():
        return [
            {"name": cls.name, "title": cls.title, "description": cls.description,
             "examples": cls.examples,
             "default_settings": cls.make_settings(model=default_model()).model_dump()}
            for cls in discover().values()
        ]

    @app.get("/api/sessions")
    async def list_sessions():
        return sorted((s.summary() for s in manager.sessions.values()),
                      key=lambda s: s["updated"], reverse=True)

    @app.post("/api/sessions")
    async def create_session(body: CreateSession):
        overrides = {"model": default_model(), **(body.settings or {})}
        try:
            session = manager.create(body.agent, overrides)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return session.detail()

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):
        return session_or_404(session_id).detail()

    @app.patch("/api/sessions/{session_id}/settings")
    async def update_settings(session_id: str, changes: dict[str, Any]):
        session = session_or_404(session_id)
        unknown = set(changes) - set(AgentSettings.model_fields)
        if unknown:
            raise HTTPException(422, f"Unknown settings: {', '.join(sorted(unknown))}")
        try:
            return manager.update_settings(session, changes).model_dump()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        await manager.delete(session_id)
        return {"ok": True}

    @app.post("/api/sessions/{session_id}/stop")
    async def stop(session_id: str):
        session = session_or_404(session_id)
        if session.busy:
            session.task.cancel()
        return {"stopped": True}

    @app.post("/api/sessions/{session_id}/messages")
    async def send_message(session_id: str, body: SendMessage):
        session = session_or_404(session_id)
        if session.busy:
            raise HTTPException(409, "The agent is still working on the previous message.")
        if not body.content.strip():
            raise HTTPException(422, "Message is empty.")

        queue: asyncio.Queue = asyncio.Queue()

        async def emit(event) -> None:
            await queue.put(event)

        session.agent.emit_callback = emit
        task = asyncio.create_task(session.agent.run(body.content))
        task.add_done_callback(lambda _: queue.put_nowait(None))
        session.task = task
        session.updated = time.time()

        async def stream():
            try:
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), KEEPALIVE_SECONDS)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    if item is None:
                        break
                    data = json.dumps(event_to_dict(item), ensure_ascii=False)
                    yield f"event: {item.type}\ndata: {data}\n\n"
            finally:
                if not task.done():  # the browser went away: stop spending tokens
                    task.cancel()
                session.updated = time.time()

        return StreamingResponse(stream(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})

    if WEB_DIST.is_dir():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/{path:path}")
        async def spa(path: str):
            file = WEB_DIST / path
            if path and file.is_file() and WEB_DIST in file.resolve().parents:
                return FileResponse(file)
            return FileResponse(WEB_DIST / "index.html")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("baseagent.server.app:app", host="0.0.0.0",
                port=int(os.environ.get("PORT", "8000")),
                reload=os.environ.get("RELOAD") == "1")


if __name__ == "__main__":
    main()
