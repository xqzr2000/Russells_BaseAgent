"""HTTP API for the chat room.

    POST /api/sessions                 start a chat with an agent
    POST /api/sessions/{id}/messages   send a message (optionally with attached
                                       files); the response is an SSE stream of
                                       agent events until turn_end
    POST /api/sessions/{id}/stop       cancel the running turn
    PUT  /api/sessions/{id}/files/{name}   upload a data file (raw body) to data/
    GET  /api/sessions/{id}/files/{path}   download a file from the chat's workspace
    GET  /api/agents | /api/models | /api/config

In production (``make serve``) it also serves the built web UI from web/dist.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from baseagent.agent import AgentSettings
from baseagent.agents import discover
from baseagent.events import event_to_dict
from baseagent.llm import FAKE_MODEL, LLMClient, RoutingLLM
from baseagent.server.sessions import Session, SessionManager
from baseagent.workspace import DATA_DIR, Workspace

ROOT = Path(__file__).resolve().parents[3]
WEB_DIST = ROOT / "web" / "dist"
WORKSPACE_DIR = ROOT / "workspace"  # git-ignored; one folder per chat
KEEPALIVE_SECONDS = 15
DEFAULT_AGENT = "coordinator"
UPLOAD_SUFFIXES = {".csv", ".tsv", ".txt", ".json", ".xlsx", ".xls", ".parquet"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

# Model ids from /v1/models that are not chat-completions text models.
_EXCLUDE = ("audio", "realtime", "tts", "transcribe", "image", "embedding", "moderation",
            "dall-e", "whisper", "search", "instruct", "codex", "computer-use", "deep-research")
_CHAT_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")
SUGGESTED_MODELS = ["gpt-5-mini", "gpt-5", "gpt-5-nano", "gpt-4.1", "gpt-4.1-mini", "o4-mini"]


class CreateSession(BaseModel):
    agent: str = DEFAULT_AGENT
    settings: dict[str, Any] | None = None


class SendMessage(BaseModel):
    content: str
    files: list[str] = Field(default_factory=list)  # workspace paths from uploads


def safe_filename(name: str) -> str:
    """A plain file name: no folders, spaces become underscores, odd characters dropped."""
    base = Path(name.replace("\\", "/")).name
    base = re.sub(r"[^\w.\-]+", "_", re.sub(r"\s+", "_", base)).strip("._")
    if not base or not Path(base).stem:
        raise HTTPException(422, "Invalid file name.")
    return base


def with_attachments(content: str, files: list[str]) -> str:
    """The message the agent receives: the user's text plus the attached file paths."""
    if not files:
        return content
    label = "Attached file" if len(files) == 1 else "Attached files"
    return f"{content.rstrip()}\n\n{label}: {', '.join(files)}"


def default_model() -> str:
    """The model for new chats: OPENAI_MODEL, but the offline model if there is no key."""
    if not os.environ.get("OPENAI_API_KEY"):
        return FAKE_MODEL
    return os.environ.get("OPENAI_MODEL") or "gpt-5-mini"


def create_app(llm: LLMClient | None = None, log_dir: Path | None = None,
               workspace_dir: Path | None = None) -> FastAPI:
    load_dotenv(ROOT / ".env")  # never overrides real env vars such as Codespaces secrets
    manager = SessionManager(llm or RoutingLLM(), log_dir or ROOT / "runs",
                             workspace_dir or WORKSPACE_DIR)
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

    def workspace_of(session: Session) -> Workspace:
        if session.workspace is None:
            raise HTTPException(404, "This chat has no file workspace.")
        return session.workspace

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
            "default_agent": DEFAULT_AGENT,
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

    @app.put("/api/sessions/{session_id}/files/{filename}")
    async def upload_file(session_id: str, filename: str, request: Request):
        workspace = workspace_of(session_or_404(session_id))
        name = safe_filename(filename)
        if Path(name).suffix.lower() not in UPLOAD_SUFFIXES:
            raise HTTPException(415, "Upload a data file: " + ", ".join(sorted(UPLOAD_SUFFIXES)))
        target = workspace.resolve(f"{DATA_DIR}/{name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{name}.part")
        size = 0
        try:
            with partial.open("wb") as out:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_BYTES // 2**20} MB.")
                    out.write(chunk)
            if size == 0:
                raise HTTPException(422, "File is empty.")
            os.replace(partial, target)  # re-uploading a name replaces the old file
        finally:
            partial.unlink(missing_ok=True)
        return {"name": name, "path": workspace.relative(target), "size": size}

    @app.get("/api/sessions/{session_id}/files/{path:path}")
    async def download_file(session_id: str, path: str):
        workspace = workspace_of(session_or_404(session_id))
        try:
            file = workspace.resolve(path)
        except ValueError:
            raise HTTPException(404, "File not found.") from None
        if not file.is_file() or any(part.startswith(".") for part in Path(path).parts):
            raise HTTPException(404, "File not found.")
        return FileResponse(file, filename=file.name)  # Content-Disposition: attachment

    @app.post("/api/sessions/{session_id}/messages")
    async def send_message(session_id: str, body: SendMessage):
        session = session_or_404(session_id)
        if session.busy:
            raise HTTPException(409, "The agent is still working on the previous message.")
        if not body.content.strip():
            raise HTTPException(422, "Message is empty.")
        files: list[str] = []
        if body.files:
            workspace = workspace_of(session)
            for rel in body.files:
                try:
                    path = workspace.resolve(rel)
                except ValueError:
                    raise HTTPException(422, f"Invalid attachment path {rel!r}.") from None
                if not path.is_file():
                    raise HTTPException(422, f"Attached file {rel!r} was not found; upload it again.")
                files.append(workspace.relative(path))

        queue: asyncio.Queue = asyncio.Queue()

        async def emit(event) -> None:
            await queue.put(event)

        session.agent.emit_callback = emit
        task = asyncio.create_task(session.agent.run(with_attachments(body.content, files)))
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
