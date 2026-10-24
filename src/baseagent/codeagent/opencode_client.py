"""Runs one code-fix request through OpenCode in a fresh, locked-down container.

``OpenCodeFixer.fix(task)`` does the whole request and always cleans up:

1. start a ``baseagent-opencode`` container (read-only, non-root, RAM folders
   only, no API keys) and write the user's code to ``/workspace/<file>``
2. pick a free model from what OpenCode lists there with no key
3. open a bridge ticket (``run_code``'s token and run cap for this mode)
4. start ``opencode serve`` with the fixer config, password-protected and
   published on loopback only
5. send the task with the answer schema and stream OpenCode's events as
   progress lines; abort after ``time_limit`` seconds (3 minutes)
6. read back the fixed file, the structured answer and the bridge's run log,
   then stop the container

The result (``CodeFix``) is turned into the chat answer by ``answer.py``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Awaitable, Callable

import httpx
from agent_sandbox_runtime import Sandbox
from agent_sandbox_runtime.process import DockerError

from baseagent.codeagent.bridge import RunBridge, shared_bridge
from baseagent.codeagent.opencode_config import (
    ANSWER_SCHEMA, FIXER_AGENT, RUN_LIMITS, Mode, NoFreeModel, build_config, parse_model_list,
    pick_model,
)
from baseagent.codeagent.language import guess_language
from baseagent.codeagent.runner import RunResult, normalize_language

OPENCODE_IMAGE = os.environ.get("BASEAGENT_OPENCODE_IMAGE", "baseagent-opencode:1")
OPENCODE_PORT = 4096
WORKSPACE = "/workspace"
TIME_LIMIT = 180.0  # seconds for the whole OpenCode task
STARTUP_LIMIT = 60.0  # seconds for the container and server to come up
MODEL_LIST_TTL = 600.0  # seconds to reuse OpenCode's free-model list
MAX_CODE_CHARS = 100_000

Progress = Callable[[str], Awaitable[None]]

EXTENSIONS = {
    "python": ".py", "javascript": ".js", "typescript": ".ts", "java": ".java", "c": ".c",
    "cpp": ".cpp", "c++": ".cpp", "csharp": ".cs", "c#": ".cs", "go": ".go", "rust": ".rs",
    "ruby": ".rb", "php": ".php", "swift": ".swift", "kotlin": ".kt", "scala": ".scala",
    "r": ".r", "sql": ".sql", "bash": ".sh", "shell": ".sh", "sh": ".sh", "html": ".html",
    "css": ".css", "lua": ".lua", "perl": ".pl", "haskell": ".hs", "dart": ".dart",
    "matlab": ".m", "vba": ".bas",
}
FENCES = {ext: lang for lang, ext in EXTENSIONS.items()}
FENCES.update({".py": "python", ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
               ".ts": "typescript", ".cpp": "cpp", ".cs": "csharp", ".sh": "bash", ".txt": ""})


class CodeAgentError(RuntimeError):
    """The request couldn't be done; the message is meant for the user."""


@dataclass
class CodeTask:
    code: str
    request: str = "Check this code and fix what is wrong."  # what the user asked, in their words
    language: str | None = None  # e.g. "python"; from the file name, else guessed from the code
    filename: str | None = None  # e.g. "main.py" (only the name is used)
    mode: Mode = "auto"


@dataclass
class CodeFix:
    filename: str
    fence: str  # Markdown code-fence language, e.g. "python"
    runnable: bool  # Python, JavaScript or TypeScript
    mode: str
    original: str
    fixed: str
    issues: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    not_run_reason: str = ""
    tips: list[str] = field(default_factory=list)  # learning points, not fixes
    runs: list[RunResult] = field(default_factory=list)
    model: str = ""
    zero_retention: bool = True
    stopped: str | None = None  # why it ended early: "time limit", or an error message
    answered: bool = False  # OpenCode returned its structured answer
    duration: float = 0.0


def resolve_file(task: CodeTask) -> tuple[str, str, bool]:
    """(file name, code-fence language, runnable) for a task."""
    name = PurePosixPath((task.filename or "").replace("\\", "/")).name
    name = re.sub(r"[^\w.\-]+", "_", name).strip("._")
    lang = (task.language or "").strip().lower() or (guess_language(task.code) or "")
    if name and "." in name:
        ext = "." + name.rsplit(".", 1)[1].lower()
    else:
        runnable_lang = normalize_language(lang) if lang else None
        # Unknown language: a .txt file, reviewed but not run (never assume Python).
        ext = EXTENSIONS.get(runnable_lang or lang, ".txt")
        name = f"{name or 'main'}{ext}"
    fence = FENCES.get(ext, ext.lstrip("."))
    runnable = ext in (".py", ".js", ".mjs", ".cjs", ".ts")
    return name, fence, runnable


def task_message(task: CodeTask, filename: str, fence: str, runnable: bool) -> str:
    note = "" if runnable else (f" This language can't be run here (only Python, JavaScript and "
                                f"TypeScript can), so review and fix it by reading.")
    return (f"The user asked: {task.request.strip() or 'Check and fix this code.'}\n\n"
            f"Their code is saved as {filename} in the workspace; edit that file.{note}\n\n"
            f"```{fence}\n{task.code.rstrip()}\n```")


def describe_tool(part: dict[str, Any], filename: str) -> str | None:
    """A progress line for one of OpenCode's tool calls, or None to skip it."""
    tool = part.get("tool")
    state = part.get("state") or {}
    status = state.get("status")
    args = state.get("input") or {}
    if tool == "run_code":
        target = args.get("path") or filename
        if status == "running":
            return f"{'checking' if args.get('mode') == 'check' else 'running'} {target}"
        if status == "completed":
            result = (state.get("metadata") or {}).get("status") or "done"
            words = {"ok": "ok", "error": "failed", "timeout": "stopped: too slow (10 s limit)",
                     "out_of_memory": "stopped: out of memory", "too_much_output": "stopped: too much output",
                     "not_runnable": "can't be run here"}
            return f"{'check' if args.get('mode') == 'check' else 'run'} result: {words.get(result, result)}"
        return None
    if status != "running":
        return None
    if tool == "read":
        return f"reading {PurePosixPath(args.get('filePath') or filename).name}"
    if tool in ("edit", "write", "apply_patch"):
        return f"editing {PurePosixPath(args.get('filePath') or filename).name}"
    if tool in ("glob", "grep"):
        return "searching the code"
    if tool == "StructuredOutput":
        return "writing the explanation"
    return None


_model_cache: tuple[float, list[str]] | None = None


class OpenCodeFixer:
    """Fixes code snippets with OpenCode. One instance can serve many requests."""

    def __init__(self, bridge: RunBridge | None = None, *, bridge_url: str | None = None,
                 image: str = OPENCODE_IMAGE, time_limit: float = TIME_LIMIT,
                 sandbox_factory: Callable[..., Any] = Sandbox,
                 available_models: list[str] | None = None,
                 configure: Callable[[dict], dict] | None = None):
        """``bridge_url`` is how the container reaches this server's bridge
        (default: host.docker.internal on $PORT). ``available_models`` skips
        asking OpenCode for its list; ``configure`` may edit the config
        (tests use both to swap in a scripted model)."""
        self.bridge = bridge or shared_bridge()
        port = os.environ.get("PORT", "8000")
        self.bridge_url = (bridge_url or os.environ.get("BASEAGENT_BRIDGE_URL")
                           or f"http://host.docker.internal:{port}/internal/code-runs")
        self.image = image
        self.time_limit = time_limit
        self.sandbox_factory = sandbox_factory
        self.available_models = available_models
        self.configure = configure

    async def fix(self, task: CodeTask, progress: Progress | None = None) -> CodeFix:
        say = progress or _quiet
        if not task.code.strip():
            raise CodeAgentError("There is no code to check.")
        if len(task.code) > MAX_CODE_CHARS:
            raise CodeAgentError(f"That code is too long for the code agent ({MAX_CODE_CHARS:,} characters max).")
        filename, fence, runnable = resolve_file(task)
        result = CodeFix(filename=filename, fence=fence, runnable=runnable, mode=task.mode,
                         original=task.code, fixed=task.code)
        started = time.monotonic()
        await say("starting the OpenCode sandbox")
        try:
            sandbox = await asyncio.to_thread(self._start_sandbox)
        except DockerError as exc:
            raise CodeAgentError(f"The code sandbox couldn't start ({str(exc)[:200]}). Is Docker running, "
                                 "and were the images built with `make sandbox-images`?") from exc
        ticket = None
        try:
            choice = pick_model(await self._models(sandbox), preferred=os.environ.get("OPENCODE_FREE_MODEL"))
            result.model, result.zero_retention = choice.model, choice.zero_retention
            await say(f"using free model {choice.model}")
            path = f"{WORKSPACE}/{filename}"
            await asyncio.to_thread(sandbox.write_file, path, task.code)
            env = {"OPENCODE_SERVER_PASSWORD": secrets.token_urlsafe(24)}
            config = build_config(choice.model, task.mode)
            if self.configure:
                config = self.configure(config)
            env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
            if RUN_LIMITS[task.mode] > 0 and runnable:
                ticket = self.bridge.open(filename, max_runs=RUN_LIMITS[task.mode], ttl=self.time_limit + 60)
                env.update(BASEAGENT_RUN_URL=self.bridge_url, BASEAGENT_RUN_TOKEN=ticket.token)
            await asyncio.to_thread(sandbox.spawn, [
                "sh", "-c", f"exec opencode serve --hostname 0.0.0.0 --port {OPENCODE_PORT} --print-logs "
                            "> /tmp/opencode.log 2>&1"], env=env)
            url = await asyncio.to_thread(sandbox.tunnel_url, OPENCODE_PORT)
            async with httpx.AsyncClient(base_url=url, auth=("opencode", env["OPENCODE_SERVER_PASSWORD"]),
                                         timeout=30, trust_env=False) as client:
                await self._wait_ready(client, sandbox)
                await self._run(client, task, filename, fence, runnable, result, say)
            result.fixed = await asyncio.to_thread(sandbox.read_file, path)
        except NoFreeModel as exc:
            raise CodeAgentError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise CodeAgentError(f"Lost contact with OpenCode in the sandbox ({exc}).") from exc
        except DockerError as exc:
            raise CodeAgentError(f"The code sandbox failed ({str(exc)[:200]}).") from exc
        finally:
            if ticket is not None:
                result.runs = list(self.bridge.close(ticket.token))
            await asyncio.to_thread(sandbox.stop)
            result.duration = time.monotonic() - started
        return result

    def _start_sandbox(self) -> Any:
        return self.sandbox_factory(
            self.image, cwd=WORKSPACE, ports=(OPENCODE_PORT,), network="bridge", memory="1g", cpus=1.0,
            pids_limit=256, user="1000:1000", read_only=True,
            tmpfs={WORKSPACE: "size=16m,uid=1000,gid=1000,mode=0700",
                   "/home/opencode": "size=256m,uid=1000,gid=1000,mode=0700",
                   "/tmp": "size=256m,uid=1000,gid=1000,mode=0700"},
            add_hosts={"host.docker.internal": "host-gateway"},
            labels={"org.baseagent.role": "opencode"},
            deployment_timeout=self.time_limit + STARTUP_LIMIT + 60, runtime_timeout=60)

    async def _models(self, sandbox: Any) -> list[str]:
        global _model_cache
        if self.available_models is not None:
            return list(self.available_models)
        if _model_cache and time.monotonic() - _model_cache[0] < MODEL_LIST_TTL:
            return _model_cache[1]
        out = await asyncio.to_thread(sandbox.execute, ["opencode", "models", "opencode"],
                                      shell=False, timeout=60, cwd="/tmp")
        models = parse_model_list(out.get("stdout", ""))
        if not models:
            raise CodeAgentError("OpenCode didn't list any free models (no internet in the sandbox?): "
                                 + (out.get("output") or "")[-300:])
        _model_cache = (time.monotonic(), models)
        return models

    async def _wait_ready(self, client: httpx.AsyncClient, sandbox: Any) -> None:
        deadline = time.monotonic() + STARTUP_LIMIT
        while time.monotonic() < deadline:
            try:
                # Short timeout: a request sent while OpenCode is still starting
                # can be accepted and never answered.
                if (await client.get("/global/health", timeout=2)).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.25)
        log = await asyncio.to_thread(sandbox.execute, ["sh", "-c", "tail -c 1500 /tmp/opencode.log"],
                                      shell=False, timeout=10)
        raise CodeAgentError("OpenCode didn't start in the sandbox: " + (log.get("output") or "").strip()[-600:])

    async def _run(self, client: httpx.AsyncClient, task: CodeTask, filename: str, fence: str,
                   runnable: bool, result: CodeFix, say: Progress) -> None:
        session = (await client.post("/session", json={"title": f"Fix {filename}"})).json()
        sid = session["id"]
        done = asyncio.Event()
        failure: list[str] = []
        listening = asyncio.Event()
        listener = asyncio.create_task(self._listen(client, sid, filename, say, done, failure, listening))
        try:
            try:
                await asyncio.wait_for(listening.wait(), 10)
            except asyncio.TimeoutError as exc:
                raise CodeAgentError("OpenCode's event stream didn't connect.") from exc
            if failure:
                raise CodeAgentError(failure[0])
            # The waiting prompt call returns the final assistant message. (In
            # 1.19.2, listing a session's messages fails once a message used
            # `format: json_schema`, so the answer can't be fetched afterwards.)
            prompt = asyncio.create_task(client.post(f"/session/{sid}/message", timeout=self.time_limit + 60, json={
                "agent": FIXER_AGENT,
                "parts": [{"type": "text", "text": task_message(task, filename, fence, runnable)}],
                "format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
            }))
            await say("OpenCode is working on it")
            response: httpx.Response | None = None
            try:
                response = await asyncio.wait_for(asyncio.shield(prompt), self.time_limit)
            except asyncio.TimeoutError:
                result.stopped = "time limit"
                await say(f"stopping: the {self.time_limit / 60:g}-minute limit was reached")
                await client.post(f"/session/{sid}/abort", timeout=10)
                try:
                    response = await asyncio.wait_for(prompt, 20)
                except (asyncio.TimeoutError, httpx.HTTPError):
                    prompt.cancel()
        finally:
            listener.cancel()
        if failure and not result.stopped:
            result.stopped = failure[0]
        reply = response.json() if response is not None and response.status_code == 200 else None
        if response is not None and not isinstance(reply, dict):
            raise CodeAgentError(f"OpenCode's reply wasn't understood (HTTP {response.status_code}): "
                                 f"{response.text[:300]}")
        info = (reply or {}).get("info") or {}
        structured = info.get("structured")
        if isinstance(structured, dict):
            result.answered = True
            result.issues = [i for i in structured.get("issues") or [] if isinstance(i, dict)]
            result.summary = str(structured.get("summary") or "")
            result.not_run_reason = str(structured.get("not_run_reason") or "")
            result.tips = [str(t).strip() for t in structured.get("tips") or [] if str(t).strip()][:3]
        error = info.get("error")
        if error and not result.stopped:
            data = error.get("data") or {}
            result.stopped = str(data.get("message") or error.get("name") or "OpenCode stopped with an error")

    async def _listen(self, client: httpx.AsyncClient, sid: str, filename: str, say: Progress,
                      done: asyncio.Event, failure: list[str], listening: asyncio.Event) -> None:
        said: set[tuple[str, str]] = set()
        try:
            async with client.stream("GET", "/event", timeout=None) as stream:
                async for line in stream.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        event = json.loads(line[5:])
                    except json.JSONDecodeError:
                        continue
                    kind = event.get("type")
                    props = event.get("properties") or {}
                    if kind == "server.connected":
                        listening.set()
                    if props.get("sessionID", (props.get("part") or {}).get("sessionID")) != sid:
                        continue
                    if kind == "message.part.updated":
                        part = props.get("part") or {}
                        if part.get("type") != "tool":
                            continue
                        key = (part.get("callID", ""), (part.get("state") or {}).get("status", ""))
                        if key in said:
                            continue
                        said.add(key)
                        line_ = describe_tool(part, filename)
                        if line_:
                            await say(line_)
                    elif kind == "session.status" and (props.get("status") or {}).get("type") == "retry":
                        status = props["status"]
                        reason = str(status.get("message") or "the free model didn't answer").strip()
                        await say(f"retrying (attempt {status.get('attempt')}): {reason[:120]}")
                    elif kind == "session.error":
                        error = props.get("error") or {}
                        failure.append(str((error.get("data") or {}).get("message") or error.get("name")
                                           or "OpenCode error"))
                    elif kind == "session.idle":
                        return
        except httpx.HTTPError as exc:
            failure.append(f"lost contact with OpenCode ({exc})")
        finally:
            listening.set()  # never leave the caller waiting on a dead stream
            done.set()


async def _quiet(_: str) -> None:
    return None
