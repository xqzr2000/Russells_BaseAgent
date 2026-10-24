"""The runner bridge: how OpenCode's ``run_code`` tool reaches the runner.

OpenCode runs in its own container with no shell. Its one way to run code is
``POST /internal/code-runs`` on this server, with a one-time token:

* ``open()`` issues a ticket (token) for one code-agent request; only that
  request's OpenCode container is given the token.
* each call runs the snippet in a fresh runner container (``runner.py``) and
  is logged on the ticket, so the answer can report exactly what happened;
* a ticket allows ``max_runs`` runs, expires after ``ttl`` seconds and is
  closed when the request ends; runs on one ticket happen one at a time.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from baseagent.codeagent.runner import CodeRunner, RunnerUnavailable, RunResult

DEFAULT_MAX_RUNS = 6
DEFAULT_TTL = 240.0  # seconds; a bit more than the 3-minute task limit


class RunRequest(BaseModel):
    language: str = Field(max_length=40)
    code: str = Field(max_length=200_000)
    mode: Literal["check", "run"] = "run"
    stdin: str | None = Field(default=None, max_length=20_000)


class BridgeError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class Ticket:
    token: str
    label: str
    max_runs: int
    expires: float  # time.monotonic() deadline
    runs: list[RunResult] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    closed: bool = False

    @property
    def runs_left(self) -> int:
        return max(0, self.max_runs - len(self.runs))


class RunBridge:
    def __init__(self, runner: CodeRunner | None = None, *, max_runs: int = DEFAULT_MAX_RUNS,
                 ttl: float = DEFAULT_TTL):
        self.runner = runner or CodeRunner()
        self.max_runs = max_runs
        self.ttl = ttl
        self.tickets: dict[str, Ticket] = {}

    def open(self, label: str = "", *, max_runs: int | None = None, ttl: float | None = None) -> Ticket:
        self._forget_expired()
        token = secrets.token_urlsafe(32)
        ticket = Ticket(token, label, max_runs or self.max_runs,
                        time.monotonic() + (ttl or self.ttl))
        self.tickets[token] = ticket
        return ticket

    def close(self, token: str) -> list[RunResult]:
        """End a ticket; returns its run log."""
        ticket = self.tickets.pop(token, None)
        if ticket is None:
            return []
        ticket.closed = True
        return ticket.runs

    def _find(self, token: str | None) -> Ticket:
        if token:
            for known, ticket in self.tickets.items():
                if hmac.compare_digest(known, token):
                    if ticket.closed or time.monotonic() > ticket.expires:
                        break
                    return ticket
        raise BridgeError(401, "Invalid or expired token.")

    def _forget_expired(self) -> None:
        now = time.monotonic()
        for token in [t for t, ticket in self.tickets.items() if now > ticket.expires + 600]:
            self.tickets.pop(token, None)

    async def handle(self, token: str | None, request: RunRequest) -> RunResult:
        ticket = self._find(token)
        async with ticket.lock:
            if ticket.runs_left == 0:
                raise BridgeError(429, f"Run limit reached ({ticket.max_runs} runs for this task). "
                                       "Finish with what you have.")
            try:
                result = await asyncio.to_thread(self.runner.run, request.language, request.code,
                                                 request.mode, request.stdin)
            except RunnerUnavailable as exc:
                raise BridgeError(503, f"The code runner is not available: {exc}") from exc
            result.meta["code_sha256"] = code_sha256(request.code)
            if not ticket.closed:
                ticket.runs.append(result)
            return result


def code_sha256(code: str) -> str:
    """Identifies which version of the code a run used (original, fixed, ...)."""
    return hashlib.sha256(code.encode()).hexdigest()


def make_router(bridge: RunBridge) -> APIRouter:
    """``POST /internal/code-runs``: what OpenCode's run_code tool calls."""
    router = APIRouter()

    @router.post("/internal/code-runs")
    async def code_run(body: RunRequest, request: Request):
        auth = request.headers.get("authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else None
        try:
            result = await bridge.handle(token, body)
        except BridgeError as exc:
            raise HTTPException(exc.status, str(exc)) from exc
        return result.to_dict()

    return router


_shared: RunBridge | None = None


def shared_bridge() -> RunBridge:
    """The process's bridge: the server's endpoint and the code agent use the same one."""
    global _shared
    if _shared is None:
        _shared = RunBridge()
    return _shared
