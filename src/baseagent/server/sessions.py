"""In-memory chat sessions: one agent instance (and its memory) per chat.

A chat starts with one agent (``session.agent``, usually the coordinator). That
agent can hand the chat to a specialist (see ``baseagent.handoff``); from then
on ``session.current`` answers, until it hands the chat back. Every agent in
the chat writes into one event log (``session.events``), with turn numbers
that keep counting across agents, so the UI shows one continuous conversation.
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

from baseagent import events as ev
from baseagent.agent import AgentSettings, BaseAgent
from baseagent.agents import get_agent_class
from baseagent.delegation import INHERITED_SETTINGS
from baseagent.handoff import handoff_targets
from baseagent.llm import LLMClient
from baseagent.workspace import Workspace

ATTACHED = re.compile(r"\n\nAttached files?: [^\n]+$")  # added by app.with_attachments
MAX_HANDOFFS_PER_MESSAGE = 2  # coordinator -> specialist -> back is the longest useful chain

Emit = Callable[[BaseModel], Awaitable[None]]


@dataclass
class Session:
    id: str
    agent: BaseAgent  # the agent the chat was started with
    make_agent: Callable[[str], BaseAgent] | None = None  # builds a specialist for a handoff
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    task: asyncio.Task | None = None
    specialists: dict[str, BaseAgent] = field(default_factory=dict)
    active: str | None = None  # name of the specialist answering now; None = self.agent
    events: list[dict[str, Any]] = field(default_factory=list)  # every agent's events, in order
    turns: int = 0  # turn numbers keep counting across agents

    @property
    def busy(self) -> bool:
        return self.task is not None and not self.task.done()

    @property
    def current(self) -> BaseAgent:
        return self.specialists[self.active] if self.active else self.agent

    @property
    def workspace(self) -> Workspace | None:
        return self.agent.workspace

    @property
    def title(self) -> str:
        first = next((e for e in self.events if e["type"] == "turn_start"), None)
        text = ATTACHED.sub("", first["user_message"]) if first else "New chat"
        return text if len(text) <= 60 else text[:57] + "..."

    def summary(self) -> dict[str, Any]:
        return {"id": self.id, "agent": self.agent.name, "title": self.title,
                "created": self.created, "updated": self.updated, "busy": self.busy}

    def detail(self) -> dict[str, Any]:
        usage = self.agent.total_usage
        for specialist in self.specialists.values():
            usage = usage + specialist.total_usage
        current = self.current
        return {**self.summary(), "settings": self.agent.settings.model_dump(),
                "events": self.events, "usage": usage.model_dump(),
                "active_agent": {"name": current.name, "title": current.title},
                **current.ui_state()}

    # ---------------------------------------------------------------- running

    async def run_message(self, content: str, emit: Emit) -> None:
        """Answer one user message, following handoffs between agents within the same reply."""

        async def record(event: BaseModel) -> None:
            if getattr(event, "type", None) != "text_delta":
                self.events.append(ev.event_to_dict(event))
            await emit(event)

        agent, message, via = self.current, content, None
        for hop in range(MAX_HANDOFFS_PER_MESSAGE + 1):
            agent.emit_callback = record
            agent.turn = self.turns  # run() adds one, so numbers stay unique across agents
            try:
                await agent.run(message, via=via)
            finally:
                self.turns = agent.turn
            request = agent.state.pop("handoff", None)
            if not request or hop == MAX_HANDOFFS_PER_MESSAGE:
                break
            target = request.get("to")
            next_agent = self.agent if target is None else self.specialist(target)
            self.active = None if next_agent is self.agent else target
            await record(ev.Handoff(turn=self.turns, from_agent=agent.name, to_agent=next_agent.name,
                                    title=next_agent.title, ui_state=next_agent.ui_state()))
            note = request.get("note")
            message = content + (f"\n\n(Note from the {agent.title}: {note})" if note else "")
            via, agent = agent.name, next_agent

    def specialist(self, name: str) -> BaseAgent:
        """The chat's instance of a handoff specialist: created on first use, then kept."""

        agent = self.specialists.get(name)
        if agent is None:
            if self.make_agent is None:
                raise KeyError(f"This chat cannot create the {name!r} agent.")
            agent = self.make_agent(name)
            agent.handoff_parent = (self.agent.name, type(self.agent).referred_as())
            self.specialists[name] = agent
        inherited = {key: getattr(self.agent.settings, key) for key in INHERITED_SETTINGS}
        agent.settings = agent.settings.model_copy(update=inherited)
        return agent


class SessionManager:
    def __init__(self, llm: LLMClient, log_dir: Path | None = None,
                 workspace_root: Path | None = None):
        self.llm = llm
        self.log_dir = log_dir
        self.workspace_root = workspace_root  # each chat gets <root>/<session id>/
        self.sessions: dict[str, Session] = {}

    def create(self, agent_name: str, overrides: dict[str, Any] | None = None) -> Session:
        cls = get_agent_class(agent_name)
        settings = cls.make_settings(**(overrides or {}))
        session_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        workspace = None
        if self.workspace_root is not None:
            workspace = Workspace(self.workspace_root / session_id,
                                  url_prefix=f"/api/sessions/{session_id}/files")
        agent = cls(self.llm, settings, log_dir=self.log_dir, session_id=session_id,
                    workspace=workspace)

        def make_agent(name: str) -> BaseAgent:
            targets = handoff_targets()
            if name not in targets:
                raise KeyError(f"{name!r} is not an agent the chat can be handed to.")
            specialist_cls = targets[name]
            return specialist_cls(self.llm, specialist_cls.make_settings(), log_dir=self.log_dir,
                                  session_id=f"{session_id}.{name}", workspace=workspace)

        session = Session(session_id, agent, make_agent=make_agent)
        self.sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session:
        try:
            return self.sessions[session_id]
        except KeyError:
            raise KeyError(f"No session {session_id!r}") from None

    def update_settings(self, session: Session, changes: dict[str, Any]) -> AgentSettings:
        merged = {**session.agent.settings.model_dump(), **changes}
        session.agent.settings = AgentSettings.model_validate(merged)
        for specialist in session.specialists.values():  # they follow the chat's model choice
            inherited = {key: getattr(session.agent.settings, key) for key in INHERITED_SETTINGS}
            specialist.settings = specialist.settings.model_copy(update=inherited)
        return session.agent.settings

    async def delete(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session is None:
            return
        if session.busy:
            session.task.cancel()
        for agent in [session.agent, *session.specialists.values()]:
            try:
                await agent.close()
            except Exception:  # noqa: BLE001 - closing the rest matters more
                pass

    async def close_all(self) -> None:
        for session_id in list(self.sessions):
            await self.delete(session_id)
