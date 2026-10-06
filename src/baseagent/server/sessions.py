"""In-memory chat sessions: one agent instance (and its memory) per chat."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from baseagent.agent import AgentSettings, BaseAgent
from baseagent.agents import get_agent_class
from baseagent.llm import LLMClient


@dataclass
class Session:
    id: str
    agent: BaseAgent
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    task: asyncio.Task | None = None

    @property
    def busy(self) -> bool:
        return self.task is not None and not self.task.done()

    @property
    def title(self) -> str:
        first = next((e for e in self.agent.transcript if e["type"] == "turn_start"), None)
        text = first["user_message"] if first else "New chat"
        return text if len(text) <= 60 else text[:57] + "..."

    def summary(self) -> dict[str, Any]:
        return {"id": self.id, "agent": self.agent.name, "title": self.title,
                "created": self.created, "updated": self.updated, "busy": self.busy}

    def detail(self) -> dict[str, Any]:
        return {**self.summary(), "settings": self.agent.settings.model_dump(),
                "events": self.agent.transcript, "usage": self.agent.total_usage.model_dump()}


class SessionManager:
    def __init__(self, llm: LLMClient, log_dir: Path | None = None):
        self.llm = llm
        self.log_dir = log_dir
        self.sessions: dict[str, Session] = {}

    def create(self, agent_name: str, overrides: dict[str, Any] | None = None) -> Session:
        cls = get_agent_class(agent_name)
        settings = cls.make_settings(**(overrides or {}))
        session_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        agent = cls(self.llm, settings, log_dir=self.log_dir, session_id=session_id)
        session = Session(session_id, agent)
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
        return session.agent.settings

    async def delete(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session is None:
            return
        if session.busy:
            session.task.cancel()
        await session.agent.close()

    async def close_all(self) -> None:
        for session_id in list(self.sessions):
            await self.delete(session_id)
