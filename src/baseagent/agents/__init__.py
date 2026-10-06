"""Agent registry.

Every module in this package is imported on startup, so adding an agent is:
create ``agents/<your_agent>.py`` (or a sub-package), subclass ``BaseAgent``,
and decorate it with ``@register_agent``. It then appears in the chat room's
agent picker automatically.
"""

from __future__ import annotations

import importlib
import pkgutil

from baseagent.agent import BaseAgent

AGENTS: dict[str, type[BaseAgent]] = {}


def register_agent(cls: type[BaseAgent]) -> type[BaseAgent]:
    if not (isinstance(cls, type) and issubclass(cls, BaseAgent)):
        raise TypeError("@register_agent expects a BaseAgent subclass")
    if cls.name in AGENTS and AGENTS[cls.name] is not cls:
        raise ValueError(f"Agent name {cls.name!r} is already registered by {AGENTS[cls.name]}")
    AGENTS[cls.name] = cls
    return cls


def discover() -> dict[str, type[BaseAgent]]:
    """Import every module in this package so their ``@register_agent`` runs."""
    for module in pkgutil.iter_modules(__path__):
        if not module.name.startswith("_"):
            importlib.import_module(f"{__name__}.{module.name}")
    return AGENTS


def get_agent_class(name: str) -> type[BaseAgent]:
    discover()
    try:
        return AGENTS[name]
    except KeyError:
        raise KeyError(f"Unknown agent {name!r}. Available: {', '.join(sorted(AGENTS))}") from None
