"""Handoff: an agent passes the chat to another agent, which then talks to the user directly.

Unlike ``delegate`` (a one-shot task whose summary comes back), a handoff
switches who answers. It suits interactive specialists such as the chess
coach: "let's play chess" in the main chat hands over to the coach, which
answers in the same reply and keeps the chat until it calls ``handoff_back``.

The tools only record the request in ``agent.state["handoff"]`` and end the
turn; ``Session.run_message`` (server/sessions.py) does the switching.
Agents opt in with ``handoff_target = True``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from baseagent.tools import Tool, ToolContext, ToolError, ToolResult, tool

if TYPE_CHECKING:
    from baseagent.agent import BaseAgent

HANDOFF_DESCRIPTION = """\
Hand this chat over to a specialist who talks with the user directly from now on (for \
interactive work: lessons, games). The specialist answers the user's current message itself \
in this same reply, so do not answer it yourself. Use this instead of `delegate` for these \
specialists.

Specialists:
{roster}"""


def handoff_targets(exclude: type["BaseAgent"] | None = None) -> dict[str, type["BaseAgent"]]:
    from baseagent.agents import discover

    return {name: cls for name, cls in sorted(discover().items())
            if cls.handoff_target and cls is not exclude}


def make_handoff_tool(targets: dict[str, type["BaseAgent"]]) -> Tool:
    roster = "\n".join(f"- {name}: {cls.description}" for name, cls in targets.items()) or "(none)"

    @tool(name="handoff", description=HANDOFF_DESCRIPTION.format(roster=roster), exclusive=True)
    def handoff(agent: str, note: str = "", ctx: ToolContext = None) -> ToolResult:
        """
        Args:
            agent: The specialist's name, e.g. chess_coach.
            note: Optional context the specialist should know, e.g. "they have never played".
        """
        if agent not in targets:
            raise ToolError(f"Unknown specialist {agent!r}. Choose one of: {', '.join(targets) or '(none)'}.")
        ctx.agent.state["handoff"] = {"to": agent, "note": note.strip()}
        return ToolResult(f"Handing you over to the {targets[agent].title}.", end_turn=True)

    return handoff


@tool(exclusive=True)
def handoff_back(note: str = "", ctx: ToolContext = None) -> ToolResult:
    """Hand the chat back to the main chat when the user wants something outside your
    speciality. The main chat answers the user's current message in this same reply.

    Args:
        note: Optional context for the main chat, e.g. "they want EDA on a CSV now".
    """
    ctx.agent.state["handoff"] = {"to": None, "note": note.strip()}
    parent = ctx.agent.handoff_parent[1] if ctx.agent.handoff_parent else "main chat"
    return ToolResult(f"Handing you back to the {parent}.", end_turn=True)
