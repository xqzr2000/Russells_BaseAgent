"""Agent-as-tool: an agent hands a task to a specialist agent and waits for it.

The specialist runs its own ReAct loop with its own memory and tools. Only its
final answer (and any files it produced) come back as the tool result, so the
caller's context stays small. Its steps are shown as progress on the
``delegate`` call and logged in full to ``runs/<session>.<agent>.json``.

Give an agent this ability by holding a ``SubAgentPool`` as ``self.subagents``
and adding ``make_delegate_tool(self.subagents.available())`` to its tools.
Any registered agent with ``delegatable = True`` can be delegated to.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from baseagent.tools import Tool, ToolContext, ToolError, ToolResult, tool

if TYPE_CHECKING:
    from baseagent.agent import BaseAgent

DELEGATE_TIMEOUT = 1800.0  # a whole sub-agent run, not one call
INHERITED_SETTINGS = ("model", "reasoning_effort", "temperature")

DELEGATE_DESCRIPTION = """\
Hand a task to a specialist agent and wait for its result. The specialist cannot see \
this conversation, so write a complete, self-contained task: exact file paths \
(e.g. data/sales.csv), what to do, and any preferences the user stated. It returns the \
specialist's summary; files it produces are offered to the user as downloads. The same \
specialist handles follow-ups in this chat and remembers its earlier work.

Specialists:
{roster}"""


class SubAgentPool:
    """The specialists one chat delegates to: created on first use, kept for follow-ups."""

    def __init__(self, parent: "BaseAgent"):
        self.parent = parent
        self.agents: dict[str, BaseAgent] = {}

    def available(self) -> dict[str, type["BaseAgent"]]:
        from baseagent.agents import discover

        return {name: cls for name, cls in sorted(discover().items())
                if cls.delegatable and cls is not type(self.parent)}

    def get(self, name: str) -> "BaseAgent":
        options = self.available()
        if name not in options:
            raise ToolError(f"Unknown agent {name!r}. Choose one of: {', '.join(options) or '(none)'}.")
        # The specialist follows the chat's model choice but keeps its own step limits etc.
        inherited = {key: getattr(self.parent.settings, key) for key in INHERITED_SETTINGS}
        agent = self.agents.get(name)
        if agent is None:
            cls = options[name]
            parent = self.parent
            agent = cls(parent.llm, cls.make_settings(**inherited), log_dir=parent.log_dir,
                        session_id=f"{parent.session_id}.{name}", workspace=parent.workspace)
            self.agents[name] = agent
        else:
            agent.settings = agent.settings.model_copy(update=inherited)
        return agent

    async def close(self) -> None:
        for agent in self.agents.values():
            try:
                await agent.close()
            except Exception:  # noqa: BLE001 - closing the rest matters more
                pass
        self.agents.clear()


def progress_message(event: BaseModel) -> str | None:
    """A one-line status for the UI from a specialist's event, or None to skip it."""
    kind = getattr(event, "type", "")
    if kind == "tool_start":
        label = event.name
        try:
            args: dict[str, Any] = json.loads(event.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        note = str(args.get("note") or "").strip().splitlines()
        code = [line for line in str(args.get("code") or "").splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
        if note:
            label = note[0].lstrip("#").strip() or label
        elif code:
            label = code[0].strip()
        return f"step {event.step} · {label[:80]}"
    if kind == "compaction":
        return "compacting its context"
    if kind == "error":
        return f"error: {event.message}"
    return None


async def _forget(_: BaseModel) -> None:
    return None


async def delegate(agent: str, task: str, ctx: ToolContext) -> ToolResult:
    """Hand a task to a specialist agent.

    Args:
        agent: The specialist's name, from the list above.
        task: The complete, self-contained task for the specialist.
    """
    pool = getattr(ctx.agent, "subagents", None)
    if not isinstance(pool, SubAgentPool):
        raise ToolError("This agent cannot delegate.")
    if not task.strip():
        raise ToolError("task is empty.")
    specialist = pool.get(agent)

    async def forward(event: BaseModel) -> None:
        message = progress_message(event)
        if message:
            await ctx.progress(f"{specialist.title}: {message}")

    specialist.emit_callback = forward
    try:
        result = await specialist.run(task)
    finally:
        specialist.emit_callback = _forget
    files = specialist.output_files()

    status = "" if result.stop_reason == "final" else f", stopped: {result.stop_reason}"
    parts = [f"`{agent}` finished ({result.steps} steps{status}).", result.final_text]
    if files:
        names = ", ".join(f.title or "file" for f in files)
        parts.append(f"Files ready for the user to download below your reply: {names}")
    return ToolResult("\n\n".join(parts), attachments=files, is_error=result.stop_reason == "error")


def make_delegate_tool(options: dict[str, type["BaseAgent"]]) -> Tool:
    roster = "\n".join(f"- {name}: {cls.description}" for name, cls in options.items())
    return tool(delegate, name="delegate", timeout=DELEGATE_TIMEOUT,
                description=DELEGATE_DESCRIPTION.format(roster=roster or "- (none registered)"))
