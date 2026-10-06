"""Copy-paste starting point for a new specialized agent.

Files starting with "_" are NOT auto-registered. To make a real agent:

    cp src/baseagent/agents/_template.py src/baseagent/agents/my_agent.py

then rename the class, set ``name``/``title``/``description``, and restart the
server (``make dev`` auto-reloads). It appears in the chat room's agent picker.
"""

from __future__ import annotations

from baseagent import BaseAgent, ToolContext, ToolError, ToolResult, register_agent, tool
from baseagent.events import Attachment


@tool
def example_tool(query: str, limit: int = 5) -> str:
    """One sentence the model reads to decide when to call this tool.

    Args:
        query: What the argument means, in the model's terms.
        limit: Optional arguments get defaults and are not required.
    """
    if limit < 1:
        raise ToolError("limit must be at least 1.")  # recoverable: the model sees it
    return f"Looked up {query!r} (limit {limit})."


@tool
async def example_rich_tool(title: str, ctx: ToolContext) -> ToolResult:
    """Tools can be async, use per-session state, and attach images/tables for the UI.

    Args:
        title: A caption.
    """
    ctx.state["calls"] = ctx.state.get("calls", 0) + 1
    table = "| a | b |\n|---|---|\n| 1 | 2 |"
    return ToolResult(
        content=f"Made a table titled {title!r}.",  # what the model sees
        attachments=[Attachment(kind="table", mime="text/markdown", title=title, data=table)],
    )


# @register_agent   <- uncomment in your copy
class TemplateAgent(BaseAgent):
    name = "template"  # unique id used by the API
    title = "Template agent"  # shown in the agent picker
    description = "Describe what this agent is for."
    examples = ["A starter prompt that shows off this agent."]
    default_settings = {"max_steps": 20}
    skills_dir = None  # or Path(...) to a folder of SKILL.md skills

    def __init__(self, *args, workdir: str = ".", **kwargs):
        super().__init__(*args, **kwargs)
        self.workdir = workdir  # store options only; no I/O here

    async def setup(self) -> None:
        # Open connections, start kernels, load data: anything async.
        self.resource = "ready"

    async def close(self) -> None:
        self.resource = None

    def system_prompt(self) -> str:
        return "You are ... Use `example_tool` when ... Reply in Markdown when done."

    def get_tools(self):
        return [example_tool, example_rich_tool]


_ = register_agent  # keeps the import used while the decorator is commented out
