"""The coordinator: the chat room's default agent.

It talks with the user, answers simple things itself, and delegates
specialized work (any agent with ``delegatable = True``) through the
``delegate`` tool. Adding a new specialist needs no change here.
"""

from __future__ import annotations

from baseagent.agent import BaseAgent
from baseagent.agents import register_agent
from baseagent.builtin_tools import calculator, current_time
from baseagent.delegation import SubAgentPool, make_delegate_tool
from baseagent.handoff import handoff_targets, make_handoff_tool

SYSTEM_PROMPT = """\
You are the coordinator in Russell's BaseAgent chat room. You talk with the user and \
route specialized work to specialist agents with the `delegate` tool.

- When the user wants to learn or play chess ("teach me chess", "let's play chess", "how does \
the knight move?"), call `handoff` with agent "chess_coach" right away and don't answer it \
yourself: the coach shows a board and answers in this same reply. Use `handoff` for every \
specialist listed in its description; those talk with the user directly.
- When a request matches a `delegate` specialist's description, delegate it. The specialist cannot \
see this conversation: write a complete task with exact file paths as they appear in the \
user's message (e.g. data/titanic.csv), what to do, and any preferences the user stated.
- Files the user attached are listed at the end of their message. If a request needs a \
file that isn't attached or mentioned, ask the user to drag and drop it into the chat.
- Answer general questions and small talk yourself. Use `calculator` for arithmetic and \
`current_time` for the date or time.
- After a specialist finishes, reply with a short Markdown summary of what it found: key \
numbers and issues, no long tables. If it produced files, tell the user they can download \
them below your reply."""


@register_agent
class CoordinatorAgent(BaseAgent):
    name = "coordinator"
    title = "Coordinator"
    description = (
        "Your main chat. Answers simple questions itself and hands specialized work to the "
        "right agent. Drag a CSV into the chat and ask for EDA to get a Jupyter notebook back, "
        "or say \"let's play chess\" to bring in the chess coach."
    )
    examples = ["What can you do?", "Let's play chess", "Teach me chess. I've never played.",
                "What is 12 * (3 + 4)^2 / 7?"]
    default_settings = {"max_steps": 10}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subagents = SubAgentPool(self)

    async def close(self) -> None:
        await self.subagents.close()

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def get_tools(self):
        tools = [calculator, current_time, make_delegate_tool(self.subagents.available())]
        targets = handoff_targets(exclude=type(self))
        if targets:
            tools.append(make_handoff_tool(targets))
        return tools
