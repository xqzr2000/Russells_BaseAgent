"""The coordinator: the chat room's default agent, shown as "Jack (the coordinator)".

It talks with the user, answers simple things itself, and delegates
specialized work (any agent with ``delegatable = True``) through the
``delegate`` tool. Adding a new specialist needs no change here. Code to
check or fix goes to the code agent through ``fix_code`` (baseagent.codeagent),
whose answer is shown to the user exactly as built.

The Jack persona is UI-only, apart from one fixed reply to "who are you?".
"""

from __future__ import annotations

import re

from baseagent.agent import BaseAgent, TurnResult
from baseagent.agents import register_agent
from baseagent.builtin_tools import calculator, current_time
from baseagent.codeagent.tool import fix_code
from baseagent.delegation import SubAgentPool, make_delegate_tool
from baseagent.handoff import handoff_targets, make_handoff_tool

IDENTITY_REPLY = "Woof, Jack here, I'm hunting jobs for Russell! You can ask me anything!"

# Whole-message "who are you?" phrasings answered without calling the model (0 tokens,
# word-for-word). Anything else, including unusual phrasings, goes to the model, whose
# prompt has the same rule.
_IDENTITY = re.compile(
    r"^(?:(?:hi|hey|hello|yo|ok|okay|so)\s+)*(?:jack\s+)?"
    r"(?:who\s+(?:are|r)\s+(?:you|u)|who\s+(?:is|'s|s)\s+this|what\s+are\s+you"
    r"|what(?:\s+is|'s|s)?\s+your\s+name|your\s+name|introduce\s+yourself|tell\s+me\s+about\s+yourself)"
    r"(?:\s+jack)?$")


def is_identity_question(message: str) -> bool:
    text = message.strip().lower().replace("\u2019", "'")
    if len(text) > 60:
        return False
    text = re.sub(r"[^\w\s']+", " ", text)
    return bool(_IDENTITY.match(re.sub(r"\s+", " ", text).strip()))

SYSTEM_PROMPT = f"""\
You are the coordinator in Russell's BaseAgent chat room: you talk with the user, answer \
simple things yourself and route specialized work.

- "Who are you?" and similar: reply with exactly this and nothing else: {IDENTITY_REPLY}
- Code to check, fix, debug or test: call `fix_code`. Questions about concepts or what code \
means: answer yourself.
- Learning or playing chess: call `handoff` to chess_coach right away, without answering yourself.
- Work matching a `delegate` specialist (e.g. EDA on a CSV): delegate it, then reply with a \
short Markdown summary (key numbers and issues, no long tables) and mention any downloads.
- Attached files are listed at the end of the user's message. If a request needs a file that \
isn't there, ask them to drag and drop it into the chat.
- Use `calculator` for arithmetic and `current_time` for the date or time."""


@register_agent
class CoordinatorAgent(BaseAgent):
    name = "coordinator"
    title = "Jack (the coordinator)"
    spoken_name = "Jack"
    description = (
        "Your main chat. Answers simple questions itself and hands specialized work to the "
        "right agent. Drag a CSV into the chat and ask for EDA to get a Jupyter notebook back, "
        "say \"let's play chess\" to bring in the chess coach, or paste code to have it checked and fixed."
    )
    examples = ["What can you do?", "Let's play chess", "Teach me chess. I've never played.",
                "What is 12 * (3 + 4)^2 / 7?"]
    default_settings = {"max_steps": 10}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subagents = SubAgentPool(self)

    async def close(self) -> None:
        await self.subagents.close()

    async def run(self, user_message: str, via: str | None = None) -> TurnResult:
        if via is None and is_identity_question(user_message):
            return await self.reply_without_model(user_message, IDENTITY_REPLY)
        return await super().run(user_message, via)

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def get_tools(self):
        tools = [calculator, current_time, fix_code, make_delegate_tool(self.subagents.available())]
        targets = handoff_targets(exclude=type(self))
        if targets:
            tools.append(make_handoff_tool(targets))
        return tools
