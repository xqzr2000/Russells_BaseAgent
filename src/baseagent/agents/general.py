"""The general-purpose agent: BaseAgent plus a few safe demo tools.

This is the agent to test the base loop with in the chat room. It is also the
smallest example of a specialized agent: a few class attributes, a prompt,
and a tool list.
"""

from __future__ import annotations

from pathlib import Path

from baseagent.agent import BaseAgent
from baseagent.agents import register_agent
from baseagent.builtin_tools import calculator, current_time, list_notes, save_note

SKILLS = Path(__file__).resolve().parents[3] / "skills" / "general"


@register_agent
class GeneralAgent(BaseAgent):
    name = "general"
    title = "General assistant"
    description = "The base agent with a calculator, a clock, session notes, and one skill."
    skills_dir = SKILLS if SKILLS.is_dir() else None
    examples = [
        "What is 12 * (3 + 4)^2 / 7?",
        "What time is it in Tokyo right now?",
        "Teach me what a ReAct loop is.",
        "Remember that my EDA dataset is titanic.csv, then list my notes.",
    ]

    def system_prompt(self) -> str:
        return (
            "You are Russell's general-purpose assistant, built on BaseAgent.\n"
            "- Use `calculator` for any arithmetic instead of computing in your head.\n"
            "- Use `current_time` for questions about the date or time.\n"
            "- Use `save_note` / `list_notes` when the user asks you to remember things.\n"
            "- When you have the answer, reply in concise Markdown without calling a tool."
        )

    def get_tools(self):
        return [calculator, current_time, save_note, list_notes]
