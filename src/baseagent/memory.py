"""Conversation memory and model-generated context compaction.

History is a flat list of chat messages (user / assistant / tool). For
compaction it is viewed as *blocks*: a user message, or an assistant message
together with the tool results that answer its calls. Blocks are never split,
so the prompt always stays valid (every tool call keeps its result).
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from typing import Any

MAX_OBSERVATION_CHARS = 10_000

COMPACTION_SYSTEM_PROMPT = """You maintain the working memory of an AI agent that uses tools while chatting with a user.

You receive the agent's earlier working memory (if any) and a transcript of older
conversation turns and tool steps. Those messages are about to be removed from the
agent's context, so your summary is the only record the agent keeps. Write concise,
factual working memory so the agent can continue without redoing work.

Preserve, when present:
- User goals and requests, constraints, and preferences they stated.
- Facts established: data, files, names, values, decisions, conclusions.
- Actions taken with tools and their concrete results (numbers, errors, outputs).
- Failed approaches and why they failed, so they are not repeated.
- Open questions and the next step for any unfinished request.

Rules: be specific (exact names, paths, numbers); do not copy raw tool output; do not
invent anything; carry forward still-relevant points from earlier working memory;
use terse bullet points under short headings; output only the working memory."""


def rough_tokens(messages: list[dict[str, Any]]) -> int:
    """Provider-independent estimate: about four characters per token."""
    return max(1, math.ceil(len(json.dumps(messages, ensure_ascii=False)) / 4))


def truncate(text: str, limit: int = MAX_OBSERVATION_CHARS) -> str:
    """Keep the head and tail of long output, with a notice in between."""
    if len(text) <= limit:
        return text
    keep = (limit - 120) // 2
    omitted = len(text) - 2 * keep
    return f"{text[:keep]}\n[... {omitted} characters omitted; request a narrower range ...]\n{text[-keep:]}"


class Memory:
    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []
        self.working_memory: str | None = None

    # -- building the prompt -------------------------------------------------

    def messages(self, system_prompt: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if self.working_memory:
            out.append({
                "role": "user",
                "content": (
                    "Earlier parts of this conversation were compacted to save context. "
                    "This is your working memory of them; treat it as accurate.\n\n"
                    f"<working_memory>\n{self.working_memory}\n</working_memory>"
                ),
            })
        out.extend(deepcopy(self.history))
        return out

    def add(self, message: dict[str, Any]) -> None:
        self.history.append(message)

    # -- keeping the history valid ---------------------------------------------

    def repair(self, reason: str = "The call was interrupted before it ran.") -> None:
        """Give every tool call a result, e.g. after a cancelled turn."""
        fixed: list[dict[str, Any]] = []
        for index, message in enumerate(self.history):
            fixed.append(message)
            if message.get("role") != "assistant" or not message.get("tool_calls"):
                continue
            following = []
            for later in self.history[index + 1:]:
                if later.get("role") != "tool":
                    break
                following.append(later.get("tool_call_id"))
            for call in message["tool_calls"]:
                if call["id"] not in following:
                    fixed.append({"role": "tool", "tool_call_id": call["id"],
                                  "content": f"Error: {reason}"})
        self.history = fixed

    # -- compaction ------------------------------------------------------------

    def blocks(self) -> list[list[dict[str, Any]]]:
        blocks: list[list[dict[str, Any]]] = []
        for message in self.history:
            if message.get("role") == "tool" and blocks:
                blocks[-1].append(message)
            else:
                blocks.append([message])
        return blocks

    def split_for_compaction(
        self, keep_recent_steps: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
        """Return (old messages to summarize, messages to keep verbatim), or None.

        Keeps the last ``keep_recent_steps`` assistant steps with their tool
        results, and always keeps the user's most recent message verbatim.
        """
        blocks = self.blocks()
        assistant_positions = [i for i, b in enumerate(blocks) if b[0].get("role") == "assistant"]
        if len(assistant_positions) <= keep_recent_steps:
            return None
        cut = assistant_positions[-keep_recent_steps]
        old_blocks, kept_blocks = blocks[:cut], blocks[cut:]

        user_positions = [i for i, b in enumerate(blocks) if b[0].get("role") == "user"]
        if user_positions and user_positions[-1] < cut:
            current_request = blocks[user_positions[-1]]
            old_blocks = [b for b in old_blocks if b is not current_request]
            kept_blocks = [current_request, *kept_blocks]
        if not old_blocks:
            return None
        flatten = lambda bs: [m for b in bs for m in b]  # noqa: E731
        return flatten(old_blocks), flatten(kept_blocks)

    def compaction_request(self, old_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        parts = []
        if self.working_memory:
            parts.append(f"<previous_working_memory>\n{self.working_memory}\n</previous_working_memory>")
        transcript = "\n\n".join(render_message(m) for m in old_messages)
        parts.append(f"<transcript_to_compact>\n{transcript}\n</transcript_to_compact>")
        parts.append("Write the updated working memory now. The most recent steps and the "
                     "user's latest message stay in context verbatim.")
        return [
            {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(parts)},
        ]

    def apply_compaction(self, summary: str, kept: list[dict[str, Any]]) -> None:
        self.working_memory = summary
        self.history = kept


def _clip(text: str, limit: int) -> str:
    return truncate(text, limit) if len(text) > limit else text


def render_message(message: dict[str, Any]) -> str:
    role = message.get("role")
    content = message.get("content") or ""
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    if role == "assistant":
        lines = ["[assistant]"]
        if content.strip():
            lines.append(_clip(content.strip(), 1500))
        for call in message.get("tool_calls") or []:
            fn = call.get("function", {})
            lines.append(f"tool_call {call.get('id')} {fn.get('name')}: {_clip(fn.get('arguments', ''), 1500)}")
        return "\n".join(lines)
    if role == "tool":
        return f"[tool result {message.get('tool_call_id')}]\n{_clip(content, 2500)}"
    return f"[{role}]\n{_clip(content, 2500)}"
