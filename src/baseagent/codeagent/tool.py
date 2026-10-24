"""``fix_code``: Jack's tool for the code agent.

The tool reads the user's code itself (a ``` code block in their latest
message, or an attached file in the chat's workspace) so the coordinator model
never retypes it; a model copying code can quietly change it. It runs the
code agent (OpenCode in a sandbox, ``opencode_client``) with progress lines
under the tool call, and returns the Markdown answer with ``end_turn=True``:
the chat shows it exactly as built, with no further model call.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Literal

from baseagent.codeagent.answer import build_answer
from baseagent.codeagent.opencode_client import (
    MAX_CODE_CHARS, STARTUP_LIMIT, TIME_LIMIT, CodeAgentError, CodeTask, OpenCodeFixer,
)
from baseagent.tools import ToolContext, ToolError, ToolResult, tool

CODE_BLOCK = re.compile(r"```[ \t]*([\w+#.\-]*)[^\n]*\n(.*?)```", re.DOTALL)
ATTACHMENTS = re.compile(r"\n*Attached files?: .*$", re.DOTALL)
TOOL_TIMEOUT = TIME_LIMIT + STARTUP_LIMIT + 120  # the whole request, with room to clean up

_fixer: OpenCodeFixer | None = None


def get_fixer() -> OpenCodeFixer:
    """One fixer per server process (it shares the process's runner bridge)."""
    global _fixer
    if _fixer is None:
        _fixer = OpenCodeFixer()
    return _fixer


def latest_user_message(history: list[dict]) -> str:
    """The user's latest message text (what they typed, plus the attachment line)."""
    for message in reversed(history):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            if not message["content"].startswith("Your last response neither called a tool"):
                return message["content"]
    return ""


def code_blocks(text: str) -> list[tuple[str, str]]:
    """(language tag, code) for each fenced code block in ``text``."""
    return [(lang.strip().lower(), body.rstrip("\n") + "\n") for lang, body in CODE_BLOCK.findall(text)]


def question_from(text: str) -> str:
    """The user's request without its code blocks and attachment line."""
    question = ATTACHMENTS.sub("", CODE_BLOCK.sub(" ", text))
    return re.sub(r"\s+", " ", question).strip()[:500]


DESCRIPTION = """\
Check, fix and explain the user's code in a sandbox (OpenCode, free model). For checking, \
fixing, debugging or testing code; answer concept questions yourself. It reads the code from \
the user's ``` block or attached file itself, so don't copy it. Its answer is shown as-is \
and ends your turn."""


@tool(name="fix_code", description=DESCRIPTION, exclusive=True, timeout=TOOL_TIMEOUT)
async def fix_code(mode: Literal["review", "auto", "test"] = "auto", question: str = "", file: str = "",
                   block: int = 0, language: str = "", code: str = "", ctx: ToolContext = None) -> ToolResult:
    """
    Args:
        mode: review: they said not to run it. test: they want to see it work ("does it work?", "run it"). auto: anything else.
        question: The user's request in their words.
        file: An attached file's path, e.g. data/main.py. Empty for a code block.
        block: Which ``` block in their latest message: 1 = first, 0 = last.
        language: Only if the code block has no language tag.
        code: Only for code pasted without ``` fences: the code, copied exactly.
    """
    message = latest_user_message(ctx.agent.memory.history)
    filename, fence = None, ""
    if file.strip():
        workspace = ctx.agent.workspace
        if workspace is None:
            raise ToolError("This chat has no files.")
        try:
            path = workspace.resolve(file.strip())
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        if not path.is_file():
            raise ToolError(f"No such file: {file}. Ask the user to attach it.")
        if path.stat().st_size > MAX_CODE_CHARS * 4:
            raise ToolError(f"{file} is too large for the code agent.")
        try:
            source = path.read_text()
        except UnicodeDecodeError as exc:
            raise ToolError(f"{file} isn't a text file.") from exc
        filename = PurePosixPath(file.strip()).name
    elif code.strip():
        source = code
    else:
        blocks = code_blocks(message)
        if not blocks:
            raise ToolError("The user's latest message has no ``` code block and no file was given. If they "
                            "pasted code without fences, pass it in `code` exactly; otherwise ask them to paste "
                            "or attach it.")
        if not 0 <= block <= len(blocks):
            raise ToolError(f"There are {len(blocks)} code blocks; use block 1 to {len(blocks)}, or 0 for the last.")
        fence, source = blocks[block - 1 if block > 0 else -1]
    task = CodeTask(code=source, request=question.strip() or question_from(message) or
                    "Check this code and fix what is wrong.",
                    language=language.strip() or fence or None, filename=filename, mode=mode)
    try:
        fix = await get_fixer().fix(task, ctx.progress)
    except CodeAgentError as exc:
        raise ToolError(f"The code agent couldn't do it: {exc}") from exc
    return ToolResult(build_answer(fix), end_turn=True)
