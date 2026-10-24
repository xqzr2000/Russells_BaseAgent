"""Turns a ``CodeFix`` into the Markdown answer shown in the chat.

    **What was wrong**
    1. Line 1: ... Fix: ...

    **Fixed code** (`main.py`)
    ```python
    ...  # FIX 1: ...
    ```

    **Ran:** printed `45`
    Your original code: stopped after 10 s: it never finished (an endless loop).

    *Fixed by OpenCode with the free model deepseek-v4.1-flash-free.*

The verification line ("Ran", "Checked only", "Reviewed only") and the note
about the original code come from the bridge's run log, matched to the code
by its hash, not from what the model says it did.
"""

from __future__ import annotations

from baseagent.codeagent.bridge import code_sha256
from baseagent.codeagent.opencode_client import CodeFix
from baseagent.codeagent.runner import RunResult

OUTPUT_LINES = 12
OUTPUT_CHARS = 800

STOPPED = {
    "timeout": "stopped after {timeout:g} s: it never finished (an endless loop, or far too slow)",
    "too_much_output": "stopped: it kept printing (more than 1 MB), usually an endless loop",
    "out_of_memory": "stopped: it used more than the 512 MB memory limit",
}


def _output(text: str) -> str:
    """Program output as inline code (one short line) or a small code block."""
    text = text.rstrip()
    lines = text.splitlines()
    more = len(lines) > OUTPUT_LINES or len(text) > OUTPUT_CHARS
    shown = "\n".join(lines[:OUTPUT_LINES])[:OUTPUT_CHARS]
    if len(lines) == 1 and len(shown) <= 80 and "`" not in shown:
        return f"`{shown}`"
    return "\n```text\n" + shown + ("\n..." if more else "") + "\n```"


def _last_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1].strip()[:200] if lines else ""


def _stopped(run: RunResult) -> str | None:
    template = STOPPED.get(run.status)
    return template.format(timeout=run.meta.get("timeout", 10)) if template else None


def verification(fix: CodeFix) -> str:
    """The one-line account of how far the fixed code was verified."""
    final = code_sha256(fix.fixed)
    on_final = [r for r in fix.runs if r.meta.get("code_sha256") == final]
    run = next((r for r in reversed(on_final) if r.mode == "run"), None)
    check = next((r for r in reversed(on_final) if r.mode == "check"), None)
    reason = fix.not_run_reason.strip().rstrip(".")
    if run:
        stopped = _stopped(run)
        if stopped:
            return f"**Ran:** the fixed code was {stopped}."
        if run.status == "ok":
            return f"**Ran:** printed {_output(run.stdout)}" if run.stdout.strip() else \
                "**Ran:** it ran without errors and printed nothing."
        return f"**Ran:** it still fails: `{_last_line(run.stderr or run.stdout) or f'exit code {run.exit_code}'}`"
    if check:
        if check.status == "ok":
            line = "**Checked only:** no syntax or type errors"
            return f"{line}. Not run: {reason}." if reason else f"{line}; it wasn't run."
        return f"**Checked:** problems remain: `{_last_line(check.stderr or check.stdout)}`"
    if fix.mode == "review":
        return "**Reviewed only:** not run, as you asked."
    if not fix.runnable:
        return f"**Reviewed only:** {fix.fence or 'this language'} can't be run here (only Python, " \
               "JavaScript and TypeScript can)."
    if fix.runs:
        return "**Reviewed only:** an earlier version was run, but not the final one."
    return f"**Reviewed only:** not run. {reason}." if reason else "**Reviewed only:** not run."


def original_behaviour(fix: CodeFix) -> str | None:
    """What happened when the user's own code was run, if it was and it went wrong."""
    original = code_sha256(fix.original)
    if original == code_sha256(fix.fixed):
        return None
    for run in fix.runs:
        if run.meta.get("code_sha256") != original or run.mode != "run":
            continue
        stopped = _stopped(run)
        if stopped:
            return f"Your original code: {stopped}."
        if run.status == "error":
            last = _last_line(run.stderr or run.stdout)
            return f"Your original code failed with: `{last}`" if last else None
    return None


def build_answer(fix: CodeFix) -> str:
    changed = code_sha256(fix.fixed) != code_sha256(fix.original)
    if not fix.answered and not changed:
        if fix.stopped == "time limit":
            return ("*OpenCode ran out of time (3-minute limit).* Your code wasn't changed. Try again, or send "
                    "a smaller piece of the code.")
        reason = (fix.stopped or "no answer").rstrip(".")
        return (f"*OpenCode couldn't finish ({reason}).* Your code wasn't changed. Free models are "
                "sometimes busy, so try again in a minute.")
    parts: list[str] = []
    if fix.stopped == "time limit":
        parts.append("*OpenCode hit the 3-minute limit, so this is what it had so far.*")
    elif fix.stopped and not fix.answered:
        parts.append(f"*OpenCode stopped early: {fix.stopped}*")

    if fix.issues:
        lines = ["**What was wrong**"]
        for n, issue in enumerate(fix.issues, 1):
            line = issue.get("line")
            where = f"Line {line}: " if isinstance(line, int) and line > 0 else ""
            problem = str(issue.get("problem", "")).strip()
            remedy = str(issue.get("fix", "")).strip()
            lines.append(f"{n}. {where}{problem}" + (f" Fix: {remedy}" if remedy else ""))
        parts.append("\n".join(lines))
    elif not changed:
        parts.append("**No bugs found.** " + (fix.summary or "The code looks correct as written."))

    if changed or fix.issues:
        title = "**Fixed code**" if changed else "**Your code** (unchanged)"
        parts.append(f"{title} (`{fix.filename}`)\n```{fix.fence}\n{fix.fixed.rstrip()}\n```")

    if fix.summary and fix.issues:
        parts.append(fix.summary)
    parts.append(verification(fix))
    before = original_behaviour(fix)
    if before:
        parts.append(before)
    if fix.model:
        privacy = "" if fix.zero_retention else " (this free model may use what it's sent for training)"
        parts.append(f"*{'Fixed' if changed else 'Checked'} by OpenCode with the free model {fix.model}{privacy}.*")
    return "\n\n".join(parts)
