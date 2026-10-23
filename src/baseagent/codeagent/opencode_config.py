"""What the code agent's OpenCode is allowed to do, and which free model it uses.

``build_config()`` returns the JSON passed to ``opencode serve`` as
``OPENCODE_CONFIG_CONTENT``. It defines one agent, ``fixer``:

* allowed: read, edit (edit/write/apply_patch), glob, grep, ``run_code``
  (our tool, docker/opencode/config/tools/run_code.js) and OpenCode's own
  ``StructuredOutput`` tool for the final answer
* denied: everything else (shell, web fetch and search, sub-agents, skills,
  questions, todo lists, files outside the workspace, repeated calls)
* off: sharing, auto-format (it would reformat the user's code), LSP, update
  checks, snapshots; only the ``opencode`` provider is loaded

OpenCode (v1.19.2) hides any tool whose last matching rule is ``"*": "deny"``,
so ``StructuredOutput`` must be allowed explicitly. Rules are matched in order
and the last match wins, so ``"*"`` comes first.

``pick_model()`` chooses from the models OpenCode lists with no API key, which
are only its free models. By default it uses free models whose providers keep
no data; ``OPENCODE_ALLOW_TRAINING_MODELS=1`` also allows free models that may
log prompts or use them for training.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

OPENCODE_VERSION = "1.19.2"

# The environment flags baked into docker/opencode/Dockerfile (kept in sync by
# tests/test_opencode_config.py): no project config, no updates, no LSP
# downloads, no login plugins, no .claude or external skills, no web UI.
IMAGE_ENV = {
    "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
    "OPENCODE_DISABLE_AUTOUPDATE": "1",
    "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
    "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1",
    "OPENCODE_DISABLE_CLAUDE_CODE": "1",
    "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
    "OPENCODE_DISABLE_EMBEDDED_WEB_UI": "1",
    "OPENCODE_PURE": "1",
}
PROVIDER = "opencode"
FIXER_AGENT = "fixer"
FIXER_STEPS = 12  # agentic steps before the fixer must answer

# Free models with zero retention and no training (OpenCode Zen docs,
# checked 2026-10-22), in order of preference.
ZERO_RETENTION_MODELS = ("deepseek-v4.1-flash-free", "longcat-2.5-preview-free")
# Free models that may log prompts or train on them: used only with
# OPENCODE_ALLOW_TRAINING_MODELS=1, preferred in this order, then any other.
TRAINING_MODEL_PREFERENCE = ("big-pickle", "mimo-v2.6-flash-free", "nemotron-3-ultra-free")
# Listed as free but not general-purpose coding chat models.
NOT_FOR_CODE = ("jev-",)

PERMISSION: dict[str, str] = {
    "*": "deny",
    "read": "allow",
    "edit": "allow",  # covers edit, write and apply_patch
    "glob": "allow",
    "grep": "allow",
    "list": "allow",
    "run_code": "allow",
    "StructuredOutput": "allow",  # the final answer (format: json_schema)
    "invalid": "allow",  # OpenCode's reply to a malformed tool call
    "external_directory": "deny",
    "doom_loop": "deny",
}

FIXER_PROMPT = """\
You are a code fixer for a learner. The user's code is in the workspace (the current \
directory). Your job: find what is wrong, fix it with the smallest possible change, and \
explain each fix.

Rules
- Keep the user's style, names and structure. Change only what is needed to make the code \
correct. Do not reformat, rename or "improve" anything else.
- Edit the file in place. Do not create other files.
- On every line you change, add a short comment numbered in order, in the language's comment \
syntax: `# FIX 1: added ':'` in Python, `// FIX 1: ...` in JavaScript and TypeScript.
- Use run_code to verify: first mode "check", then mode "run" once it should work. Runs are \
limited, so don't run the same code twice. If the program reads input, pass stdin.
- If the code never finishes (an endless loop) or uses too much memory, fix the cause and say \
why it would not stop.
- If the language can't be run (anything but Python, JavaScript, TypeScript), review and fix \
it by reading only, and say it was not run.
- If the code has no bugs, change nothing and say so.

Finish with your structured answer: the issues (line numbers in the ORIGINAL code, in the \
same order as your FIX numbers), and a one-sentence summary."""

# The answer the fixer must return (OpenCode `format: json_schema`).
ANSWER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "description": "Each problem found, in the same order as the FIX n comments.",
            "items": {
                "type": "object",
                "properties": {
                    "line": {"type": "integer", "description": "Line number in the original code."},
                    "problem": {"type": "string", "description": "What was wrong, in one sentence."},
                    "fix": {"type": "string", "description": "What was changed, in one sentence."},
                },
                "required": ["line", "problem", "fix"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string", "description": "One sentence about the code and the result."},
    },
    "required": ["issues", "summary"],
    "additionalProperties": False,
}


def build_config(model: str, *, steps: int = FIXER_STEPS) -> dict:
    """The OPENCODE_CONFIG_CONTENT for one request, using free model ``model``."""
    full = f"{PROVIDER}/{model}"
    return {
        "$schema": "https://opencode.ai/config.json",
        "model": full,
        "small_model": full,
        "enabled_providers": [PROVIDER],
        "default_agent": FIXER_AGENT,
        "share": "disabled",
        "autoupdate": False,
        "formatter": False,
        "lsp": False,
        "snapshot": False,
        "permission": dict(PERMISSION),
        "agent": {
            FIXER_AGENT: {
                "mode": "primary",
                "description": "Fixes a code snippet for a learner and explains each fix.",
                "prompt": FIXER_PROMPT,
                "steps": steps,
                "temperature": 0.1,
                "permission": dict(PERMISSION),
            },
        },
    }


class NoFreeModel(RuntimeError):
    """No acceptable free model is available."""


@dataclass(frozen=True)
class ModelChoice:
    model: str  # without the "opencode/" prefix
    zero_retention: bool


def allow_training_models() -> bool:
    return os.environ.get("OPENCODE_ALLOW_TRAINING_MODELS", "").strip().lower() in {"1", "true", "yes", "on"}


def parse_model_list(output: str) -> list[str]:
    """Model ids from ``opencode models opencode`` (lines like ``opencode/big-pickle``)."""
    prefix = f"{PROVIDER}/"
    return [line.strip()[len(prefix):] for line in output.splitlines()
            if line.strip().startswith(prefix) and " " not in line.strip()]


def pick_model(available: list[str], *, preferred: str | None = None,
               allow_training: bool | None = None) -> ModelChoice:
    """Choose a free model from ``available`` (what OpenCode lists with no key).

    ``preferred`` (e.g. OPENCODE_FREE_MODEL) wins if it is available and
    allowed. Without ``allow_training`` only zero-retention models qualify.
    """
    allow = allow_training_models() if allow_training is None else allow_training
    listed = [m for m in dict.fromkeys(available) if m and not m.startswith(NOT_FOR_CODE)]

    def allowed(model: str) -> bool:
        return model in listed and (model in ZERO_RETENTION_MODELS or allow)

    if preferred:
        preferred = preferred.removeprefix(f"{PROVIDER}/")
        if allowed(preferred):
            return ModelChoice(preferred, preferred in ZERO_RETENTION_MODELS)
    for model in ZERO_RETENTION_MODELS:
        if model in listed:
            return ModelChoice(model, True)
    if allow:
        ordered = [m for m in TRAINING_MODEL_PREFERENCE if m in listed]
        ordered += [m for m in listed if m not in ordered]
        if ordered:
            return ModelChoice(ordered[0], False)
        raise NoFreeModel("OpenCode lists no free models right now.")
    raise NoFreeModel(
        "No zero-retention free model is available right now "
        f"({', '.join(ZERO_RETENTION_MODELS)}). Set OPENCODE_ALLOW_TRAINING_MODELS=1 to allow "
        "free models that may log or train on your code."
    )
