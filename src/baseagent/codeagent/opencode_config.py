"""What the code agent's OpenCode is allowed to do, and which free model it uses.

``build_config()`` returns the JSON passed to ``opencode serve`` as
``OPENCODE_CONFIG_CONTENT``. It defines one agent, ``fixer``, for one of three
verification modes (Jack picks one from the user's wording):

* ``review``: read and fix only; ``run_code`` is denied
* ``auto`` (default): always check; run only when behaviour matters and the
  code can run on its own, otherwise say why not
* ``test``: check and run; if the code only defines functions, add one marked
  call (``# TEST: ...``) to show it works

The fixer may use:

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
from typing import Literal

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

Mode = Literal["review", "auto", "test"]
MODES: tuple[str, ...] = ("review", "auto", "test")
RUN_LIMITS = {"review": 0, "auto": 4, "test": 6}  # run_code calls per request

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
- Focus on fixing and explaining. Don't run the user's original code: they have already seen \
what it does. Use run_code only on your fixed version.
- If the code never finishes (an endless loop) or uses too much memory, fix the cause and say \
why it would not stop.
- If the code has no bugs, change nothing and say so.

{verification}

Finish with your structured answer: the issues (line numbers in the ORIGINAL code, in the \
same order as your FIX numbers), a one-sentence summary, and not_run_reason (why the fixed \
code was not run, or "" if it was run)."""

VERIFICATION: dict[str, str] = {
    "review": """\
Verification: REVIEW ONLY. The user asked you not to run the code, and you can't: read it, \
fix it and explain. Set not_run_reason to "Reviewed only, as asked.\"""",
    "auto": """\
Verification: AUTO.
- Python, JavaScript and TypeScript: always call run_code with mode "check" on the fixed file.
- Then call run_code with mode "run" when the result depends on what the code does (loops, \
printed output, calculations, logic bugs) AND it can run on its own here.
- Don't run it when it needs things this sandbox doesn't have (input files, network, a GUI, \
packages that aren't installed, a server that runs forever) or when it only defines functions \
that nothing calls; say which in not_run_reason.
- Runs are limited (at most {runs}), so don't run the same code twice. If the program reads \
input, pass stdin.
- Other languages can't be run: review only, and say so in not_run_reason.""",
    "test": """\
Verification: TEST. The user wants to see it work.
- Call run_code with mode "check", then mode "run" on the fixed file.
- If the code only defines functions or classes and nothing calls them, add ONE short call at \
the end that shows the fix works, marked with a comment: `print(add_up(10))  # TEST: added to \
show it works` (`// TEST: ...` in JavaScript and TypeScript). Don't number it as a FIX and \
don't list it as an issue.
- If it can't run on its own here (input files, network, GUI, missing packages), say why in \
not_run_reason.
- Runs are limited (at most {runs}), so don't run the same code twice. If the program reads \
input, pass stdin.
- Other languages can't be run: review only, and say so in not_run_reason.""",
}

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
        "not_run_reason": {"type": "string",
                           "description": "Why the fixed code was not run, or \"\" if it was run."},
    },
    "required": ["issues", "summary", "not_run_reason"],
    "additionalProperties": False,
}


def fixer_prompt(mode: Mode = "auto") -> str:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    verification = VERIFICATION[mode].replace("{runs}", str(RUN_LIMITS[mode]))
    return FIXER_PROMPT.replace("{verification}", verification)


def permission_for(mode: Mode = "auto") -> dict[str, str]:
    rules = dict(PERMISSION)
    if mode == "review":
        rules["run_code"] = "deny"  # enforced, not just asked for
    return rules


def build_config(model: str, mode: Mode = "auto", *, steps: int = FIXER_STEPS) -> dict:
    """The OPENCODE_CONFIG_CONTENT for one request, using free model ``model``."""
    full = f"{PROVIDER}/{model}"
    rules = permission_for(mode)
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
        "permission": rules,
        "agent": {
            FIXER_AGENT: {
                "mode": "primary",
                "description": "Fixes a code snippet for a learner and explains each fix.",
                "prompt": fixer_prompt(mode),
                "steps": steps,
                "temperature": 0.1,
                "permission": dict(rules),
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
