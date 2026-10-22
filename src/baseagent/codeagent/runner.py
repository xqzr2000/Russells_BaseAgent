"""Runs one snippet of code in a throwaway, locked-down container.

This is the only place the code agent's snippets ever execute. Every call
starts a fresh container from the runner image (``docker/runner/Dockerfile``)
and deletes it afterwards:

* no network, read-only file system, non-root user, no API keys
* 512 MB of memory with no swap, 1 CPU, at most 64 processes
* ``run`` stops after 10 s; ``check`` (syntax check and lint) after 30 s
* a run that prints more than 1 MB is stopped (usually a printing endless loop)

The result says exactly what happened (``status``), so the answer the user sees
("stopped after 10 s: it never finished") comes from here, not from the model.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal

from agent_sandbox_runtime import Sandbox
from agent_sandbox_runtime.process import DockerError

RUNNER_IMAGE = os.environ.get("BASEAGENT_RUNNER_IMAGE", "baseagent-runner:1")
WORKDIR = "/work"
MAX_CODE_CHARS = 200_000

Status = Literal["ok", "error", "timeout", "out_of_memory", "too_much_output", "not_runnable"]
Mode = Literal["check", "run"]

LANGUAGE_ALIASES = {
    "python": "python", "py": "python", "python3": "python",
    "javascript": "javascript", "js": "javascript", "node": "javascript", "nodejs": "javascript",
    "typescript": "typescript", "ts": "typescript",
}
FILE_NAMES = {"python": "main.py", "javascript": "main.js", "typescript": "main.ts"}

# Each mode is a list of steps; a step that fails stops the rest.
COMMANDS: dict[str, dict[str, list[list[str]]]] = {
    "python": {
        # Compile without writing a .pyc, then pyflakes-style checks (undefined
        # names, unused imports, ...) and syntax errors only, not style.
        "check": [["python", "-I", "-c",
                   "import sys; compile(open(sys.argv[1]).read(), sys.argv[1], 'exec')", "main.py"],
                  ["ruff", "check", "--no-cache", "--quiet", "--output-format", "concise", "--select", "E9,F",
                   "main.py"]],
        "run": [["python", "-I", "-B", "main.py"]],
    },
    "javascript": {
        "check": [["node", "--check", "main.js"]],
        "run": [["node", "main.js"]],
    },
    "typescript": {
        "check": [["tsc", "--noEmit", "--pretty", "false", "--target", "es2022", "--module", "nodenext",
                   "--moduleResolution", "nodenext", "--skipLibCheck",
                   "--typeRoots", "/usr/local/lib/node_modules/@types", "--types", "node", "main.ts"]],
        "run": [["tsx", "main.ts"]],
    },
}

OOM_MARKERS = ("MemoryError", "JavaScript heap out of memory", "Cannot allocate memory")


@dataclass(frozen=True)
class RunLimits:
    run_timeout: float = 10.0  # seconds for `run`
    check_timeout: float = 30.0  # seconds for `check` (tsc is slow to start)
    memory: str = "512m"
    cpus: float = 1.0
    pids: int = 64
    output_chars: int = 4000  # per stream, kept from the start and the end
    max_output_bytes: int = 1_000_000  # stdout + stderr while running; more stops the run


@dataclass
class RunResult:
    language: str
    mode: str
    status: Status
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0  # seconds
    note: str = ""  # one plain sentence about what happened
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration"] = round(self.duration, 2)
        return data


def normalize_language(language: str) -> str | None:
    return LANGUAGE_ALIASES.get(language.strip().lower())


def trim(text: str, limit: int) -> str:
    """Keep the start and the end of long output."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return f"{text[:head]}\n... [{len(text) - limit} characters cut] ...\n{text[-tail:]}"


def classify(returncode: int, elapsed: float, timeout: float, output: str) -> tuple[Status, str]:
    """What happened to one step, from its exit code, run time and output."""
    if returncode == 0:
        return "ok", ""
    if returncode == 124 or (returncode in (137, -9) and elapsed >= timeout - 0.5):
        return "timeout", f"Stopped after {timeout:g} s: it did not finish in time (for example an endless loop)."
    if any(marker in output for marker in OOM_MARKERS) or returncode in (137, -9):
        return "out_of_memory", "Stopped: it used more than the memory limit."
    return "error", f"Exited with code {returncode}."


class CodeRunner:
    """Runs snippets in fresh runner containers. Thread-safe: each call has its own container."""

    def __init__(self, image: str = RUNNER_IMAGE, limits: RunLimits | None = None,
                 sandbox_factory: Callable[..., Any] = Sandbox):
        self.image = image
        self.limits = limits or RunLimits()
        self.sandbox_factory = sandbox_factory

    def run(self, language: str, code: str, mode: Mode = "run", stdin: str | None = None) -> RunResult:
        lang = normalize_language(language)
        if mode not in ("check", "run"):
            raise ValueError("mode must be 'check' or 'run'")
        if lang is None:
            return RunResult(language, mode, "not_runnable",
                             note=f"{language} can't be run here; only Python, JavaScript and TypeScript can.")
        if not code.strip():
            return RunResult(lang, mode, "error", note="There is no code to run.")
        if len(code) > MAX_CODE_CHARS:
            return RunResult(lang, mode, "error", note=f"The code is longer than {MAX_CODE_CHARS} characters.")

        limits = self.limits
        timeout = limits.run_timeout if mode == "run" else limits.check_timeout
        steps = COMMANDS[lang][mode]
        try:
            sandbox = self.sandbox_factory(
                self.image, cwd=WORKDIR, network="none", memory=limits.memory, cpus=limits.cpus,
                pids_limit=limits.pids, user="1000:1000", read_only=True,
                tmpfs={WORKDIR: "size=16m,uid=1000,gid=1000,mode=0700",
                       "/tmp": "size=64m,uid=1000,gid=1000,mode=0700"},
                labels={"org.baseagent.role": "code-runner"},
                deployment_timeout=timeout * len(steps) + 60, runtime_timeout=timeout,
            )
        except DockerError as exc:
            raise RunnerUnavailable(str(exc)) from exc

        try:
            sandbox.write_file(f"{WORKDIR}/{FILE_NAMES[lang]}", code)
            stdout, stderr, elapsed_total = [], [], 0.0
            status: Status = "ok"
            note, exit_code = "", 0
            for step in steps:
                started = time.monotonic()
                out = sandbox.execute(step, shell=False, timeout=timeout,
                                      stdin=stdin if mode == "run" else None,
                                      output_limit=limits.max_output_bytes)
                elapsed = time.monotonic() - started
                elapsed_total += elapsed
                stdout.append(out.get("stdout", ""))
                stderr.append(out.get("stderr", ""))
                exit_code = out.get("returncode")
                if out.get("output_limited"):
                    status, exit_code = "too_much_output", None
                    note = (f"Stopped: it printed more than {limits.max_output_bytes // 1_000_000} MB "
                            "of output (often an endless loop that prints).")
                    break
                status, note = classify(exit_code, elapsed, timeout, out.get("output", ""))
                if status != "ok":
                    break
            if status == "ok":
                note = "Ran without errors." if mode == "run" else "No problems found."
            elif status == "error" and mode == "check":
                note = "The check found problems (see the output)."
            return RunResult(
                lang, mode, status, exit_code=exit_code,
                stdout=trim("".join(stdout), limits.output_chars),
                stderr=trim("".join(stderr), limits.output_chars),
                duration=elapsed_total, note=note,
                meta={"timeout": timeout, "memory": limits.memory},
            )
        except DockerError as exc:  # the container died under the code (e.g. killed as a whole)
            return RunResult(lang, mode, "error", note=f"The sandbox stopped unexpectedly: {str(exc)[:300]}")
        finally:
            sandbox.stop()


class RunnerUnavailable(RuntimeError):
    """Docker or the runner image isn't available."""
