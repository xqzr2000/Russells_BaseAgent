"""A persistent IPython kernel that runs code the way notebook cells do.

Variables live between ``execute`` calls. Each call returns the cell's outputs
as nbformat output nodes (ready to drop into an .ipynb) plus a plain-text view
for the model. The kernel runs in this project's .venv, in the chat's
workspace folder, with secret environment variables removed.

Failure handling, so one bad cell never wedges the agent:

* error in the code -> outputs include the traceback, status "error"
* runs too long     -> interrupt (KeyboardInterrupt), status "timeout";
                       earlier variables survive
* ignores interrupt or the process dies (e.g. out of memory)
                    -> restart, status "restarted"; variables are lost
* the turn is cancelled (user pressed Stop) -> interrupt, then re-raise
"""

from __future__ import annotations

import asyncio
import os
import queue
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nbformat
from jupyter_client.manager import AsyncKernelManager

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
INTERRUPT_GRACE = 5.0  # seconds to wait for a cell to stop after an interrupt
TRACEBACK_LINES = 12  # traceback lines the model sees (the notebook keeps them all)


def kernel_env() -> dict[str, str]:
    """The server's environment without secrets, so code in the kernel can't print them."""
    return {k: v for k, v in os.environ.items() if not SECRET_ENV.search(k)}


def _note(text: str) -> Any:
    return nbformat.v4.new_output("stream", name="stderr", text=text.rstrip() + "\n")


@dataclass
class CellResult:
    execution_count: int | None
    outputs: list[Any]  # nbformat output nodes
    status: str = "ok"  # "ok" | "error" | "timeout" | "restarted"
    error: str | None = None  # one line, e.g. "KeyError: 'Age'"

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def figures(self) -> int:
        return sum(1 for o in self.outputs
                   if any(k.startswith("image/") for k in o.get("data", {})))

    def text(self, traceback_lines: int = TRACEBACK_LINES) -> str:
        """What the model reads: stdout/stderr, results as text, a short traceback."""
        parts: list[str] = []
        for out in self.outputs:
            kind = out.get("output_type")
            if kind == "stream":
                parts.append(out.get("text", ""))
            elif kind in ("execute_result", "display_data"):
                data = out.get("data", {})
                if any(k.startswith("image/") for k in data):
                    parts.append("[figure saved to the notebook; you cannot view it]\n")
                elif "text/plain" in data:
                    parts.append(str(data["text/plain"]).rstrip("\n") + "\n")
            elif kind == "error":
                lines = ANSI.sub("", "\n".join(out.get("traceback", []))).splitlines()
                parts.append("\n".join(lines[-traceback_lines:]) + "\n")
        return "".join(parts).strip()


@dataclass
class _Run:
    msg_id: str
    outputs: list[Any] = field(default_factory=list)
    count: int | None = None
    error: str | None = None
    idle: bool = False
    dead: bool = False


class KernelSandbox:
    """One kernel process. Cells run one at a time, in the order they're sent."""

    def __init__(self, kernel_name: str = "python3", startup_timeout: float = 60.0):
        self.kernel_name = kernel_name
        self.startup_timeout = startup_timeout
        self.km: AsyncKernelManager | None = None
        self.kc = None
        self._ipc_dir: str | None = None
        self._lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self.kc is not None

    async def start(self, cwd: str | Path) -> None:
        options: dict[str, Any] = {}
        if os.name == "posix":  # Unix sockets: no localhost TCP ports for anyone to reach
            self._ipc_dir = tempfile.mkdtemp(prefix="baseagent-kernel-")
            options = {"transport": "ipc", "ip": os.path.join(self._ipc_dir, "kernel")}
        self.km = AsyncKernelManager(kernel_name=self.kernel_name, **options)
        await self.km.start_kernel(cwd=str(cwd), env=kernel_env())
        self.kc = self.km.client()
        self.kc.start_channels()
        try:
            await self.kc.wait_for_ready(timeout=self.startup_timeout)
        except Exception:
            await self.shutdown()
            raise

    async def restart(self) -> None:
        assert self.km is not None and self.kc is not None, "kernel not started"
        await self.km.restart_kernel(now=True)
        await self.kc.wait_for_ready(timeout=self.startup_timeout)

    async def shutdown(self) -> None:
        if self.kc is not None:
            self.kc.stop_channels()
            self.kc = None
        if self.km is not None:
            try:
                await self.km.shutdown_kernel(now=True)
            except Exception:  # noqa: BLE001 - already gone is fine
                pass
            self.km = None
        if self._ipc_dir:
            shutil.rmtree(self._ipc_dir, ignore_errors=True)
            self._ipc_dir = None

    async def execute(self, code: str, timeout: float = 120.0) -> CellResult:
        if not self.running:
            raise RuntimeError("The kernel is not running.")
        async with self._lock:
            # stop_on_error=False: a failed or interrupted cell must not make the
            # kernel silently skip the next one (which a Stop can race with).
            run = _Run(self.kc.execute(code, store_history=True, allow_stdin=False,
                                       stop_on_error=False))
            try:
                await self._collect(run, timeout)
                if run.idle:
                    if await self._drain_reply(run.msg_id) == "aborted" and not run.outputs:
                        note = "The kernel skipped this cell. Run it again."
                        run.outputs.append(_note(note))
                        return CellResult(run.count, run.outputs, "error", note)
                    return CellResult(run.count, run.outputs, "error" if run.error else "ok", run.error)
                if not run.dead:
                    await self.km.interrupt_kernel()
                    await self._collect(run, INTERRUPT_GRACE)
                    if run.idle:
                        await self._drain_reply(run.msg_id)
                        note = (f"The cell ran longer than {timeout:g} s and was interrupted. "
                                "Variables from earlier cells are still defined.")
                        run.outputs.append(_note(note))
                        return CellResult(run.count, run.outputs, "timeout", note)
                why = ("the kernel process died (often from running out of memory)" if run.dead else
                       f"the cell ran longer than {timeout:g} s and ignored the interrupt")
                note = f"The kernel was restarted because {why}. All variables were lost."
                await self.restart()
                run.outputs.append(_note(note))
                return CellResult(run.count, run.outputs, "restarted", note)
            except asyncio.CancelledError:
                if not run.idle and self.km is not None:
                    try:  # free the kernel for the next cell
                        await self.km.interrupt_kernel()
                    except Exception:  # noqa: BLE001
                        pass
                raise

    async def _collect(self, run: _Run, seconds: float) -> None:
        """Read the cell's messages until it goes idle, the time is up, or the kernel dies."""
        deadline = time.monotonic() + seconds
        while not run.idle:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                msg = await self.kc.get_iopub_msg(timeout=min(1.0, remaining))
            except queue.Empty:
                if not await self.km.is_alive():
                    run.dead = True
                    return
                continue
            if msg["parent_header"].get("msg_id") != run.msg_id:
                continue  # left over from an earlier, interrupted cell
            self._handle(run, msg)

    @staticmethod
    def _handle(run: _Run, msg: dict[str, Any]) -> None:
        kind, content = msg["msg_type"], msg["content"]
        if kind == "status":
            run.idle = content.get("execution_state") == "idle"
        elif kind == "execute_input":
            run.count = content.get("execution_count")
        elif kind == "clear_output":
            run.outputs.clear()
        elif kind in ("stream", "display_data", "execute_result", "error"):
            out = nbformat.v4.output_from_msg(msg)
            last = run.outputs[-1] if run.outputs else None
            if (kind == "stream" and last is not None and last.get("output_type") == "stream"
                    and last.get("name") == out["name"]):
                last["text"] += out["text"]  # merge consecutive prints, as Jupyter does
            else:
                run.outputs.append(out)
            if kind == "error":
                run.error = f"{content.get('ename')}: {content.get('evalue')}"

    async def _drain_reply(self, msg_id: str, seconds: float = 2.0) -> str | None:
        """Consume this cell's shell reply (and stale ones before it); return its status."""
        deadline = time.monotonic() + seconds
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                reply = await self.kc.get_shell_msg(timeout=remaining)
            except queue.Empty:
                return None
            if reply["parent_header"].get("msg_id") == msg_id:
                return reply["content"].get("status")
        return None
