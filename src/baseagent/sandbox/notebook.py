"""A kernel plus the .ipynb notebook that records it, and the ``run_cell`` tool.

Any agent can use notebook-backed Python: create a ``NotebookSession`` in
``setup()``, store it as ``self.notebook``, and add ``run_cell`` to its tools.
Every cell the model runs is appended to the notebook (with its outputs, so
charts appear when the user opens it) and the file is saved after each cell.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import nbformat

from baseagent.events import Attachment
from baseagent.sandbox.kernel import CellResult, KernelSandbox
from baseagent.tools import ToolContext, ToolError, ToolResult, tool
from baseagent.workspace import Workspace

SETUP_CODE = """\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 160)"""

NOTEBOOK_MIME = "application/x-ipynb+json"


class NotebookRecorder:
    """Builds an nbformat v4 notebook cell by cell."""

    def __init__(self) -> None:
        self.nb = nbformat.v4.new_notebook()
        self.nb.metadata = {
            "kernelspec": {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"},
            "language_info": {"name": "python"},
        }

    @property
    def cells(self) -> list:
        return self.nb.cells

    def add_markdown(self, text: str) -> None:
        if text.strip():
            self.nb.cells.append(nbformat.v4.new_markdown_cell(text.strip()))

    def add_code(self, source: str, outputs: list, execution_count: int | None) -> None:
        self.nb.cells.append(nbformat.v4.new_code_cell(
            source.strip("\n"), outputs=outputs, execution_count=execution_count))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        nbformat.write(self.nb, str(tmp))
        os.replace(tmp, path)  # atomic: a download never sees a half-written file


class NotebookSession:
    """What ``run_cell`` needs from its agent: a kernel and the notebook recording it."""

    def __init__(self, workspace: Workspace, setup_code: str = SETUP_CODE, title: str = ""):
        self.workspace = workspace
        self.setup_code = setup_code
        self.title = title
        self.kernel = KernelSandbox()
        self.recorder = NotebookRecorder()
        self.path: Path | None = None
        self.cells_run = 0  # model-written cells, not counting setup

    async def start(self) -> None:
        self.workspace.ensure()
        await self.kernel.start(self.workspace.root)  # cwd = workspace: data/ paths just work
        self.recorder.add_markdown(self.title)
        result = await self.kernel.execute(self.setup_code, timeout=120)
        if not result.ok:
            await self.kernel.shutdown()
            raise RuntimeError(f"Kernel setup failed: {result.error or result.text()}")
        self.recorder.add_code(self.setup_code, result.outputs, result.execution_count)

    async def close(self) -> None:
        await self.kernel.shutdown()

    def set_filename(self, filename: str) -> None:
        self.path = self.workspace.resolve(filename)

    async def run(self, code: str, timeout: float, note: str = "") -> CellResult:
        self.recorder.add_markdown(note)
        try:
            result = await self.kernel.execute(code, timeout=timeout)
        except asyncio.CancelledError:
            stopped = nbformat.v4.new_output("stream", name="stderr",
                                             text="Stopped before this cell finished.\n")
            self.recorder.add_code(code, [stopped], None)
            self.save()
            raise
        self.cells_run += 1
        self.recorder.add_code(code, result.outputs, result.execution_count)
        if result.status == "restarted":
            await self.kernel.execute(self.setup_code, timeout=120)  # restore the imports
        self.save()
        return result

    def save(self) -> Path:
        if self.path is None:
            self.set_filename("analysis.ipynb")
        self.recorder.save(self.path)
        return self.path

    def attachment(self) -> Attachment:
        path = self.save() if self.path is None or not self.path.exists() else self.path
        return Attachment(kind="file", mime=NOTEBOOK_MIME, title=path.name,
                          url=self.workspace.url(path))


@tool(timeout=3600)  # the kernel enforces settings.tool_timeout per cell and interrupts
async def run_cell(code: str, note: str = "", ctx: ToolContext = None) -> ToolResult:
    """Run Python in the persistent Jupyter kernel, like one notebook cell.

    Variables, imports and loaded data persist between calls. Every cell and its
    output is saved to the notebook the user downloads.

    Args:
        code: Python source for one cell.
        note: Optional Markdown placed above the cell in the notebook, e.g. "## Missing values".
    """
    session = getattr(ctx.agent, "notebook", None)
    if not isinstance(session, NotebookSession):
        raise ToolError("This agent has no notebook kernel.")
    if not code.strip():
        raise ToolError("code is empty.")
    result = await session.run(code, timeout=ctx.agent.settings.tool_timeout, note=note)
    return ToolResult(result.text() or "(the cell ran with no output)",
                      attachments=[session.attachment()], is_error=not result.ok)
