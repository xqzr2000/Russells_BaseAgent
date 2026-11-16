"""A kernel plus the .ipynb notebook that records it, and the ``run_cell`` tool.

Any agent can use notebook-backed Python: create a ``NotebookSession`` in
``setup()``, store it as ``self.notebook``, and add ``run_cell`` to its tools.
Every cell the model runs is appended to the notebook (with its outputs, so
charts appear when the user opens it) and the file is saved after each cell.

The saved notebook is meant to be run from the top somewhere else too, Google
Colab in particular, so two things differ from a plain recording:

* cells that failed are left out (the model reruns a corrected cell, and one
  failing cell would stop "Run all");
* a data check cell after the imports lists the ``data/`` files the notebook
  reads, and in Colab asks for the missing ones to be uploaded.
"""

from __future__ import annotations

import asyncio
import os
import re
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

# A data/ path inside a string literal, e.g. pd.read_csv("data/train.csv").
DATA_PATH = re.compile(r"""["'](data/[\w\-./ ]+?)["']""")

DATA_CHECK_NOTE = """\
## Data files

This notebook reads the files listed in the next cell from a `data/` folder beside it. \
In Google Colab the cell asks you to upload any that are missing; anywhere else, put \
them in `data/` and run it again."""

DATA_CHECK_CODE = """\
import sys
from pathlib import Path

DATA_FILES = {files}

print("Python", sys.version.split()[0])
missing = [f for f in DATA_FILES if not Path(f).exists()]
if missing:
    try:
        from google.colab import files  # only exists in Google Colab
    except ImportError:
        raise FileNotFoundError(f"Put these files beside this notebook: {{missing}}") from None
    print("Upload:", ", ".join(Path(f).name for f in missing))
    uploaded = files.upload()
    for f in missing:
        if Path(f).name in uploaded:
            Path(f).parent.mkdir(parents=True, exist_ok=True)
            Path(f).write_bytes(uploaded[Path(f).name])
    missing = [f for f in DATA_FILES if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(f"Still missing: {{missing}}")
print("Data files ready:", ", ".join(DATA_FILES))"""


def data_check_code(files: list[str]) -> str:
    """The source of the data check cell for these ``data/`` paths."""
    return DATA_CHECK_CODE.format(files=repr(list(files)))


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

    def data_paths(self) -> list[str]:
        """The ``data/...`` paths quoted in code cells, in order of first use."""
        found: dict[str, None] = {}
        for cell in self.nb.cells:
            if cell.cell_type == "code":
                found.update(dict.fromkeys(DATA_PATH.findall(cell.source)))
        return list(found)

    def export(self, data_files: list[str]) -> nbformat.NotebookNode:
        """The notebook as saved: with a data check cell after the first code cell.

        The check is added at save time, not recorded, because the files a
        notebook reads are only known once its cells exist. It has no outputs:
        it has not been run here.
        """
        nb = self.nb.copy()
        nb.cells = list(self.nb.cells)
        if data_files:
            first_code = next((i for i, c in enumerate(nb.cells) if c.cell_type == "code"), -1)
            nb.cells[first_code + 1:first_code + 1] = [
                nbformat.v4.new_markdown_cell(DATA_CHECK_NOTE),
                nbformat.v4.new_code_cell(data_check_code(data_files)),
            ]
        return nb

    def save(self, path: Path, data_files: list[str] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        nbformat.write(self.export(data_files or []), str(tmp))
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
        self.cells_failed = 0  # of those, cells that raised and were left out of the notebook

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
        try:
            result = await self.kernel.execute(code, timeout=timeout)
        except asyncio.CancelledError:
            stopped = nbformat.v4.new_output("stream", name="stderr",
                                             text="Stopped before this cell finished.\n")
            self.recorder.add_markdown(note)
            self.recorder.add_code(code, [stopped], None)
            self.save()
            raise
        self.cells_run += 1
        if result.status == "error":
            self.cells_failed += 1  # left out: the notebook must run from the top
        else:
            self.recorder.add_markdown(note)
            self.recorder.add_code(code, result.outputs, result.execution_count)
        if result.status == "restarted":
            await self.kernel.execute(self.setup_code, timeout=120)  # restore the imports
        self.save()
        return result

    def data_files(self) -> list[str]:
        """The files under data/ that the notebook's cells read and that exist here."""
        files = []
        for relative in self.recorder.data_paths():
            try:
                if self.workspace.resolve(relative).is_file():
                    files.append(relative)
            except ValueError:  # e.g. "data/../../x": not a workspace file
                continue
        return files

    def save(self) -> Path:
        if self.path is None:
            self.set_filename("analysis.ipynb")
        self.recorder.save(self.path, self.data_files())
        return self.path

    def attachment(self) -> Attachment:
        path = self.save() if self.path is None or not self.path.exists() else self.path
        return Attachment(kind="file", mime=NOTEBOOK_MIME, title=path.name,
                          url=self.workspace.url(path))


@tool(timeout=3600)  # the kernel enforces settings.tool_timeout per cell and interrupts
async def run_cell(code: str, note: str = "", ctx: ToolContext = None) -> ToolResult:
    """Run Python in the persistent Jupyter kernel, like one notebook cell.

    Variables, imports and loaded data persist between calls. Every cell that runs
    without an error is saved, with its output, to the notebook the user downloads.

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
    text = result.text() or "(the cell ran with no output)"
    if result.status == "error":
        text += ("\n\n(This cell is not kept in the notebook. Run the whole corrected cell "
                 "again, including any lines that worked.)")
    return ToolResult(text, attachments=[session.attachment()], is_error=not result.ok)
