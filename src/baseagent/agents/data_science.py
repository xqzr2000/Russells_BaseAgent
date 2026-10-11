"""The data science agent: EDA in a persistent Jupyter kernel, delivered as an .ipynb.

It runs every step as a notebook cell (``run_cell``), so the work it does is
exactly the notebook the user downloads. One agent instance keeps one kernel
and one notebook for the whole chat, so follow-up requests ("now look at
correlations") reuse the loaded data and extend the same notebook.

v1 does basic EDA. It cannot see images: charts are saved into the notebook
for the user, and the agent reasons from printed numbers.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from baseagent.agent import BaseAgent, TurnResult
from baseagent.agents import register_agent
from baseagent.events import Attachment
from baseagent.sandbox.notebook import NotebookSession, run_cell
from baseagent.workspace import Workspace

ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = re.compile(r"[\w\-.]+\.(?:csv|tsv|txt|json|xlsx?|parquet)\b", re.IGNORECASE)

SYSTEM_PROMPT = """\
You are a data scientist working in a live Jupyter kernel. Each `run_cell` call runs \
Python in the same kernel (variables persist, like notebook cells) and is recorded in a \
notebook the user downloads as an .ipynb file, so write clean, readable cells.

Environment
- The working directory is the chat workspace. Uploaded files are in data/ \
(e.g. data/titanic.csv); use paths exactly as given in the task.
- Already imported: pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns. \
scipy, scikit-learn and statsmodels are installed.
- You cannot see images. Charts are saved into the notebook for the user, but base \
every conclusion on numbers you print.

How to work
- One logical step per cell. Give each cell a short Markdown `note` heading \
(e.g. "## Missing values") so the notebook reads like a report.
- Load each dataset once into a variable (e.g. df) and reuse it.
- Keep printed output compact: never print a whole large DataFrame; use head(), \
value_counts().head(20), and small summary tables.
- When cells don't depend on each other's output, call run_cell several times in one response.
- If a cell fails, read the error and run a corrected cell.

Basic EDA (when asked for EDA without more specific instructions)
1. Load the file (if it looks wrong, check the delimiter, header and encoding) and show the first rows.
2. Size: total rows and total columns.
3. Columns: name, dtype and non-null count; flag columns whose type looks wrong \
(numbers or dates stored as text).
4. Missing values: count and percentage per column, highest first.
5. Unique values: nunique per column; value counts for columns with at most 20 \
distinct values; flag constant columns and ID-like columns (one distinct value per row).
6. Duplicate rows: how many.
7. Summary statistics: describe() for numeric columns; count, unique, top and freq for text columns.

When you are done, reply without calling a tool, in concise Markdown: the dataset's \
size, key facts about the columns, data quality issues (missing values, duplicates, \
suspicious types, constant or ID columns) and notable statistics. Use exact numbers \
from your outputs. The notebook is saved automatically; don't paste its contents."""


def notebook_filename(task: str) -> str:
    """titanic_eda.ipynb for a task that mentions data/titanic.csv; else analysis.ipynb."""
    match = DATA_FILE.search(task)
    stem = re.sub(r"[^\w\-]+", "_", Path(match.group(0)).stem).strip("_") if match else ""
    return f"{stem}_eda.ipynb" if stem else "analysis.ipynb"


def blockquote(text: str) -> str:
    return "\n".join(f"> {line}" if line.strip() else ">" for line in text.strip().splitlines())


@register_agent
class DataScienceAgent(BaseAgent):
    name = "data_science"
    title = "Data scientist"
    description = (
        "Explores tabular data files (CSV and similar) in a persistent Jupyter kernel: row and "
        "column counts, column types, missing values, unique values, duplicates and summary "
        "statistics. Records every step in a downloadable .ipynb notebook and returns a short "
        "summary of its findings. Give it the file path (e.g. data/sales.csv)."
    )
    delegatable = True
    default_settings = {"max_steps": 25, "tool_timeout": 180, "compact_threshold_tokens": 40_000}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.notebook: NotebookSession | None = None  # started in setup()

    async def setup(self) -> None:
        if self.workspace is None:  # running outside the server (scripts, tests)
            self.workspace = Workspace(ROOT / "workspace" / self.session_id)
        title = (
            "# Exploratory data analysis\n\n"
            f"Recorded by the `{self.name}` agent of Russell's BaseAgent on "
            f"{datetime.now():%Y-%m-%d %H:%M}. Run it from the top to reproduce; data files are "
            "read from `data/` next to this notebook."
        )
        self.notebook = NotebookSession(self.workspace, title=title)
        await self.notebook.start()

    async def close(self) -> None:
        if self.notebook is not None:
            await self.notebook.close()  # stops the kernel; the saved notebook stays

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def get_tools(self):
        return [run_cell]

    async def run(self, user_message: str) -> TurnResult:
        """The base loop, framed in the notebook by the request and the findings."""
        await self.ensure_ready()
        notebook = self.notebook
        if notebook.path is None:
            notebook.set_filename(notebook_filename(user_message))
        notebook.recorder.add_markdown(f"## Request\n\n{blockquote(user_message)}")
        result: TurnResult | None = None
        try:
            result = await super().run(user_message)
        finally:  # also on Stop: the user keeps whatever was done
            if result and result.stop_reason == "final" and result.final_text.strip():
                notebook.recorder.add_markdown(f"## Findings\n\n{result.final_text}")
            notebook.save()
        return result

    def output_files(self) -> list[Attachment]:
        if self.notebook is None or self.notebook.cells_run == 0:
            return []
        return [self.notebook.attachment()]
