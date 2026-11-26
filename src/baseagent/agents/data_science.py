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
from baseagent.sandbox.colab import (
    run_colab_notebook,
    save_colab_cell,
    smoke_test_notebook,
    start_colab_notebook,
)
from baseagent.sandbox.kaggle import kaggle_dataset_in
from baseagent.sandbox.notebook import NotebookSession, run_cell
from baseagent.sandbox.probe import inspect_dataset
from baseagent.workspace import Workspace

ROOT = Path(__file__).resolve().parents[3]
KNOWLEDGE_SKILLS = ROOT / "knowledge" / "skills"  # private submodule; absent for public users
COLAB_MIN_STEPS = 45

BASICS = """\
Data-science basics (every task)
- Splits: use a split the data ships with, unchanged. Otherwise hold out a stratified test \
set, or use cross-validation for small data; split by group or time when rows are related. \
Fit every preprocessing step on training data only. Choose on validation; evaluate on \
test once, at the end.
- Leakage: nothing from validation, test or the target may reach the training features.
- Start with the simplest sensible baseline and a metric that fits the problem. Report \
test results with the test size, and don't call differences smaller than the noise real.
- Skills: before work a skill in the catalog covers, load it with invoke_skill and follow \
it. In your final answer, name the skills you used ("From the data-splitting skill: ...") \
and mark other method advice as general knowledge, not from a skill."""
DATA_FILE = re.compile(r"[\w\-.]+\.(?:csv|tsv|txt|json|xlsx?|parquet)\b", re.IGNORECASE)

SYSTEM_PROMPT = f"""\
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
- If a cell fails, read the error and run the whole cell again, corrected. Failed cells \
are left out of the notebook, so later cells must not depend on anything a failed cell did.
- The notebook must run from the top on another machine (the user may open it in Google \
Colab): read data only with relative data/ paths written as plain strings, and don't \
depend on files or variables that no cell creates.

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

{BASICS}

When you are done, reply without calling a tool, in concise Markdown: the dataset's \
size, key facts about the columns, data quality issues (missing values, duplicates, \
suspicious types, constant or ID columns) and notable statistics. Use exact numbers \
from your outputs. The notebook is saved automatically; don't paste its contents."""


COLAB_PROMPT = f"""\
You are a data scientist building a notebook that runs on a Colab T4. You can't run cells \
one by one: you write the draft, check it with a smoke run on this machine's CPU, and only \
then run it on Colab (one run, at most 15 minutes, including the data download).

Workflow
0. Load the skills in the catalog below that fit the task (invoke_skill), one per call.
1. Look before you write. For a Kaggle dataset (URL or owner/dataset) call \
inspect_dataset with it; for uploaded files call inspect_dataset("data"). Base every \
path, column name, label and count in your code on that report, never on guesses.
2. start_colab_notebook (pass the Kaggle URL as kaggle_source). Cells written by BaseAgent \
come first and define:
   - SMOKE (bool): True only in the smoke run.
   - OUTPUT_DIR (pathlib.Path): write only the files to return here, e.g. OUTPUT_DIR / "model.pt". \
Keep caches and intermediate tensors in memory or under /tmp.
   - dataset_root (pathlib.Path), for a Kaggle dataset. It can be read-only on Colab.
   Use these variables as they are, in every run: no fallbacks, no reassigning (save_colab_cell \
refuses a cell that does), and don't download the data again.
3. save_colab_cell: one short, complete cell per tool call, one call per response. Give \
each cell a Markdown note (a "## Heading" and a sentence). Edit a cell by saving its number \
again. Split a cell if its tool call gets long.
4. smoke_test_notebook(inputs, outputs). It runs every cell from the top with SMOKE = True. \
When SMOKE is True the notebook must use a tiny subset (a few examples per class) and one \
short epoch or a few batches, but otherwise run exactly the same code, including writing \
every declared output. Fix the cell it reports and run it again until it passes. Outputs are \
the files you write to OUTPUT_DIR; analysis.ipynb is added for you.
5. run_colab_notebook with the same inputs and outputs. It refuses a draft that hasn't \
passed a smoke run since its last change.

Writing the cells
- Build your own table of examples (path, label, split, ...) from the report, over the full \
data in every run, SMOKE included: apply the exclusions and handle every problem the report \
flags (such as rows that aren't files), then check that every path in the table exists. Only \
then take the SMOKE subset, for decoding and training. A problem the subset skips would \
otherwise first appear on Colab.
- Before training, assert what the report and the task lead you to expect: the class count, \
non-empty splits, the excluded rows. Print the counts you'll need to report, from the rows \
actually used.
- Print what someone would need to debug a failure: shapes, counts, column names.
- Devices: device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); never \
assume a GPU. Colab already has torch, torchvision, numpy, pandas, scikit-learn and Pillow: \
don't install them.
- Put training on a time budget and print each stage's duration.
- Don't add fallback downloads, credential files or API keys. If Kaggle denies access, \
explain that and ask the user to resolve it on Kaggle.
- Don't use model files that ship with a dataset unless the task asks for them.
- Declare uploaded inputs by exact data/ path; use [] when the notebook downloads its data.

{BASICS}

The task
- Honor the user's model, preprocessing, split and evaluation requirements. Choose \
hyperparameters and checkpoints on validation data and evaluate the final model once on \
the held-out test data.
- An inference function asked for must work from the saved files alone (weights, label \
map) in a fresh session: it may use functions, classes and UPPER_CASE constants defined \
earlier, nothing built from data or training. The notebook runs unattended: demo it on an \
example from the data.
- Report metrics, trained models and completed execution only after a successful Colab \
run; smoke-run numbers come from a tiny subset. If blocked, explain the exact blocker and \
say the attached file is an unexecuted draft.
"""


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
    brief = ("EDA on a tabular data file or model training on a dataset in Jupyter or Colab. "
             "Returns a notebook and artifacts.")
    delegatable = True
    optional_skills_dirs = (KNOWLEDGE_SKILLS,)
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
            f"{datetime.now():%Y-%m-%d %H:%M}. Run it from the top to reproduce, here or in "
            "Google Colab; data files are read from `data/` next to this notebook."
        )
        self.notebook = NotebookSession(self.workspace, title=title)
        await self.notebook.start()

    async def close(self) -> None:
        if self.notebook is not None:
            await self.notebook.close()  # stops the kernel; the saved notebook stays

    def system_prompt(self) -> str:
        if self.settings.notebook_runtime == "colab-t4":
            return COLAB_PROMPT
        return SYSTEM_PROMPT

    def get_tools(self):
        return [run_cell, inspect_dataset, start_colab_notebook, save_colab_cell, smoke_test_notebook,
                run_colab_notebook]

    async def run(self, user_message: str) -> TurnResult:
        """The base loop, framed in the notebook by the request and the findings."""
        self.state.pop("colab_outputs", None)
        if self.settings.notebook_runtime == "colab-t4":
            self.state["requested_kaggle_dataset"] = kaggle_dataset_in(user_message)
            # inspect, ~12 cells, smoke runs with repairs, then Colab: more steps than EDA.
            if self.settings.max_steps < COLAB_MIN_STEPS:
                self.settings = self.settings.model_copy(update={"max_steps": COLAB_MIN_STEPS})
            return await super().run(user_message)
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
        if outputs := self.state.get("colab_outputs"):
            return outputs
        if self.settings.notebook_runtime == "colab-t4" or self.notebook is None or self.notebook.cells_run == 0:
            return []
        return [self.notebook.attachment()]
