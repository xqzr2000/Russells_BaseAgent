# Data scientist

The data scientist is a delegate specialist: Jack hands it a self-contained task, it
works in a Jupyter kernel in the Codespace, and it returns a summary plus downloadable
files. Every step is a notebook cell, so the work it did is exactly the notebook you
download.

Code: [`src/baseagent/agents/data_science/`](../../src/baseagent/agents/data_science/).
Tests: [`tests/agents/test_data_science.py`](../../tests/agents/test_data_science.py),
[`tests/agents/test_colab.py`](../../tests/agents/test_colab.py).
How it was built up task by task: [`showcase/`](../../showcase/).

## Two ways to run

| Notebook compute (Settings) | What happens | Typical use |
| --- | --- | --- |
| **Local CPU** (default) | Cells run in the chat's kernel, one at a time | EDA on an uploaded CSV |
| **Colab T4** | The agent drafts a notebook, smoke-runs it on the CPU, then runs it on a Colab T4 | Training a model on a Kaggle dataset |

### EDA on a CSV (Local CPU)

1. Drag a CSV into the chat (or click the paperclip).
2. Ask: *do EDA on it*.
3. Jack delegates to the data scientist. Its progress appears under the `delegate` step;
   when it's done you get Jack's summary and a **Download** card for the notebook.

The kernel runs in `workspace/<chat id>/`, next to the uploaded `data/` folder, so the
notebook re-runs from the top there. Nothing in `workspace/` or `runs/` is committed.
Without an API key, the offline `fake-echo` model walks the same path with one canned
EDA cell, which is handy for checking the plumbing.

### A Colab T4 run

Sign in once first: [Colab setup](../setup/colab.md). Then select **Colab T4** under
**Settings → Notebook compute**, use a real chat model, and paste a public Kaggle dataset
URL (or upload the data files) with the objective and the metric.

1. **Look at the data.** `inspect_dataset` downloads the dataset in the Codespace (cached
   after the first time) and reports its layout, each table's columns, whether path
   columns point to real files, image sizes and any model files shipped with it. It works
   the same for any dataset, or for the chat's uploaded `data/` files.
2. **Load the skills that fit** (see below), before the first cell.
3. **Write the draft,** one cell at a time (`start_colab_notebook`, `save_colab_cell`).
   Cells written by BaseAgent come first and define the harness variables: `SMOKE`,
   `OUTPUT_DIR` (where every returned file is written; `/content` on Colab) and, for
   Kaggle, `dataset_root`. A cell that reassigns them is refused.
4. **Smoke run.** `smoke_test_notebook` runs every cell from the top on the Codespace CPU
   with `SMOKE = True` (a tiny subset, one short epoch) and checks that each declared
   output was written. A failure names the cell, in seconds, not after a Colab run.
5. **Colab run.** `run_colab_notebook` only runs a draft that passed a smoke run after its
   last change, with the same inputs and outputs.

Each run starts a fresh Colab session, checks the assigned GPU, uploads the declared
input files, and returns the executed notebook and the declared outputs. The runtime is
released on completion, failure or Stop. A run has a 15-minute limit, and T4 access
depends on Colab capacity and your quota. If execution fails, the chat still returns the
saved `colab_draft.ipynb`.

The downloaded notebook runs from the top in Colab (**File → Upload notebook**, then
**Run all**): a **Data files** cell lists the files it reads and asks you to upload the
missing ones, and cells that raised an error are left out.

## Knowledge skills (private)

The data scientist loads concept skills from `knowledge/`, a git submodule of the
private `jack-knows-it-all` repo (for example `data-splitting`, `composite-labels`,
`image-classification-baseline`). Its prompt lists each skill under its area with the
scenarios it explains; the agent loads a skill with `invoke_skill` before the work it
covers, and its answer names the skills it used. Skills are written from finished tasks,
then reviewed by Russell.

- New codespaces check out `knowledge/` by themselves; GitHub asks you to authorize read
  access to the private repo when the codespace is created.
- `make knowledge` pulls newly merged skills; new chats use them. Commit the moved
  `knowledge` pointer when convenient.
- Without access (public users), `knowledge/` stays empty and the agent works without it.

## Inside

| Module | What it does |
| --- | --- |
| `agent.py` | The prompt (EDA checklist, data-science basics, Colab rules) and a `run()` wrapper that frames each request in the notebook with "## Request" and "## Findings" |
| `kernel.py` | `KernelSandbox`: one IPython kernel in this project's `.venv` (so the `datascience` dependency group is importable), started in the chat's workspace with secret environment variables removed. Handles errors, timeouts (interrupt; variables survive), crashes (restart) and Stop |
| `notebook.py` | `NotebookSession` (kernel + `.ipynb` recorder) and the `run_cell(code, note)` tool; the file is saved after every cell. Any agent can reuse it |
| `probe.py` | `inspect_dataset` |
| `kaggle.py` | Kaggle dataset references and downloads |
| `colab.py` | Drafts, cell checks, the smoke run and the Colab CLI run |

The model reads text only: charts are saved into the notebook for the user, and the
prompt tells the agent to base conclusions on printed numbers.

## Graded runs

Real runs are graded by hand with the evals in [`evals/`](../../evals/), for example
`make eval-card` for the card classifier.
