# Adding a specialized agent

Every agent is a small subclass of `BaseAgent`. The base class owns the ReAct
loop: prompting, streaming, tool dispatch, argument validation, error
recovery, step limits, compaction, events, and logs. Your subclass only says
what is different.

## The five-minute version

1. `cp src/baseagent/agents/_template.py src/baseagent/agents/my_agent.py`
2. Rename the class, set `name`, `title`, `description`, `examples`.
3. Uncomment `@register_agent`.
4. Write tools with `@tool` and return them from `get_tools()`.
5. Save. `make dev` reloads the server, and the agent appears in the chat
   room's agent picker. Start a new chat with it.

No server or UI changes are needed: `/api/agents` lists every registered
agent, and the UI renders any tool's results and attachments.

## What you can override

| Hook | Default | Override when |
|---|---|---|
| `system_prompt()` | generic assistant | always |
| `get_tools()` | none | the agent needs to act |
| `async setup()` / `close()` | nothing | you hold resources: a kernel, DB connection, browser |
| `is_final(message)` | final when the reply has no tool calls | the agent must finish through a specific tool (e.g. `submit`) |
| `format_observation(result)` | `Error: ` prefix + head/tail truncation | results need special formatting |
| `default_settings` | `AgentSettings()` | different step limit, compaction, model |
| `skills_dir` | none | the agent has `SKILL.md` playbooks |
| `delegatable` | `False` | the coordinator should be able to hand it tasks |
| `output_files()` | none | it produces files the user should download after a delegation |

### The three subclassing rules

1. **`__init__` only stores options.** Call `super().__init__(*args, **kwargs)`
   first, then set attributes. Do I/O in `setup()`, which runs once before the
   first message and can `await`.
2. **Don't reimplement the loop.** If you need a different flow, add a hook to
   `BaseAgent` so every agent benefits.
3. **Make capabilities composable.** Put reusable tools in a module (like
   `builtin_tools.py`) and import them into any agent; don't inherit from
   another specialized agent just to get its tools.

## Tools in detail

```python
@tool
def describe_column(column: str, top_n: int = 10, ctx: ToolContext = None) -> ToolResult:
    """Summarize one column of the loaded dataframe.

    Args:
        column: Column name exactly as in df.columns.
        top_n: How many most-frequent values to list.
    """
```

* The docstring's first paragraph is the tool description; `Args:` lines
  become argument descriptions in the JSON Schema.
* Type hints define and **validate** arguments (Pydantic). Wrong types,
  missing or extra arguments come back to the model as a readable error.
* A `ToolContext` parameter is injected (hidden from the model):
  `ctx.agent`, `ctx.state` (per-chat scratch dict), `ctx.call_id`.
* Raise `ToolError("...")` for problems the model should fix and retry.
  Any other exception is also caught and reported; the run continues.
* Return a `str`, any JSON-able value, or a `ToolResult(content, attachments)`.
  `content` is what the model sees. `attachments` are only for the UI:
  `Attachment(kind="image", mime="image/png", data=<base64>)` for charts,
  `kind="table"` with `text/markdown` or `text/html` for dataframes.
* `@tool(exclusive=True)` means at most one call per model response (for
  tools that change shared state). `@tool(timeout=300)` overrides the
  per-call timeout.

## Delegation: making a specialist the coordinator can use

The chat room's default agent is the **coordinator**. It answers simple
questions itself and hands specialized work to other agents through one tool,
`delegate(agent, task)` (see `delegation.py`):

* Any registered agent with `delegatable = True` appears in that tool's
  description, using its `name` and `description`. Write the `description`
  for the coordinator: what the agent does and what to put in the task.
* The specialist runs its own loop with its own memory. Only its final answer
  (and `output_files()`) goes back to the coordinator, so the coordinator's
  context stays small. The specialist cannot see the conversation, which is
  why the coordinator is told to write self-contained tasks.
* One specialist instance per chat, created on first use and kept for
  follow-ups. It follows the chat's model choice (`model`,
  `reasoning_effort`, `temperature`) but keeps its own step limits.
* Its steps show up as progress under the `delegate` call
  (`ctx.progress(...)`), and its full trajectory is logged to
  `runs/<session>.<agent>.json`.

Nothing in the coordinator changes when you add a specialist.

## Files: workspaces, uploads and downloads

Each chat gets a folder, `workspace/<session id>/` (`agent.workspace`).
Files dropped into the chat are uploaded to its `data/` folder and listed at
the end of the user's message ("Attached file: data/sales.csv"). Any file in
the workspace can be offered as a download: return
`Attachment(kind="file", url=workspace.url(path), title=...)` from a tool, and
the chat room lists it under the turn's answer. Specialists share the
coordinator's workspace.

## How the data science agent works

* `sandbox/kernel.py`: `KernelSandbox`, one IPython kernel in this project's
  `.venv` (so every package in the `datascience` dependency group is
  importable), started in the chat's workspace with secret environment
  variables removed. It handles errors, timeouts (interrupt, variables
  survive), crashes (restart) and Stop (interrupt).
* `sandbox/notebook.py`: `NotebookSession` (kernel + `.ipynb` recorder) and
  the `run_cell(code, note)` tool. Each cell is appended to the notebook with
  its outputs, and the file is saved after every cell. Any agent can reuse
  this: start a `NotebookSession` in `setup()`, keep it as `self.notebook`,
  and add `run_cell` to its tools.
* `agents/data_science.py`: the EDA instructions, and a `run()` wrapper that
  frames each request in the notebook with "## Request" and "## Findings".

The model reads text only: charts are saved into the notebook for the user,
and the prompt tells the agent to base conclusions on printed numbers.

### Ideas for later

* EDA recipes as skills (`skills/data_science/...`): missing-data review,
  outliers, correlations, time series.
* Let a vision model see its charts (send `image/png` outputs back to it).
* Run the kernel in its own container (no network, CPU and memory limits)
  before analysing untrusted data. Only `KernelSandbox.start` changes.
