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

## Plan for the EDA agent

Goal: the agent loads a CSV, explores it with pandas and seaborn in a
**persistent Python kernel**, and the chat room shows its charts inline.

### 1. Dependencies

```bash
uv add pandas seaborn matplotlib ipykernel jupyter-client
```

### 2. A kernel sandbox (`src/baseagent/sandbox/kernel.py`)

A Jupyter kernel keeps variables between calls, like notebook cells, and
reports charts as `image/png` display data.

```python
from jupyter_client.manager import AsyncKernelManager

class KernelSandbox:
    async def start(self, cwd: str):
        self.km = AsyncKernelManager(kernel_name="python3")
        await self.km.start_kernel(cwd=cwd)
        self.kc = self.km.client()
        self.kc.start_channels()
        await self.kc.wait_for_ready(timeout=60)
        await self.run("%matplotlib inline\nimport pandas as pd, seaborn as sns, matplotlib.pyplot as plt")

    async def run(self, code: str, timeout: float = 120):
        msg_id = self.kc.execute(code)
        text, images, error = [], [], None
        while True:
            msg = await self.kc.get_iopub_msg(timeout=timeout)
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            kind, content = msg["msg_type"], msg["content"]
            if kind == "stream":
                text.append(content["text"])
            elif kind in ("execute_result", "display_data"):
                data = content["data"]
                if "image/png" in data:
                    images.append(data["image/png"])        # base64 already
                elif "text/plain" in data:
                    text.append(data["text/plain"])
            elif kind == "error":
                error = f"{content['ename']}: {content['evalue']}"
            elif kind == "status" and content["execution_state"] == "idle":
                return "".join(text), images, error

    async def stop(self):
        self.kc.stop_channels()
        await self.km.shutdown_kernel(now=True)
```

### 3. The agent (`src/baseagent/agents/eda.py`)

```python
@tool(timeout=300)
async def run_python(code: str, ctx: ToolContext) -> ToolResult:
    """Run Python in the persistent analysis kernel (variables persist).
    pandas as pd, seaborn as sns, matplotlib.pyplot as plt are imported.

    Args:
        code: Python code. End with plt.show() to display a chart.
    """
    text, images, error = await ctx.agent.kernel.run(code)
    content = (text or "(no output)") + (f"\nError: {error}" if error else "")
    content += f"\n[{len(images)} chart(s) shown to the user]" if images else ""
    return ToolResult(content, is_error=bool(error), attachments=[
        Attachment(kind="image", mime="image/png", data=img) for img in images])

@register_agent
class EDAAgent(BaseAgent):
    name, title = "eda", "Data scientist (EDA)"
    description = "Explores a dataset with pandas and seaborn and charts what it finds."
    default_settings = {"max_steps": 30, "compact_threshold_tokens": 24000}
    skills_dir = ROOT / "skills" / "eda"           # e.g. an eda-workflow skill

    async def setup(self):
        self.kernel = KernelSandbox()
        await self.kernel.start(cwd=str(ROOT / "data"))

    async def close(self):
        await self.kernel.stop()

    def get_tools(self):
        return [run_python, list_data_files]
```

### 4. Then, in order

* Add a `data/` folder (git-ignored) and a `list_data_files` tool.
* Add an `eda-workflow` skill: profile → missing values → univariate →
  bivariate → findings, with conventions for chart titles and sizes.
* Optional: send each chart back to the model as an image so it can read
  its own plots (vision models), and export the session as `.ipynb`.
* Isolation: the kernel first runs inside the dev container, which is
  fine for your own datasets. Before running untrusted data or code, move
  the kernel into a Docker container (add the `docker-in-docker` devcontainer
  feature) with no network and CPU/memory limits; the agent code doesn't
  change, only `KernelSandbox.start`.
