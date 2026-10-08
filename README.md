# Russell's BaseAgent

A small [ReAct](https://arxiv.org/pdf/2210.03629) agent framework in Python, designed to be subclassed into specialized agents. 

```python
class YourAgent(BaseAgent):
    # your tools, instructions, and domain magic
```

It grew out of the [CMU 11-768 AI Agents assignment](https://www.cmu-agents.com/#/assignments) and borrows ideas from [smolagents](https://github.com/huggingface/smolagents) (typed tools, small core), [OpenManus](https://github.com/FoundationAgents/OpenManus) (agent state, cleanup), and [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) (composition, events for a dev UI).


## What does `BaseAgent` do?

`BaseAgent` is the shared runtime for your specialized agents.

Instead of re-implementing the agent loop each time, you can focus on the parts that make your agent unique:
- its instructions
- its tools
- its domain-specific behavior

`BaseAgent` handles the common runtime tasks for you, including:
- the conversation loop
- LLM calls
- tool calling and execution
- memory management
- long-context compaction
- event streaming
- step limits
- usage tracking
- logging

### Example: `class DataScienceAgent(BaseAgent):`

A subclass only describes what makes it different. This is the shape of the real one in
[`agents/data_science.py`](src/baseagent/agents/data_science.py):

```python
@register_agent
class DataScienceAgent(BaseAgent):
    name = "data_science"
    title = "Data scientist"
    description = "Explores tabular data files in a persistent Jupyter kernel ..."
    delegatable = True  # the coordinator can hand it tasks

    async def setup(self):  # one kernel + one notebook per chat
        self.notebook = NotebookSession(self.workspace, title=...)
        await self.notebook.start()

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT  # the basic EDA checklist

    def get_tools(self):
        return [run_cell]  # runs Python as a notebook cell, saves the .ipynb
```

## Agents in the chat room

| Agent | What it does |
|---|---|
| **Coordinator** (default) | Your main chat. Answers simple things itself and delegates specialized work to any agent marked `delegatable`. |
| **Data scientist** | Basic EDA in a Jupyter kernel: rows and columns, types, missing values, unique values, duplicates, summary statistics. Returns a summary and a downloadable `.ipynb`. |
| **General assistant** | The base agent with a calculator, a clock and notes, for testing the loop. |

### EDA on a CSV

1. Drag a CSV into the chat (or click the paperclip).
2. Ask: *do EDA on it*.
3. The coordinator delegates to the data scientist. Its progress appears under the
   `delegate` step; when it's done you get a summary and a **Download** card for the notebook.

The kernel runs inside the Codespace, in `workspace/<chat id>/` next to the uploaded
`data/` folder, so the notebook re-runs from the top there. Nothing in `workspace/` or
`runs/` is committed. Without an API key, the offline `fake-echo` model walks the same
path with one canned EDA cell, which is handy for checking the plumbing.

## GitHub Codespaces Quick Start

### 1. Fork or clone this repository, then create a Codespace
- If you like the project, give it a star.
- The dev container installs `Python 3.12`, `Node.js 22`, and `uv`, then runs `uv sync` and `npm ci`.
- `uv sync` also installs the data science stack the agent's kernel uses (pandas, numpy, scipy,
  matplotlib, seaborn, scikit-learn, statsmodels, pyarrow, openpyxl). In an existing Codespace,
  run `uv sync` once after pulling.

### 2. Add `OPENAI_API_KEY` as a Codespaces secret
- In GitHub, go to **Settings → Codespaces → New secret**
- Name it `OPENAI_API_KEY`
- Grant access to this repository
- Optional: set `OPENAI_MODEL` to override the default model
- Without an API key, the chat room still works using the offline `fake-echo` model

### 3. Start the app
```bash
make dev
```
- When prompted, open the Chat room port (5173)
- You can also open it later from the Ports tab

### See this chat room in action.

https://github.com/user-attachments/assets/7746dc0f-3124-4213-a29c-baa4d36d265c

## Repo tree

```
src/baseagent/
  agent.py              BaseAgent: the ReAct loop every agent inherits
  tools.py, events.py   typed tools; events the UI and logs listen to
  delegation.py         agent-as-tool: `delegate` and the sub-agent pool
  workspace.py          each chat's folder (uploads in data/, outputs)
  sandbox/              persistent Jupyter kernel + .ipynb recorder, `run_cell`
  agents/               coordinator, data_science, general, _template
  server/               FastAPI: sessions, SSE streaming, file upload/download
web/                    React chat room (Vite)
tests/                  offline tests (scripted model, real kernel)
docs/ADDING_AN_AGENT.md how to write a specialized agent
```

