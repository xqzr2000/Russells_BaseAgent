# Russell's BaseAgent

A small [ReAct](https://arxiv.org/pdf/2210.03629) agent framework in Python, designed to be subclassed into specialized agents, with a web chat room to try them in.

```python
class YourAgent(BaseAgent):
    # your tools, instructions, and domain magic
```

It grew out of the [CMU 11-768 AI Agents assignment](https://www.cmu-agents.com/#/assignments) and borrows ideas from [smolagents](https://github.com/huggingface/smolagents) (typed tools, small core), [OpenManus](https://github.com/FoundationAgents/OpenManus) (agent state, cleanup), and [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) (composition, events for a dev UI).

**Where it's going:** Jack, the coordinator, will represent me to recruiters at a public URL by summer 2027. The data scientist behind him is being built task by task on real data-science work: a Kaggle competition, then the DABStep and Spider 2.0 benchmarks. See the [roadmap](docs/roadmap.md) and the [showcase](showcase/).

## What does `BaseAgent` do?

`BaseAgent` is the shared runtime, so a specialized agent only describes what makes it different: its instructions, its tools and its domain behaviour. The base class handles the conversation loop, LLM calls, tool calling and validation, memory and long-context compaction, event streaming, step limits, usage tracking and logging.

This is the shape of the real data scientist in [`agents/data_science/agent.py`](src/baseagent/agents/data_science/agent.py):

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
        return SYSTEM_PROMPT

    def get_tools(self):
        return [run_cell, inspect_dataset, ...]  # notebook cells, dataset probe, Colab runs
```

## Agents in the chat room

| Agent | What it does | Docs |
|---|---|---|
| **Jack (the coordinator)** | The default chat. Answers simple things and routes the rest: delegates tasks to specialists, hands the chat to the chess coach | [coordinator](docs/agents/coordinator.md) |
| **Data scientist** | EDA in a Jupyter kernel, or a model trained on a Colab T4 after a CPU smoke run; returns a summary and a downloadable `.ipynb` | [data scientist](docs/agents/data-scientist.md) |
| **Chess coach** | Say *teach me chess* for a guided board with Stockfish hints, or *let's play chess* to play as White | [chess coach](docs/agents/chess-coach.md) |
| **General assistant** | The base agent with a calculator, a clock and notes, for testing the loop | |

> Jack the coordinator is named after a Jack Russell Terrier.

https://github.com/user-attachments/assets/57b80f7d-bdc9-4745-9fb1-bfad1a152d7d

## Quick start

1. Create a GitHub Codespace on this repo (the dev container installs everything).
2. Add `OPENAI_API_KEY` as a Codespaces secret. Without it, the chat room runs on an offline model.
3. Open the **Chat room** port (5173). Drag a CSV in and ask *do EDA on it*, or say *teach me chess*.

Details: [Codespaces setup](docs/setup/codespaces.md) · [Colab T4 setup](docs/setup/colab.md) · `make test` runs the offline tests.

## Repo tree

```
src/baseagent/
  core/            the framework: agent loop, LLM clients, tools, events, memory, skills, delegation, handoff
  agents/          coordinator.py · general.py · chess_coach/ · data_science/ · _template.py
  server/          FastAPI: sessions, SSE stream, uploads, chess endpoints
  sandbox/         locked-down Docker code runner (for the planned Claude Code sandbox)
web/               the chat room (React + Vite)
tests/             offline tests, mirroring src/
evals/             graders for real agent runs, one folder per case
showcase/          each task the agent was built around: task, harness, results, build log
docs/              architecture, adding an agent, per-agent docs, setup, roadmap
skills/            public SKILL.md playbooks
knowledge/         private skills submodule (empty without access)
packages/          agent-sandbox-runtime (Docker sandboxes)
docker/runner/     the code runner's image
scripts/           dev and background-start scripts
```

Start with [the architecture](docs/architecture.md), or build your own agent with [adding an agent](docs/adding-an-agent.md).
