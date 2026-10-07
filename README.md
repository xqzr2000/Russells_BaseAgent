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

Conceptually, a subclass only needs to describe what makes it different:

```python
class DataScienceAgent(BaseAgent):
    name = "data_science"
    title = "Data Science Assistant"
    description = "Performs exploratory data analysis on datasets."

    default_settings = {
        "model": "example-model",
        "max_steps": 10,
    }

    def system_prompt(self) -> str:
        return (
            "You are a data science assistant. "
            "Help users explore datasets, summarize structure, "
            "compute descriptive statistics, and identify patterns or anomalies. "
            "Use the available tools when needed."
        )

    def get_tools(self) -> list[Tool]:
        return [load_dataset_tool, describe_dataset_tool, plot_histogram_tool]
```

## GitHub Codespaces Quick Start

### 1. Fork or clone this repository, then create a Codespace
- If you like the project, give it a star.
- The dev container installs `Python 3.12`, `Node.js 22`, and `uv`, then runs `uv sync` and `npm ci`.

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

## Repo tree
- Still evolving.

