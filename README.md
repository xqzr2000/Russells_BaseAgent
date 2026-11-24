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
| **Coordinator** (default) | Jack the coordinator, answers simple things and delegates specialized work to any agent marked `delegatable`. |
| **Data scientist** | Do basic EDA in a Jupyter kernel. Returns a summary and a downloadable `.ipynb`. |
| **General assistant** | The base agent with a calculator, a clock and notes, for testing the loop. |
| **Chess coach** | Say *teach me chess* to open a guided board with Stockfish hints and an explanation beside it. |
| **Code agent** | Paste code (in a ``` block) or attach a file and ask *fix this*: OpenCode fixes and explains it in a sandbox with a free model. |

> Jack the Coordinator is a Jack Russell Terrier. I'm building him to be interactive, with an LLM handling the intelligence and `Three.js` bringing him to life.
> Wish me luck!

### Chess teaching demo

Say *teach me chess* for a piece drill or *let’s play chess* to play as White.
Select **Hint** for a move and its arrow without waiting for a chat reply.
**Play next move** plays one engine move for the side to move so you can follow both sides.
The coach’s latest explanation, move narration, and next step stay next to the board.

Stockfish is installed in new Codespaces. Rebuild an existing container to pick it up.
For local runs, install Stockfish and put it on your PATH or set `STOCKFISH_PATH` to the executable.
The built-in practice engine covers lessons without kings and installations without Stockfish.

### EDA on a CSV

1. Drag a CSV into the chat (or click the paperclip).
2. Ask: *do EDA on it*.
3. The coordinator delegates to the data scientist. Its progress appears under the
   `delegate` step; when it's done you get a summary and a **Download** card for the notebook.

The kernel runs inside the Codespace, in `workspace/<chat id>/` next to the uploaded
`data/` folder, so the notebook re-runs from the top there. Nothing in `workspace/` or
`runs/` is committed. Without an API key, the offline `fake-echo` model walks the same
path with one canned EDA cell, which is handy for checking the plumbing.

**Opening the notebook in Google Colab.** The downloaded notebook is written to run from
the top on another machine:

- After the imports there is a **Data files** cell listing the `data/` files the notebook
  reads. In Colab it asks you to upload the missing ones; anywhere else it tells you to
  put them in `data/` beside the notebook. Data files are never stored in the notebook.
- Cells that raised an error are left out. The data scientist reruns a corrected cell,
  and only cells that ran are kept, so **Runtime → Run all** doesn't stop halfway.

In Colab: **File → Upload notebook** (or open it from GitHub once it's in a repo), then
**Run all**.

### Run a notebook on Colab T4

Use the official [Google Colab CLI](https://github.com/googlecolab/google-colab-cli)
to run an analysis or training notebook from the chat on a remote T4.

1. In the Codespace terminal, run `make colab-setup` to install the CLI. Have
   [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) available to sign in.
2. Configure Google Application Default Credentials for the account you use with
   Colab. Complete sign-in in your browser, then check access with `colab --auth adc usage`.

   ```bash
   gcloud auth application-default login --no-launch-browser \
     --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
   ```

3. In a chat with Jack or the data scientist, open **Settings → Notebook compute**,
   select **Colab T4**, and save. Use a real chat model to plan the notebook.
4. For a competition pilot, upload the data files you are allowed to use with Colab
   (for example, train, test, and sample submission CSVs). Give the agent the
   challenge objective, metric and submission requirements. Ask it to validate a
   baseline and write `submission.csv`. Its notebook and generated files appear
   as downloads in the chat.

Each run starts a fresh Colab session, checks the assigned GPU, and transfers the
uploaded files declared by the notebook. The app uses Google credentials locally
for the CLI. Successful downloads contain the executed notebook and requested
outputs. The app attempts to release the runtime on completion, failure, or Stop.
A run has a 15-minute limit. T4 access depends on Colab capacity and your account's
quota. New chats start with **Local CPU**.

### Fix and explain code

1. Paste code in a ``` block (or attach a `.py`, `.js`, `.ts`, ... file) and ask, e.g.
   *why doesn't this stop?*
2. Jack calls `fix_code`. OpenCode starts in a fresh, locked-down container (read-only,
   non-root, no API keys) with a **free** OpenCode model, and its progress appears under the step.
3. You get *What was wrong*, the fixed code with `# FIX n:` comments, and one line saying how
   far it was verified: **Ran** (output shown), **Checked only** (syntax/type check, with the
   reason it wasn't run) or **Reviewed only**.

How much is run depends on your wording: *just look at it* / *don't run it* = review only;
*does it work?* / *run it* = check and run (a marked `# TEST:` call is added to function-only
code); anything else = always checked, run when it matters. Code runs in a second throwaway
container: no network, 10 s, 512 MB. Concept questions (*what does `enumerate` do?*) Jack
answers himself, with no containers.

Needs Docker (in the devcontainer) and `make sandbox-images`. Only zero-retention free models
are used; set `OPENCODE_ALLOW_TRAINING_MODELS=1` to also allow free models that may log or
train on what they're sent. Try it without the chat: `uv run python -m baseagent.codeagent file.py`.

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

### See it in action.

- Fixing Python code
- Teaching Chess, get interrupted
- Data Science Capabilities and more

https://github.com/user-attachments/assets/57b80f7d-bdc9-4745-9fb1-bfad1a152d7d


## Repo tree

Still evolving...

