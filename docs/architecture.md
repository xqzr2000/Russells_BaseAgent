# Architecture

BaseAgent is a small [ReAct](https://arxiv.org/pdf/2210.03629) agent framework in Python
with a web chat room for trying its agents. It grew out of the
[CMU 11-768 AI Agents](https://www.cmu-agents.com/#/assignments) assignment harness.

```
web/ (React chat room) ──HTTP + SSE──> server/ (FastAPI)
                                          │  one Session per chat
                                          ▼
                                 Jack (agents/coordinator.py)
                         delegate │                    │ handoff
                                  ▼                    ▼
                agents/data_science/            agents/chess_coach/
                Jupyter kernel, Colab T4         board, Stockfish
                                  │
                         core/ (BaseAgent loop, LLM, tools, events, skills)
```

## Layers

| Folder | What lives there |
| --- | --- |
| `src/baseagent/core/` | The framework. `agent.py` has `BaseAgent`: the loop (prompt → model → tool calls → observations, until a final answer or the step limit), argument validation, error recovery, compaction of long contexts, usage tracking and logging. `llm.py` holds the model clients, `tools.py` the `@tool` decorator, `events.py` the event stream, `skills.py` the `SKILL.md` loader, `delegation.py` and `handoff.py` the two ways agents work together, `workspace.py` each chat's folder |
| `src/baseagent/agents/` | Specialized agents, each a `BaseAgent` subclass registered with `@register_agent`. Small ones are a module; ones with their own machinery are a package |
| `src/baseagent/server/` | The HTTP API: sessions, the SSE message stream, uploads and downloads, the chess board endpoints. In production (`make serve`) it also serves the built UI |
| `src/baseagent/sandbox/` | A locked-down Docker runner for code (no network, read-only, tight limits), on top of [`packages/agent-sandbox-runtime`](../packages/agent-sandbox-runtime). Kept for the planned Claude Code sandbox |
| `web/` | The chat room (React + Vite). It renders any agent's events, so a new agent needs no UI change |
| `skills/` | Public `SKILL.md` playbooks (for example `general/tutor-explanations`) |
| `knowledge/` | Private skills submodule (`jack-knows-it-all`) for the data scientist; empty for public users |
| `evals/` | Graders for real agent runs, one folder per case, run by hand |
| `showcase/` | Each task the agent was built around, with the harness work and results |
| `tests/` | Offline tests mirroring `src/`: `core/`, `agents/`, `server/`, `sandbox/`, `evals/` |

## How agents work together

- **Delegate.** Jack sends a self-contained task to a specialist (`delegatable = True`).
  The specialist runs its own loop with its own memory; only its final answer and output
  files come back, so Jack's context stays small. Its steps show as progress under the
  `delegate` call.
- **Handoff.** Jack switches the whole chat to a specialist that needs a conversation
  (`handoff_target = True`, the chess coach). Later messages go straight to it until it
  hands back.
- **One event log per chat.** Every agent in a chat writes into the same stream with
  continuous turn numbers; the UI shows who is speaking and any agent's `ui_state()`
  (such as the chess board).

## Skills

A skill is a folder with a `SKILL.md`. Only a one-line catalog entry per skill goes into
the system prompt; the full body is returned when the model calls `invoke_skill`
(progressive disclosure). Skill roots can be required (`skills/`) or optional
(`knowledge/`, skipped when absent). Area folders group skills, and a skill's
"Scenarios it explains" section adds a scenarios line to the catalog, so a question that
only describes a symptom still finds the right skill.

## Design choices

- **Subclass, don't fork the loop.** An agent says only what is different: its prompt,
  tools and hooks. A new behaviour every agent might need becomes a `BaseAgent` hook.
- **Facts come from tools.** The chess coach never invents a move; the data scientist
  reasons from printed numbers, not from charts it can't see.
- **Check before the expensive step.** A Colab draft must pass a CPU smoke run, unchanged,
  before it may run on the T4.
- **Owner and visitor.** Paid compute and code execution are for the owner only; the
  public will only talk to Jack (see [the roadmap](roadmap.md)).

To build your own agent, start with [adding an agent](adding-an-agent.md).
