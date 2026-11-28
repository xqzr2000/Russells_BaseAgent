# Showcase

How this agent was built, one task at a time. Each task is a real piece of data-science
work given to Jack in the chat room. Each failure on it became a change to the harness,
a skill or an eval check, never a one-off patch for one dataset.

Pick a task you're interested in and open its folder:

| # | Task | What it shows | Status |
| --- | --- | --- | --- |
| 01 | [Card classifier](01-card-cnn/): a small CNN on a Kaggle image dataset, trained on a Colab T4 | Building the harness: dataset probe, smoke run before the GPU, skills, a grader | in progress |
| 02 | Kaggle Playground Series competition (tabular) | Classic DS on a live leaderboard: CV, experiments, submissions | planned (Dec – Jan) |
| 03 | DABStep benchmark | Multi-step analysis over documents and tables, run unattended, scored automatically | planned (Feb – Mar) |
| 04 | Spider 2.0-Lite, SQLite part | Text-to-SQL with schema retrieval | planned (Apr) |

## In each task folder

| File | What's in it |
| --- | --- |
| `TASK.md` | The task as given: data, goal, metric, constraints, and why this task |
| `HARNESS.md` | What we built so the agent could do it: tools, guards, prompts, skills (named, not quoted) and evals, each with the failure it answered and its PR |
| `RESULTS.md` | Each run's scores with their uncertainty, what went wrong, what changed, and what's still open |

[`BUILD_LOG.md`](BUILD_LOG.md) is the dated history of the whole project and links each
task.

## Who did what

Russell chooses the tasks, makes the design decisions, tests every change in the chat
room, reviews and merges every PR. Coding agents (Claude and Codex) write most of the
code; Claude also reviews the runs. Jack's model is never fine-tuned: he improves only
through changes to BaseAgent and through skills.

The skills themselves live in a private repo. These folders name them and say what they
changed, without reproducing them.
