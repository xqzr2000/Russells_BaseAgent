# Build log

Newest first. Task entries link their folder; PR numbers link the code.

## 2026-11-28 · New plan, reorganized repo

- Plan reset: one Kaggle Playground competition, then DABStep as the flagship, Spider
  2.0-Lite as the flex item, then deploy, monitor and A/B test. See
  [the roadmap](../docs/roadmap.md).
- Removed the OpenCode code agent; its locked-down runner stays in
  `src/baseagent/sandbox/` for a future Claude Code sandbox.
- Reorganized the repo: `core/` for the framework, one package per agent, tests mirroring
  `src/`, one folder per eval, docs by topic, and this showcase
  ([#23](https://github.com/xqzr2000/Russells_BaseAgent/pull/23),
  [#24](https://github.com/xqzr2000/Russells_BaseAgent/pull/24)).

## 2026-11-24 – 11-26 · Task 01: card classifier

Two graded runs on Colab T4 and two rounds of harness changes
([#11](https://github.com/xqzr2000/Russells_BaseAgent/pull/11)–[#22](https://github.com/xqzr2000/Russells_BaseAgent/pull/22)):
dataset probe, smoke run before Colab, private skills, a 9-check grader. Same accuracy
both runs (93% exact, ±3 points), much better process. →
[01-card-cnn](01-card-cnn/)

## 2026-11-23 · Stockfish and Colab T4 runs

- The chess coach plays and hints with Stockfish, and hands back to Jack on request
  ([#8](https://github.com/xqzr2000/Russells_BaseAgent/pull/8),
  [#9](https://github.com/xqzr2000/Russells_BaseAgent/pull/9)).
- The data scientist can run a notebook on a Colab T4 from the chat
  ([#10](https://github.com/xqzr2000/Russells_BaseAgent/pull/10)).

## 2026-11-22 · New chat room look

The meadow UI was replaced with a flat dark workspace and Jack's portrait
([#6](https://github.com/xqzr2000/Russells_BaseAgent/pull/6),
[#7](https://github.com/xqzr2000/Russells_BaseAgent/pull/7)).

## 2026-11-16 · Notebooks that run anywhere

Downloaded notebooks run from the top in Google Colab
([#5](https://github.com/xqzr2000/Russells_BaseAgent/pull/5)).

## 2026-10-22 – 10-24 · Code agent (since removed)

Jack's `fix_code`: OpenCode with a free model fixed and explained pasted code in a
read-only container, and ran it in a second container with no network
([#4](https://github.com/xqzr2000/Russells_BaseAgent/pull/4)). Removed on 2026-11-28 to
make room for a Claude Code sandbox; the runner was kept.

## 2026-10-15 – 10-17 · Chess coach

A handoff agent that teaches beside a board: the board above the chat, board tools for
the coach, then a beginner pass after Russell tried it as someone who has never played
(Black replies by itself, rewind, plain-English narration)
([#3](https://github.com/xqzr2000/Russells_BaseAgent/pull/3)). →
[chess coach](../docs/agents/chess-coach.md)

## 2026-10-11 · Meadow chat room

A meadow background with Jack, the dog the coordinator is named after
([#2](https://github.com/xqzr2000/Russells_BaseAgent/pull/2)). Replaced on 2026-11-22.

## 2026-10-05 – 10-10 · BaseAgent core and the first specialist

The ReAct core, carried over and generalized from the CMU 11-768 Assignment 1 harness,
with a web chat room. Then the data scientist (EDA in a Jupyter kernel, returned as an
`.ipynb`) and the coordinator that delegates to it
([#1](https://github.com/xqzr2000/Russells_BaseAgent/pull/1)).
