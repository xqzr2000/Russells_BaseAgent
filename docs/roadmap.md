# Roadmap

Updated 2026-11-28. The detailed plan lives in
[Russell-Study-ML](https://github.com/xqzr2000/Russell-Study-ML#2-order-to-fall-2027);
this is the BaseAgent part of it.

## Goal

By June 30, 2027, Jack runs at a public URL and can represent Russell to recruiters and
employers: introduce him, answer questions honestly, show how the agent was built, and
pass a message on. The proof of data-science ability is the data scientist's work on
real tasks, each documented in [`showcase/`](../showcase/).

## The plan

| Dates | Task | Status |
| --- | --- | --- |
| Oct – Nov 2026 | BaseAgent core, Jack, chess coach, Colab T4 runs, knowledge skills | done |
| Nov 30 – Dec 13 | Close the card-classifier task: last harness fixes, one confirming run | committed |
| Dec 14 – Jan 24 | One Kaggle Playground Series competition (tabular, live leaderboard) | committed |
| Jan 25 – Mar 28 | **DABStep**, the flagship: a multi-step data-analysis benchmark over documents and tables | committed |
| Mar 29 – May 2 | **Spider 2.0-Lite**, SQLite part: text-to-SQL with table retrieval | flex: shrinks first |
| May 3 – Jun 30 | Deploy, monitor, A/B test two harness versions; Jack's recruiter abilities; public URL | committed |

Russell has about 3 hours a week, so a few tasks done well beat many done fast.

## What gets built along the way

- **For the benchmarks:** a batch runner that runs the agent over many questions without
  supervision, a document-reading tool, a SQLite tool with schema search, answer-format
  checks, and failure reports. Each benchmark gets an `evals/` folder.
- **For the Kaggle competition:** a fixed CV split saved to disk, one idea per experiment
  with its CV score, and a reproducible submission.
- **For deployment:** visitor and owner modes (visitors only talk to Jack), cost limits,
  prompt-injection tests, monitoring, and a measured A/B test.
- **Later:** a Claude Code sandbox for heavier coding, on the existing runner.

## Parked

A chemistry-dataset challenge, more Kaggle competitions, a course-learning mode, Jack in
3D, and the rest of the chess coach's plan ([chess coach](agents/chess-coach.md#not-built-yet)).
