# Task 01: what we built so the agent could do it

Each change answers a failure seen on this task, but is written for any dataset. PR links
are to this repo; skills are in the private knowledge repo and are named, not quoted.

## Getting a notebook onto a GPU

| Change | Why | PR |
| --- | --- | --- |
| Notebooks run from the top in Colab: a data-files cell, failed cells left out | A downloaded notebook stopped halfway on another machine | [#5](https://github.com/xqzr2000/Russells_BaseAgent/pull/5) |
| Run a notebook on a Colab T4 from the chat, through the Colab CLI | The Codespace has no GPU | [#10](https://github.com/xqzr2000/Russells_BaseAgent/pull/10) |
| Draft the notebook one cell at a time, then run the saved draft; return the draft if the run fails | The first try came back with only an imports cell: the whole CNN notebook in one tool call broke its JSON | [#11](https://github.com/xqzr2000/Russells_BaseAgent/pull/11) |
| A supported Kaggle download cell, added automatically for a dataset URL | The draft called a `kagglehub` class that doesn't exist, and guessed labels from file names instead of the dataset's CSV. (The cards-only label table this PR added was later replaced by the generic probe, #15) | [#12](https://github.com/xqzr2000/Russells_BaseAgent/pull/12) |
| Colab CLI and Google Cloud CLI in the dev container; the sign-in kept across rebuilds | Only the one-time sign-in should be done by hand | [#13](https://github.com/xqzr2000/Russells_BaseAgent/pull/13) |

## Checking before the expensive step

| Change | Why | PR |
| --- | --- | --- |
| `inspect_dataset`: layout, columns, whether path columns point to real files, image sizes | The agent guessed the dataset's layout | [#15](https://github.com/xqzr2000/Russells_BaseAgent/pull/15) |
| `smoke_test_notebook`: the whole draft on the CPU with `SMOKE = True`; Colab only runs a draft that passed, unchanged | Bugs surfaced only after a Colab run | [#15](https://github.com/xqzr2000/Russells_BaseAgent/pull/15) |
| Refuse cells that reassign the harness variables (`SMOKE`, `OUTPUT_DIR`, `dataset_root`) | Run 1: a fallback pointed `OUTPUT_DIR` at Colab's path and broke the smoke run | [#19](https://github.com/xqzr2000/Russells_BaseAgent/pull/19) |
| Smoke runs use the full example table and check paths; the subset is taken only for decoding and training | Run 1: a bad row the probe had reported was outside the smoke subset and broke the Colab run | [#19](https://github.com/xqzr2000/Russells_BaseAgent/pull/19) |

## Knowing what to do

| Change | Why | PR |
| --- | --- | --- |
| Skill loader: a private submodule, areas, a scenarios line per skill | Lessons from one task should carry over to the next | [#16](https://github.com/xqzr2000/Russells_BaseAgent/pull/16) |
| Skills from this task: `data-splitting`, `composite-labels`, `image-classification-baseline` (round 2: GPU batch augmentation, a learning-rate schedule on the time budget, one 52-way head as the baseline) | Run 1: slow input pipeline, noisy validation, two heads chosen without reading the skill | knowledge repo |
| Load every skill that fits, before the first cell | Run 1 stopped after loading one skill | [#21](https://github.com/xqzr2000/Russells_BaseAgent/pull/21) |

## Reporting and grading

| Change | Why | PR |
| --- | --- | --- |
| Jack restates a delegated result: test metrics with the test size, caveats, downloads | Run 1: Jack's reply had no numbers ("see above") | [#20](https://github.com/xqzr2000/Russells_BaseAgent/pull/20) |
| Eval case 1: a 9-check grader for an executed notebook, including `predict()` from the saved files in a fresh session | "It ran" isn't the same as "it's right" | [#17](https://github.com/xqzr2000/Russells_BaseAgent/pull/17), [#18](https://github.com/xqzr2000/Russells_BaseAgent/pull/18), [#22](https://github.com/xqzr2000/Russells_BaseAgent/pull/22) |

The grader was fixed three times. Each time, the FAIL was the grader's own gap, not the
agent's mistake. Checking the checker is part of the work.
