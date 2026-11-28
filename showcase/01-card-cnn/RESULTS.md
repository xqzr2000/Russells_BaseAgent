# Task 01: results

Status: **in progress.** One more round of harness fixes and one confirming run, then the
task closes.

## Runs

| Test set (260 cards) | Run 1 (2026-11-25) | Run 2 (2026-11-26) |
| --- | --- | --- |
| Exact card | 92.3% | 93.1% |
| Rank | 93.5% | 94.6% |
| Suit | 98.1% | 97.3% |
| Best validation exact | 95.0% (epoch 21 of 25) | 95.4% (epoch 58 of 70) |
| Epoch time | ~14 s | 4.5–4.9 s |
| Smoke runs / Colab runs | 5 / 3 (1 failed on data) | 3 / 2 (none failed on data) |
| Grader | 9/9 after grader fixes | 9/9 after grader fixes |

**The two runs score the same.** With 260 test cards, each accuracy is good to about ±3
points (95%), so 92.3% and 93.1% can't be told apart. What changed is how the agent got
there: 2.8× the epochs in the same time budget, a steadier validation curve, every
fitting skill loaded, no Colab failure from the data, and Jack reporting the numbers.
Validation levelled off at about 0.93–0.95 from epoch ~40, so this model at 128 px is now
limited by its own size or the resolution, not by training time. Accuracy is not the
next lever for this task; the harness is.

## What went wrong, and what changed

| Run | Problem | Change |
| --- | --- | --- |
| 1 | A bad row the probe had reported broke the first Colab run | Full-table smoke rule ([#19](https://github.com/xqzr2000/Russells_BaseAgent/pull/19)) |
| 1 | A fallback for `OUTPUT_DIR` pointed at Colab's path | Reassigning harness variables is refused ([#19](https://github.com/xqzr2000/Russells_BaseAgent/pull/19)); in run 2 it caught one, fixed on the next try |
| 1 | Only one skill loaded; two heads chosen without reading `composite-labels` | Load every fitting skill first ([#21](https://github.com/xqzr2000/Russells_BaseAgent/pull/21)) |
| 1 | ~14 s epochs from per-image conversion; loss still falling at the end | GPU batch augmentation and a time-budget LR schedule in the skill |
| 1 | Jack gave no numbers | Jack restates results ([#20](https://github.com/xqzr2000/Russells_BaseAgent/pull/20)) |
| 2 | Smoke runs 1–2: predictions table built from every test path while the tensors held the smoke subset | Caught by the smoke run, as designed |

## Still open (round 3)

1. Files written to `OUTPUT_DIR` but not declared as outputs never came back, and Jack
   still listed them as downloads. The harness should report the difference.
2. Skills' "Outputs" sections were ignored: no mistake breakdown, no interval on the test
   metrics.
3. The notebook title was chosen before the design and no longer matches the model.
4. A dropped browser connection cancels a long turn. Only an explicit Stop should.

## What I learned

*Russell's own words, written when the task closes.*
