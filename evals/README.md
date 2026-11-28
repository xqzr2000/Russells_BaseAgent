# Evals

Graders for real agent runs, run by hand (they are not part of `make test`). Each
eval case is a task given to Jack in the chat room; its grader checks what came back.
One folder per case: the grader and a README with how to run it and what it checks.

| Case | Task | Folder |
| --- | --- | --- |
| 1 | Card CNN: rank and suit from full-card images (Kaggle `gpiosenka/cards-image-datasetclassification`, Colab T4) | [`card_cnn/`](card_cnn/) |

Planned: `dabstep/` and `spider/` (see [the roadmap](../docs/roadmap.md)).
