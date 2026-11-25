# Evals

Graders for real agent runs, run by hand (they are not part of `make test`). Each
eval case is a task given to Jack in the chat room; the grader checks what came back.

| Case | Task | Grader |
| --- | --- | --- |
| 1 | Card CNN: rank and suit from full-card images (Kaggle `gpiosenka/cards-image-datasetclassification`, Colab T4) | `card_cnn.py` |

## Card CNN

After a Colab run, the chat's workspace has `outputs/baseagent-<id>/` with
`analysis.ipynb`, `card_cnn.pt` and `label_map.json`:

```bash
make eval-card NOTEBOOK=workspace/<chat id>/outputs/baseagent-<id>/analysis.ipynb
```

It prints PASS / FAIL / WARN for each check and exits with 1 if any fails:
the notebook ran; 125 jokers excluded; 260 test images; a 52-card label map with
consistent parts; exact, rank and suit test accuracies with exact ≤ rank, suit;
no re-splitting; no credentials or shipped model files; both files saved; and
`predict()` working in a fresh session from the saved files, with only the
notebook's functions, classes and constants. The image for `predict()` comes
from the dataset in the kagglehub cache (the data scientist downloads it), or
`--image`.

Keep the result with the run: it is the baseline the next harness or skill
change is compared against.
