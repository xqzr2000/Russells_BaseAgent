# Task 01: card classifier

## The task

Given in the chat room, with **Settings → Notebook compute → Colab T4**:

- **Data:** the public Kaggle dataset
  [`gpiosenka/cards-image-datasetclassification`](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification),
  photos of single playing cards with shipped train, validation and test splits and a CSV
  that lists every image.
- **Goal:** train a small CNN from scratch that reads the card from the image, and report
  the exact card, the rank and the suit accuracy on the test set.
- **Deliver:** the executed notebook, the model weights (`card_cnn.pt`), the label map
  (`label_map.json`), and a `predict()` that works from those saved files in a fresh
  session.
- **Constraints:** a fixed training-time budget on one Colab T4; the shipped splits are
  used as they are; jokers are not a rank-and-suit card, so they are excluded and counted.

## Why this task

It was never about the accuracy. A small image task touches every part of a real run: a
dataset the agent has never seen, a GPU it can't reach from the Codespace, files that
must come back, and a result someone has to check. It was the workout for the data
scientist's harness before any competition work.

The label is also a small trap: a card is a rank *and* a suit. Whether the model has one
52-way head or two heads, and how rank and suit are read back, is a real design choice.
It became the `composite-labels` skill.

## Who ran it

- **Agent:** the data scientist (gpt-5-mini), delegated by Jack.
- **Compute:** drafts and smoke runs on the Codespace CPU; the real run on a Colab T4
  through the Colab CLI.
- **Russell:** gave the task, watched the runs, and decided each round of changes.
- **Claude:** reviewed each run and drafted the harness changes and skills for Russell's
  review.
- **Grading:** [`evals/card_cnn/`](../../evals/card_cnn/) (`make eval-card`).
