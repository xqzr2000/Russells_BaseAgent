"""Notebook cells for reproducible downloads of public Kaggle datasets."""

from __future__ import annotations

import re

from baseagent.tools import ToolError

KAGGLEHUB_VERSION = "1.0.3"
SLUG = r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+(?:/versions/[0-9]+)?"
KAGGLE_URL = re.compile(rf"https://(?:www\.)?kaggle\.com/datasets/({SLUG})(?=$|[/#?.,\s)\]])",
                        re.IGNORECASE)
CARDS_DATASET = "gpiosenka/cards-image-datasetclassification"

CARDS_INSTRUCTIONS = (
    "Use cards_manifest (path, split, label, rank, suit). It contains validated rows from cards.csv "
    "with the original train, valid, and test splits, including joker. Use exact labels. "
    "For a 52-card model, report and exclude joker rows. Train a new checkpoint on the manifest images."
)

DOWNLOAD = """\
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    _kagglehub_version = version("kagglehub")
except PackageNotFoundError:
    _kagglehub_version = None
if _kagglehub_version != {version!r}:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", {package!r}], check=True)

import kagglehub

dataset_root = Path(kagglehub.dataset_download({handle!r}))
print("Kaggle dataset:", {handle!r})
print("Dataset directory:", dataset_root)
"""

CARDS_MANIFEST = """\
from pathlib import Path

import pandas as pd

cards_manifest = pd.read_csv(
    dataset_root / "cards.csv", usecols=["filepaths", "labels", "data set"],
)
cards_manifest = cards_manifest.rename(
    columns={"filepaths": "relative_path", "labels": "label", "data set": "split"},
)
for column in ("relative_path", "label", "split"):
    if cards_manifest[column].isna().any():
        raise ValueError(f"cards.csv has missing values in {column!r}.")
# Version 2 lists a non-image path called output alongside the actual training images.
_image_rows = cards_manifest["relative_path"].str.lower().str.endswith((".jpg", ".jpeg", ".png"))
print(f"Skipped {int((~_image_rows).sum())} non-image manifest rows.")
cards_manifest = cards_manifest.loc[_image_rows].copy()
cards_manifest["label"] = cards_manifest["label"].str.strip().str.lower()
cards_manifest["split"] = cards_manifest["split"].str.strip().str.lower()
_expected_splits = {"train", "valid", "test"}
_ranks = "ace two three four five six seven eight nine ten jack queen king".split()
_suits = "clubs diamonds hearts spades".split()
_expected_labels = {f"{rank} of {suit}" for rank in _ranks for suit in _suits} | {"joker"}

if set(cards_manifest["split"]) != _expected_splits:
    raise ValueError("cards.csv must contain the original train, valid, and test splits.")
for _split, _rows in cards_manifest.groupby("split"):
    _labels = set(_rows["label"])
    if _labels != _expected_labels:
        raise ValueError(f"Unexpected card labels in {_split}: missing={sorted(_expected_labels - _labels)}, "
                         f"unexpected={sorted(_labels - _expected_labels)}")

if cards_manifest["relative_path"].duplicated().any():
    raise ValueError("cards.csv contains duplicate image paths.")

_paths = []
for _row in cards_manifest.itertuples(index=False):
    _relative = Path(_row.relative_path)
    if len(_relative.parts) != 3 or _relative.parts[:2] != (_row.split, _row.label):
        raise ValueError(f"Image path does not match its split and class: {_row.relative_path!r}")
    _image = dataset_root / _relative
    if not _image.is_file():
        raise FileNotFoundError(f"Image listed in cards.csv is missing: {_row.relative_path!r}")
    _paths.append(str(_image))
cards_manifest["path"] = _paths
_parts = cards_manifest["label"].str.partition(" of ")
_is_standard_card = cards_manifest["label"] != "joker"
cards_manifest["rank"] = _parts[0].where(_is_standard_card)
cards_manifest["suit"] = _parts[2].where(_is_standard_card)
cards_manifest = cards_manifest[["path", "split", "label", "rank", "suit"]]

for _split, _rows in cards_manifest.groupby("split"):
    print(f"{_split}: {len(_rows)} images, {int((_rows['label'] == 'joker').sum())} jokers")
"""


def kaggle_dataset(value: str) -> str:
    value = value.strip()
    handle = re.fullmatch(SLUG, value)
    url = KAGGLE_URL.match(value)
    if handle:
        return handle[0].lower()
    if url:
        return url[1].lower()
    raise ToolError("Provide a Kaggle dataset URL or owner/dataset handle.")


def kaggle_dataset_in(message: str) -> str | None:
    handles = {match[1].lower() for match in KAGGLE_URL.finditer(message)}
    return next(iter(handles)) if len(handles) == 1 else None


def is_cards_dataset(handle: str) -> bool:
    return handle.split("/versions/", 1)[0] == CARDS_DATASET


def kaggle_download_source(handle: str) -> str:
    return DOWNLOAD.format(handle=handle, version=KAGGLEHUB_VERSION, package=f"kagglehub=={KAGGLEHUB_VERSION}")
