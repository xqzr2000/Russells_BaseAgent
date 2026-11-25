"""Notebook cells for reproducible downloads of public Kaggle datasets."""

from __future__ import annotations

import re
from pathlib import Path

from baseagent.tools import ToolError

KAGGLEHUB_VERSION = "1.0.3"
SLUG = r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+(?:/versions/[0-9]+)?"
KAGGLE_URL = re.compile(rf"https://(?:www\.)?kaggle\.com/datasets/({SLUG})(?=$|[/#?.,\s)\]])",
                        re.IGNORECASE)

DOWNLOAD = """\
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

if os.environ.get("BASEAGENT_DATASET_ROOT"):
    # BaseAgent's smoke run reuses the copy it downloaded to inspect the dataset.
    dataset_root = Path(os.environ["BASEAGENT_DATASET_ROOT"])
else:
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


def download_dataset(handle: str) -> Path:
    """Download (or reuse the cached copy of) a public Kaggle dataset on this machine."""
    try:
        import kagglehub
    except ImportError as exc:
        raise ToolError("kagglehub isn't installed on the agent server. Run uv sync.") from exc
    try:
        return Path(kagglehub.dataset_download(handle))
    except Exception as exc:  # noqa: BLE001 - kagglehub raises many kinds; the message matters
        detail = str(exc).strip().splitlines()[0][:300] if str(exc).strip() else type(exc).__name__
        raise ToolError(f"Could not download the Kaggle dataset {handle}: {detail}. If Kaggle asks for "
                        "access (sign-in, rules or a private dataset), ask the user to resolve it on "
                        "Kaggle. Never ask for or use their Kaggle credentials.") from exc


def kaggle_download_source(handle: str) -> str:
    return DOWNLOAD.format(handle=handle, version=KAGGLEHUB_VERSION, package=f"kagglehub=={KAGGLEHUB_VERSION}")
