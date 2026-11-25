"""inspect_dataset: see what a dataset contains before writing notebook code for it.

The data scientist writes Colab notebooks without running them on Colab first, so
it needs facts about the data, not guesses: folder layout, file types, table
columns, whether path columns point at real files, image sizes, and model files
that ship with the dataset. This works for any Kaggle dataset or the chat's
uploaded data/ folder; nothing here knows about one particular dataset.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path

import pandas as pd

from baseagent.sandbox.kaggle import download_dataset, kaggle_dataset
from baseagent.tools import ToolContext, ToolError, tool

TABLE_SUFFIXES = {".csv", ".tsv", ".parquet"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
MODEL_SUFFIXES = {".h5", ".hdf5", ".keras", ".pt", ".pth", ".ckpt", ".onnx", ".pkl", ".pickle",
                  ".joblib", ".safetensors", ".pb", ".tflite", ".bin"}
MAX_TOP_ENTRIES = 30
MAX_EXPAND = 8  # a folder with this many subfolders or fewer is listed one level deeper
MAX_TABLES = 4
MAX_TABLE_BYTES = 100 * 2**20  # larger tables are read in part
PARTIAL_ROWS = 50_000
IMAGE_SAMPLES = 24
REPORT_CHARS = 7000
DOWNLOAD_TIMEOUT = 900


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def walk(root: Path) -> tuple[list[Path], list[Path]]:
    files, dirs = [], []
    for path in root.rglob("*"):
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        (dirs if path.is_dir() else files).append(path)
    return files, dirs


def suffix_counts(files: list[Path]) -> str:
    counts = Counter(f.suffix.lower() or "(no extension)" for f in files)
    return ", ".join(f"{suffix} {count}" for suffix, count in counts.most_common(4))


def describe_folder(path: Path, files: list[Path], indent: str, depth: int) -> list[str]:
    inside = [f for f in files if path in f.parents]
    subdirs = sorted(d for d in path.iterdir() if d.is_dir() and not d.name.startswith("."))
    line = f"{indent}{path.name}/  {len(inside)} file{'' if len(inside) == 1 else 's'}"
    if inside:
        line += f" ({suffix_counts(inside)})"
    if subdirs:
        line += f" in {len(subdirs)} folders"
    lines = [line]
    if subdirs and depth < 1 and len(subdirs) <= MAX_EXPAND:
        for sub in subdirs:
            lines += describe_folder(sub, inside, indent + "  ", depth + 1)
    elif subdirs:
        names = ", ".join(d.name for d in subdirs[:6])
        lines.append(f"{indent}  e.g. {names}{', …' if len(subdirs) > 6 else ''}")
        direct = [d for d in subdirs if any(f.parent == d for f in inside)]
        if len(direct) == len(subdirs) and all(
                {f.suffix.lower() for f in inside if f.parent == d} <= IMAGE_SUFFIXES for d in subdirs):
            lines.append(f"{indent}  (looks like one folder of images per class: {len(subdirs)} classes)")
    return lines


def describe_tree(root: Path, files: list[Path]) -> list[str]:
    entries = sorted(p for p in root.iterdir() if not p.name.startswith("."))
    lines = []
    for entry in entries[:MAX_TOP_ENTRIES]:
        if entry.is_dir():
            lines += describe_folder(entry, files, "  ", 0)
        else:
            lines.append(f"  {entry.name}  ({human(entry.stat().st_size)})")
    if len(entries) > MAX_TOP_ENTRIES:
        lines.append(f"  … and {len(entries) - MAX_TOP_ENTRIES} more entries")
    top_dirs = [e for e in entries if e.is_dir()]
    names = {d.name: frozenset(s.name for s in d.iterdir() if s.is_dir()) for d in top_dirs}
    shared = [n for n, subs in names.items() if subs and list(names.values()).count(subs) > 1]
    if len(shared) > 1:
        count = len(names[shared[0]])
        lines.append(f"  ({', '.join(f'{n}/' for n in shared)} have the same {count} subfolders)")
    return lines


def read_table(path: Path) -> tuple[pd.DataFrame, bool]:
    partial = path.stat().st_size > MAX_TABLE_BYTES
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
        return (frame.head(PARTIAL_ROWS), True) if partial else (frame, False)
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep, low_memory=False, nrows=PARTIAL_ROWS if partial else None), partial


def is_text(values: pd.Series) -> bool:
    return values.dtype == object or pd.api.types.is_string_dtype(values)


def looks_like_paths(values: pd.Series) -> bool:
    sample = values.dropna().astype(str).head(200)
    if sample.empty:
        return False
    pathish = sample.str.contains("/") | sample.str.lower().str.match(r".+\.[a-z0-9]{2,5}$")
    return bool(pathish.mean() > 0.8)


def check_paths(values: pd.Series, root: Path, table_dir: Path) -> str:
    texts = values.dropna().astype(str)
    found, missing = 0, []
    for text in texts:
        if (root / text).is_file() or (table_dir / text).is_file():
            found += 1
        elif len(missing) < 3:
            missing.append(text)
    line = f"    file paths: {found} of {len(texts)} point to existing files"
    if missing:
        line += "; not files: " + ", ".join(repr(m) for m in missing)
    return line


def describe_table(path: Path, root: Path) -> list[str]:
    try:
        frame, partial = read_table(path)
    except Exception as exc:  # noqa: BLE001 - report the problem and keep probing
        return [f"{path.relative_to(root)}: could not read ({type(exc).__name__}: {str(exc)[:200]})"]
    rows = f"first {len(frame)} rows read" if partial else f"{len(frame)} rows"
    lines = [f"{path.relative_to(root)}: {rows} × {frame.shape[1]} columns"]
    for column in frame.columns[:40]:
        values = frame[column]
        unique = values.nunique(dropna=True)
        line = f"  {column!r}: {values.dtype}, {values.notna().sum()} non-null, {unique} unique"
        if unique <= 20 and unique < max(2, len(frame) // 2):  # a category, not one value per row
            top = values.value_counts(dropna=False).head(12)
            line += "; values: " + ", ".join(f"{value!r} ({count})" for value, count in top.items())
        elif is_text(values):
            examples = values.dropna().astype(str).unique()[:3]
            line += "; e.g. " + ", ".join(repr(str(e)[:60]) for e in examples)
        lines.append(line)
        if is_text(values) and looks_like_paths(values):
            lines.append(check_paths(values, root, path.parent))
    if frame.shape[1] > 40:
        lines.append(f"  … and {frame.shape[1] - 40} more columns")
    return lines


def describe_images(images: list[Path]) -> list[str]:
    from PIL import Image

    step = max(1, len(images) // IMAGE_SAMPLES)
    sizes: Counter[str] = Counter()
    for image in images[::step][:IMAGE_SAMPLES]:
        try:
            with Image.open(image) as opened:
                sizes[f"{opened.width}×{opened.height} {opened.mode}"] += 1
        except Exception:  # noqa: BLE001 - an unreadable image is itself a finding
            sizes["unreadable"] += 1
    sampled = sum(sizes.values())
    summary = ", ".join(f"{size} ({count})" for size, count in sizes.most_common(5))
    return [f"Images: {len(images)} files; sizes in a sample of {sampled}: {summary}"]


def describe_dataset(root: Path, label: str) -> str:
    """A compact, factual report on everything under root."""
    files, _ = walk(root)
    total = sum(f.stat().st_size for f in files)
    lines = [f"Dataset {label}: {len(files)} files, {human(total)}", "", "Layout:"]
    lines += describe_tree(root, files)

    tables = sorted((f for f in files if f.suffix.lower() in TABLE_SUFFIXES),
                    key=lambda f: (len(f.relative_to(root).parts), f.name))
    if tables:
        lines += ["", "Tables:"]
        for table in tables[:MAX_TABLES]:
            lines += describe_table(table, root)
        if len(tables) > MAX_TABLES:
            lines.append(f"… and {len(tables) - MAX_TABLES} more tables")

    images = [f for f in files if f.suffix.lower() in IMAGE_SUFFIXES]
    if images:
        lines += [""] + describe_images(images)

    models = [f for f in files if f.suffix.lower() in MODEL_SUFFIXES]
    if models:
        names = ", ".join(str(m.relative_to(root)) for m in models[:5])
        lines += ["", f"Model files shipped with the dataset: {names}. Don't use them unless the task "
                      "asks for a pretrained model: train your own."]

    report = "\n".join(lines)
    if len(report) > REPORT_CHARS:
        report = report[:REPORT_CHARS] + "\n… (report shortened)"
    return report


def dataset_roots(ctx: ToolContext) -> dict[str, Path]:
    return ctx.state.setdefault("dataset_roots", {})


async def ensure_dataset(handle: str, ctx: ToolContext) -> Path:
    """The local copy of a Kaggle dataset, downloaded once per chat (kagglehub caches it)."""
    roots = dataset_roots(ctx)
    if handle not in roots or not roots[handle].is_dir():
        await ctx.progress(f"Downloading {handle} (kept in a cache after the first time)")
        roots[handle] = await asyncio.wait_for(asyncio.to_thread(download_dataset, handle), DOWNLOAD_TIMEOUT)
    return roots[handle]


@tool(exclusive=True, timeout=DOWNLOAD_TIMEOUT + 120)
async def inspect_dataset(source: str, ctx: ToolContext = None) -> str:
    """Download a dataset here and report what it contains, before writing notebook cells.

    Reports the folder layout, file types, each table's columns (types, unique values,
    whether path columns point to existing files), image sizes and model files shipped
    with the data. Base the notebook's loading code on this report, not on guesses.

    Args:
        source: A public Kaggle dataset URL or owner/dataset handle, or "data" for the files
            uploaded to this chat.
    """
    if source.strip().rstrip("/") == "data":
        if ctx.agent.workspace is None:
            raise ToolError("This chat has no data workspace.")
        root = ctx.agent.workspace.resolve("data")
        if not root.is_dir() or not any(root.iterdir()):
            raise ToolError("No files have been uploaded to this chat.")
        return describe_dataset(root, "data/ (uploaded files)") + (
            "\n\nIn the notebook, read these with relative data/... paths.")
    handle = kaggle_dataset(source)
    root = await ensure_dataset(handle, ctx)
    report = await asyncio.to_thread(describe_dataset, root, handle)
    return report + ("\n\nIn the notebook this folder is `dataset_root` (a pathlib.Path), created by the "
                     "download cell. It can be read-only on Colab: write files to OUTPUT_DIR.")
