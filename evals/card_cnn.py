"""Grade an executed card-CNN notebook (eval case 1), run by hand.

    uv run python evals/card_cnn.py [path/to/analysis.ipynb] [--outputs DIR] [--image IMAGE]

Without a notebook it grades the newest workspace/*/outputs/baseagent-*/analysis.ipynb.

The task: train a small CNN on gpiosenka/cards-image-datasetclassification that
predicts rank and suit from full-card images, using the dataset's own splits and
the 52 standard cards (jokers excluded and counted). The notebook and files come
back from run_colab_notebook (outputs/baseagent-*/ in the chat's workspace).

Checks (PASS, FAIL or WARN), each tied to a requirement or a skill:
  ran            every code cell ran, none raised
  jokers         the joker count excluded matches the data (125)
  test size      the test set has the expected size after exclusion (260)
  label map      label_map.json has exactly the 52 cards, with consistent parts
                 (the table may sit one level down, e.g. {"label_map": {...}, "ranks": [...]})
  metrics        exact, rank and suit test accuracies are printed, and
                 exact <= rank, suit (parts come from the predicted card: composite-labels)
  splits         no re-splitting of the shipped splits (data-splitting)
  hygiene        no Kaggle credentials, no model files shipped with the dataset
  outputs        card_cnn.pt and label_map.json exist
  inference      predict(image) works in a fresh session from the saved files: the
                 notebook's functions, classes and constants, then the whole cell
                 that defines predict() (e.g. a model loaded once from card_cnn.pt);
                 statements in that cell that fail, such as demo lines, are skipped

Exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import nbformat

RANKS = "ace two three four five six seven eight nine ten jack queen king".split()
SUITS = "clubs diamonds hearts spades".split()
CARDS = {f"{rank} of {suit}" for rank in RANKS for suit in SUITS}
CARD_TEXT = re.compile(rf"\b({'|'.join(RANKS)}) of ({'|'.join(SUITS)})\b", re.IGNORECASE)
METRIC = re.compile(r"\b(exact[- ]?card|exact|rank|suit)\b[^0-9\n]{0,30}?(\d+(?:\.\d+)?)(\s*%)?", re.IGNORECASE)
HANDLE = "gpiosenka/cards-image-datasetclassification"
INFERENCE_TIMEOUT = 180
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Spec:
    jokers: int = 125            # 115 train + 5 valid + 5 test
    test_images: int = 260       # 265 test images minus 5 jokers
    weights: str = "card_cnn.pt"
    label_map: str = "label_map.json"


@dataclass
class Result:
    name: str
    status: str  # PASS, FAIL or WARN
    detail: str


@dataclass
class Grade:
    results: list[Result] = field(default_factory=list)

    def add(self, name: str, ok: bool | None, detail: str) -> None:
        self.results.append(Result(name, "WARN" if ok is None else "PASS" if ok else "FAIL", detail))

    @property
    def failed(self) -> bool:
        return any(r.status == "FAIL" for r in self.results)


def output_text(cell) -> str:
    parts = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            parts.append(out.get("text", ""))
        elif out.get("output_type") in ("execute_result", "display_data"):
            parts.append(str(out.get("data", {}).get("text/plain", "")))
        elif out.get("output_type") == "error":
            parts.append(f"{out.get('ename')}: {out.get('evalue')}")
    return "".join(parts)


def code_cells(notebook) -> list:
    return [c for c in notebook.cells if c.cell_type == "code"]


def authored(notebook) -> list:
    """Cells the agent wrote (marked by BaseAgent); all code cells for other notebooks."""
    marked = [c for c in code_cells(notebook) if c.metadata.get("baseagent_cell")]
    return marked or code_cells(notebook)


def check_ran(notebook, grade: Grade) -> None:
    cells = code_cells(notebook)
    errors = [(i, out) for i, c in enumerate(cells, 1) for out in c.get("outputs", [])
              if out.get("output_type") == "error"]
    unrun = [i for i, c in enumerate(cells, 1) if c.get("execution_count") is None and c.source.strip()]
    if errors:
        index, error = errors[0]
        number = cells[index - 1].metadata.get("baseagent_cell")
        where = f"code cell {index}" + (f" (authored cell {number})" if number else "")
        grade.add("ran", False, f"{where} raised {error.get('ename')}: {str(error.get('evalue'))[:120]}")
    elif unrun:
        grade.add("ran", False, f"{len(unrun)} code cells never ran (first: cell {unrun[0]})")
    else:
        hardware = re.search(r"Executed on Google Colab using (.+?)\.", notebook.cells[0].source)
        where = f" on {hardware[1]}" if hardware else " (not marked as a Colab run)"
        grade.add("ran", True, f"{len(cells)} code cells ran{where}")


def numbers(line: str) -> list[int]:
    return [int(n) for n in re.findall(r"(?<![\d.])\d+(?![\d.])", line)]


JOKER_COUNT = re.compile(r"(\d+)\s+jokers?\b|\bjokers?\b\D{0,20}?(\d+)", re.IGNORECASE)


def check_counts(text: str, cell_texts: list[str], spec: Spec, grade: Grade) -> None:
    joker_lines = [line for line in text.splitlines() if "joker" in line.lower()]
    found = [line for line in joker_lines if spec.jokers in numbers(line) or sum(numbers(line)) == spec.jokers]
    per_split = [t for t in cell_texts  # e.g. "train: 7624 images, 115 jokers" on one line per split
                 if sum(int(a or b) for a, b in JOKER_COUNT.findall(t)) == spec.jokers]
    if found:
        grade.add("jokers", True, found[0].strip()[:120])
    elif per_split:
        grade.add("jokers", True, f"{spec.jokers} across the per-split lines of one cell")
    else:
        grade.add("jokers", False, f"no output line reports {spec.jokers} excluded jokers"
                  + (f" (saw: {joker_lines[0].strip()[:100]})" if joker_lines else ""))
    test_lines = [line for line in text.splitlines() if re.search(r"\btest\b", line, re.IGNORECASE)]
    sized = [line for line in test_lines if spec.test_images in numbers(line)]
    grade.add("test size", True if sized else None,
              sized[0].strip()[:120] if sized else f"no output line shows {spec.test_images} test images")


def table_entries(data) -> list:
    if isinstance(data, dict) and data and all(isinstance(v, int) for v in data.values()):
        return list(data)  # {"ace of clubs": 0, ...}
    return list(data.values()) if isinstance(data, dict) else list(data) if isinstance(data, list) else []


def card_count(data) -> int:
    found = set()
    for entry in table_entries(data):
        strings = [entry] if isinstance(entry, str) else \
            [v for v in entry.values() if isinstance(v, str)] if isinstance(entry, dict) else []
        found |= {s.lower() for s in strings if s.lower() in CARDS}
    return len(found)


def unwrap(data):
    """The card table: the file itself, or the value one level down that names the most cards."""
    if not isinstance(data, dict):
        return data
    nested = [v for v in data.values() if isinstance(v, (dict, list))]
    best = max(nested, key=card_count, default=None)
    return best if best is not None and card_count(best) > card_count(data) else data


def check_label_map(path: Path, grade: Grade) -> None:
    if not path.is_file():
        grade.add("label map", False, f"{path.name} is missing")
        return
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        grade.add("label map", False, f"{path.name} is not valid JSON: {exc}")
        return
    data = unwrap(data)
    entries = table_entries(data)
    names, problems = set(), []
    for entry in entries:
        strings = [entry] if isinstance(entry, str) else [v for v in entry.values() if isinstance(v, str)] \
            if isinstance(entry, dict) else []
        card = next((s.lower() for s in strings if s.lower() in CARDS), None)
        if card is None:
            problems.append(f"no card name in {entry!r}"[:80])
            continue
        names.add(card)
        if isinstance(entry, dict):
            rank = next((v for k, v in entry.items() if "rank" in k.lower()), None)
            suit = next((v for k, v in entry.items() if "suit" in k.lower()), None)
            if rank is not None and suit is not None and f"{rank} of {suit}".lower() != card:
                problems.append(f"{card!r} has parts {rank!r} and {suit!r}")
    missing = CARDS - names
    if problems or missing or len(entries) != 52:
        detail = f"{len(entries)} entries"
        if missing:
            detail += f", missing {len(missing)} cards (e.g. {sorted(missing)[0]})"
        if problems:
            detail += f"; {problems[0]}"
        grade.add("label map", False, detail)
    else:
        grade.add("label map", True, "52 cards, parts consistent")


def check_metrics(text: str, grade: Grade) -> None:
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not re.search(r"\btest\b", line, re.IGNORECASE):
            continue
        for name, number, percent in METRIC.findall(line):
            key = "exact" if name.lower().startswith("exact") else name.lower()
            value = float(number)
            values[key] = value / 100 if percent or value > 1 else value
    missing = [k for k in ("exact", "rank", "suit") if k not in values]
    if missing:
        grade.add("metrics", False, f"no test {', '.join(missing)} accuracy in the outputs")
        return
    shown = ", ".join(f"{k} {values[k]:.3f}" for k in ("exact", "rank", "suit"))
    consistent = values["exact"] <= min(values["rank"], values["suit"]) + 1e-9
    grade.add("metrics", consistent, shown + ("" if consistent else
              " - exact is above rank or suit: parts don't come from the predicted card"))


def check_sources(notebook, grade: Grade) -> None:
    source = "\n".join(c.source for c in authored(notebook))
    resplit = re.findall(r"\b(train_test_split|random_split|GroupShuffleSplit|ShuffleSplit)\b", source)
    grade.add("splits", not resplit, "re-splits the data with " + ", ".join(sorted(set(resplit)))
              if resplit else "uses the shipped train/valid/test splits")
    issues = []
    if re.search(r"kaggle\.json|KAGGLE_(?:KEY|USERNAME)", source):
        issues.append("Kaggle credentials")
    if re.search(r"load_model\(|\.h5\b|keras\.models", source):
        issues.append("a shipped model file")
    grade.add("hygiene", not issues, "uses " + " and ".join(issues) if issues else "no credentials or shipped models")


def check_outputs(outputs: Path, spec: Spec, grade: Grade) -> None:
    missing = [name for name in (spec.weights, spec.label_map) if not (outputs / name).is_file()]
    grade.add("outputs", not missing, f"missing {', '.join(missing)} in {outputs}" if missing
              else f"{spec.weights} ({(outputs / spec.weights).stat().st_size // 1024} KB), {spec.label_map}")


import builtins

DEFINES = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def defined_names(node: ast.stmt) -> set[str]:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {(a.asname or a.name).split(".")[0] for a in node.names}
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
    return {t.id for t in targets if isinstance(t, ast.Name)}


def loads(node: ast.AST | None) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} \
        if node is not None else set()


def parse_cell(source: str) -> list[ast.stmt]:
    from IPython.core.inputtransformer2 import TransformerManager

    try:
        return ast.parse(TransformerManager().transform_cell(source)).body
    except SyntaxError:
        return []


def definitions(cells: list[str], known: set[str]) -> list[str]:
    """The code of the authored cells, without anything that holds data or training state.

    Kept: imports (also those inside try blocks), functions, classes, and UPPER_CASE or
    `device` constants whose value uses only names already defined this way. Dropped:
    everything else (data tables, models, training loops, demo calls).
    """
    known = set(known) | set(dir(builtins))
    chunks = []
    for source in cells:
        kept = []
        for node in parse_cell(source):
            candidates = [node]
            if isinstance(node, ast.Try):  # e.g. try: import torch / except: install it
                candidates = [n for n in node.body if isinstance(n, (ast.Import, ast.ImportFrom))]
            for item in candidates:
                constant = (isinstance(item, (ast.Assign, ast.AnnAssign)) and defined_names(item)
                            and all(n.isupper() or n == "device" for n in defined_names(item))
                            and loads(item.value) <= known)
                if isinstance(item, DEFINES) or constant:
                    kept.append(ast.unparse(item))
                    known |= defined_names(item)
        chunks.append("\n".join(kept))
    return chunks


def defines_predict(source: str) -> bool:
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "predict"
               for n in parse_cell(source))


def statements(source: str) -> list[str]:
    return [ast.unparse(node) for node in parse_cell(source)]


def harness_names(sources: list[str]) -> set[str]:
    names: set[str] = set()
    for source in sources:
        for node in parse_cell(source):
            if isinstance(node, (*DEFINES, ast.Assign, ast.AnnAssign)):
                names |= defined_names(node)
    return names


INFERENCE = r"""\
import contextlib, io, json, sys
from IPython.core.interactiveshell import InteractiveShell
shell = InteractiveShell.instance()
setup, cell = json.loads(sys.argv[1]), json.loads(sys.argv[3])
for chunk in setup:
    chunk = "\n".join(line for line in chunk.splitlines() if not line.lstrip().startswith("%matplotlib"))
    result = shell.run_cell(chunk, silent=True)
    if not result.success:
        error = result.error_in_exec or result.error_before_exec
        print(json.dumps({"error": f"while defining: {type(error).__name__}: {error}"}))
        sys.exit(0)
skipped = []  # statements of the predict() cell that failed, e.g. demo lines
for statement in cell:
    with contextlib.redirect_stdout(io.StringIO()):
        result = shell.run_cell(statement, silent=True)
    if not result.success:
        error = result.error_in_exec or result.error_before_exec
        skipped.append(f"{statement.splitlines()[0][:60]!r}: {type(error).__name__}: {error}")
predict = shell.user_ns.get("predict")
if predict is None:
    print(json.dumps({"error": "the notebook defines no predict() function"}))
    sys.exit(0)
captured = io.StringIO()
try:
    with contextlib.redirect_stdout(captured):
        value = predict(sys.argv[2])
except Exception as exc:
    print(json.dumps({"error": f"{type(exc).__name__}: {exc}", "skipped": skipped}))
else:
    print(json.dumps({"value": repr(value), "stdout": captured.getvalue(), "skipped": skipped}))
"""


def default_image() -> Path | None:
    try:
        import kagglehub

        root = Path(kagglehub.dataset_download(HANDLE))
    except Exception:  # noqa: BLE001 - no image: the check says so
        return None
    images = sorted((root / "test").glob("*/*.jpg"))
    return next((p for p in images if "joker" not in p.parent.name), None)


def check_inference(notebook, outputs: Path, image: Path | None, grade: Grade) -> None:
    if image is None or not image.is_file():
        grade.add("inference", None, "no test image to try predict() on (pass --image)")
        return
    cells = code_cells(notebook)
    # BaseAgent's setup (imports) and notebook-settings (OUTPUT_DIR) cells; not the data download.
    harness = [c.source for c in cells if not c.metadata.get("baseagent_cell")
               and ("pd.set_option(" in c.source or "BASEAGENT_OUTPUT_DIR" in c.source)]
    sources = [c.source for c in authored(notebook)]
    cell = next((s for s in reversed(sources) if defines_predict(s)), None)
    others = [s for s in sources if s is not cell]
    setup = harness + definitions(others, harness_names(harness))
    env = {**os.environ, "BASEAGENT_OUTPUT_DIR": str(outputs), "MPLBACKEND": "Agg"}
    try:
        run = subprocess.run([sys.executable, "-c", INFERENCE, json.dumps(setup), str(image),
                              json.dumps(statements(cell) if cell else [])],
                             capture_output=True, text=True, timeout=INFERENCE_TIMEOUT, env=env, cwd=outputs)
        answer = json.loads(run.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        grade.add("inference", False, f"predict() took longer than {INFERENCE_TIMEOUT} s")
        return
    except (json.JSONDecodeError, IndexError):
        grade.add("inference", False, f"the check itself failed: {(run.stderr or run.stdout)[-200:]}")
        return
    skipped = answer.get("skipped") or []
    if "error" in answer:
        cause = f" (skipped in its cell: {skipped[0][:160]})" if skipped else ""
        grade.add("inference", False, answer["error"][:200] + cause + " - predict() must work from the "
                  "saved files alone: the notebook's functions, classes and constants, plus its own cell")
        return
    said = CARD_TEXT.search(answer["value"]) or CARD_TEXT.search(answer["stdout"])
    true = image.parent.name.lower()
    if said is None:
        grade.add("inference", False, f"predict() returned {answer['value'][:80]} with no 'rank of suit'")
    else:
        card = said.group(0).lower()
        grade.add("inference", True, f"{image.parent.name}/{image.name} -> {card}"
                  + (" (correct)" if card == true else f" (true: {true})" if true in CARDS else "")
                  + (f"; {len(skipped)} statement(s) in its cell skipped" if skipped else ""))


def grade_notebook(path: Path, outputs: Path, image: Path | None, spec: Spec = Spec()) -> Grade:
    notebook = nbformat.read(path, as_version=4)
    text = "\n".join(output_text(c) for c in code_cells(notebook))
    grade = Grade()
    check_ran(notebook, grade)
    check_counts(text, [output_text(c) for c in code_cells(notebook)], spec, grade)
    check_label_map(outputs / spec.label_map, grade)
    check_metrics(text, grade)
    check_sources(notebook, grade)
    check_outputs(outputs, spec, grade)
    check_inference(notebook, outputs, image, grade)
    return grade


def newest_notebook(root: Path = ROOT) -> Path | None:
    found = list((root / "workspace").glob("*/outputs/baseagent-*/analysis.ipynb"))
    return max(found, key=lambda p: p.stat().st_mtime, default=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade an executed card-CNN notebook.")
    parser.add_argument("notebook", type=Path, nargs="?",
                        help="the executed notebook (default: the newest Colab run in workspace/)")
    parser.add_argument("--outputs", type=Path, help="folder with card_cnn.pt and label_map.json "
                                                     "(default: the notebook's folder)")
    parser.add_argument("--image", type=Path, help="a full-card image for predict() "
                                                   "(default: a test image from the cached dataset)")
    parser.add_argument("--jokers", type=int, default=Spec.jokers, help="expected excluded jokers")
    parser.add_argument("--test-images", type=int, default=Spec.test_images, help="expected test images")
    parser.add_argument("--json", action="store_true", help="print the results as JSON")
    args = parser.parse_args(argv)

    if args.notebook is None:
        args.notebook = newest_notebook()
        if args.notebook is None:
            parser.error("no workspace/*/outputs/baseagent-*/analysis.ipynb found; pass the notebook")
        print(f"Grading {args.notebook.relative_to(ROOT)}\n")
    outputs = (args.outputs or args.notebook.parent).resolve()
    image = args.image or default_image()
    grade = grade_notebook(args.notebook, outputs, image, Spec(args.jokers, args.test_images))
    if args.json:
        print(json.dumps([r.__dict__ for r in grade.results], indent=1))
    else:
        width = max(len(r.name) for r in grade.results)
        for r in grade.results:
            print(f"{r.status:<5} {r.name:<{width}}  {r.detail}")
        passed = sum(r.status == "PASS" for r in grade.results)
        print(f"\n{passed}/{len(grade.results)} passed" + (" - FAILED" if grade.failed else ""))
    return 1 if grade.failed else 0


if __name__ == "__main__":
    sys.exit(main())
