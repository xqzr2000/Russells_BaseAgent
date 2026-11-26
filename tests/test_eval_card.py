"""The card-CNN grader (evals/card_cnn.py) on small hand-made notebooks; no torch needed."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import nbformat
import pytest
from PIL import Image

from baseagent.sandbox.colab import HARNESS
from baseagent.sandbox.notebook import SETUP_CODE

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("card_cnn", ROOT / "evals" / "card_cnn.py")
card_cnn = importlib.util.module_from_spec(SPEC)
sys.modules["card_cnn"] = card_cnn  # dataclasses look the module up
SPEC.loader.exec_module(card_cnn)

PREDICT = '''\
import json
from pathlib import Path
LABELS = "label_map.json"

def predict(path, labels=OUTPUT_DIR / LABELS):
    names = json.loads(Path(labels).read_text())
    card = names[str(len(Path(path).parent.name) % 52)]
    return f"{card['rank']} of {card['suit']}", 0.5

print(predict(example_path))'''


def stream(text: str):
    return nbformat.v4.new_output("stream", name="stdout", text=text + "\n")


def build(tmp_path: Path, metrics="test exact-card 0.900 | rank 0.950 | suit 0.920 | n=260",
          predict=PREDICT, extra_code: str = "", error: bool = False) -> tuple[Path, Path, Path]:
    cells = [nbformat.v4.new_markdown_cell("# Card CNN\n\nExecuted on Google Colab using Tesla T4.")]
    for source in (SETUP_CODE, HARNESS):
        cells.append(nbformat.v4.new_code_cell(source, execution_count=len(cells)))
    authored = [
        ("cards = load()\nprint(jokers)" + extra_code, "jokers excluded per split: {'train': 115, 'valid': 5, 'test': 5}"),
        ("print(sizes)", "{'test': 260, 'train': 7509, 'valid': 260}"),
        ("print(scores)", metrics),
        (predict, "('queen of hearts', 0.5)"),
    ]
    for number, (source, text) in enumerate(authored, start=1):
        cell = nbformat.v4.new_code_cell(source, execution_count=len(cells))
        cell.metadata["baseagent_cell"] = number
        cell.outputs = [nbformat.v4.new_output("error", ename="KeyError", evalue="'rank'", traceback=[])
                        if error and number == 2 else stream(text)]
        cells.append(cell)
    notebook = nbformat.v4.new_notebook(cells=cells)
    out = tmp_path / "outputs"
    out.mkdir()
    nbformat.write(notebook, out / "analysis.ipynb")
    cards = sorted(card_cnn.CARDS)
    (out / "label_map.json").write_text(json.dumps(
        {i: {"card": c, "rank": c.split(" of ")[0], "suit": c.split(" of ")[1]} for i, c in enumerate(cards)}))
    (out / "card_cnn.pt").write_bytes(b"weights")
    image = tmp_path / "test" / "queen of hearts" / "1.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (224, 224)).save(image)
    return out / "analysis.ipynb", out, image


def statuses(grade) -> dict[str, str]:
    return {r.name: r.status for r in grade.results}


def test_a_complete_run_passes_every_check(tmp_path):
    notebook, out, image = build(tmp_path)
    grade = card_cnn.grade_notebook(notebook, out, image)
    assert set(statuses(grade).values()) == {"PASS"}, [r.__dict__ for r in grade.results]
    assert "on Tesla T4" in grade.results[0].detail


def test_metrics_must_come_from_the_predicted_card(tmp_path):
    notebook, out, image = build(tmp_path, metrics="Test exact-card accuracy: 95.0% | rank 97% | suit 92%")
    result = next(r for r in card_cnn.grade_notebook(notebook, out, image).results if r.name == "metrics")
    assert result.status == "FAIL" and "exact 0.950, rank 0.970, suit 0.920" in result.detail


def test_predict_may_not_use_notebook_state(tmp_path):
    uses_model = PREDICT.replace("return f\"{card['rank']} of {card['suit']}\", 0.5",
                                 "return model(path)")
    notebook, out, image = build(tmp_path, predict=uses_model)
    result = next(r for r in card_cnn.grade_notebook(notebook, out, image).results if r.name == "inference")
    assert result.status == "FAIL" and "NameError: name 'model' is not defined" in result.detail


def test_a_label_map_with_a_joker_fails(tmp_path):
    notebook, out, image = build(tmp_path)
    data = json.loads((out / "label_map.json").read_text())
    data["52"] = {"card": "joker", "rank": "nan", "suit": "nan"}
    (out / "label_map.json").write_text(json.dumps(data))
    result = next(r for r in card_cnn.grade_notebook(notebook, out, image).results if r.name == "label map")
    assert result.status == "FAIL" and "53 entries" in result.detail


def test_resplitting_and_errors_are_reported(tmp_path):
    notebook, out, image = build(tmp_path, extra_code="\ntrain, test = train_test_split(cards)", error=True)
    grade = statuses(card_cnn.grade_notebook(notebook, out, image))
    assert grade["splits"] == "FAIL" and grade["ran"] == "FAIL"


def test_the_command_line_exit_code(tmp_path, capsys):
    notebook, out, image = build(tmp_path)
    assert card_cnn.main([str(notebook), "--image", str(image)]) == 0
    assert "9/9 passed" in capsys.readouterr().out
    (out / "card_cnn.pt").unlink()
    assert card_cnn.main([str(notebook), "--image", str(image)]) == 1


@pytest.mark.parametrize("line, total", [
    ("train: 7624 images, 115 jokers\nvalid: 265 images, 5 jokers\ntest: 265 images, 5 jokers", 125),
    ("Jokers removed: 125", 125),
])
def test_joker_counts_are_read_in_either_form(line, total):
    grade = card_cnn.Grade()
    card_cnn.check_counts(line, [line], card_cnn.Spec(jokers=total), grade)
    assert grade.results[0].status == "PASS"


LOAD_ONCE = '''\
import json
from pathlib import Path

def load_model_for_inference(weights):
    labels = json.loads((OUTPUT_DIR / "label_map.json").read_text())["label_map"]
    return Path(weights).read_bytes(), labels

model_inf, labels_inf = load_model_for_inference(OUTPUT_DIR / "card_cnn.pt")

def predict(path):
    card = labels_inf[str(len(Path(path).parent.name) % 52)]
    return f"{card['rank']} of {card['suit']}", 0.5

print(predict(test_df.path[0]))'''


def nest_label_map(out: Path) -> None:
    table = json.loads((out / "label_map.json").read_text())
    (out / "label_map.json").write_text(json.dumps({
        "label_map": table, "ranks": card_cnn.RANKS, "suits": card_cnn.SUITS}))


def test_run_1_shapes_pass(tmp_path):
    """Run 1 (2026-11-25): a nested label map and a model loaded once in the predict() cell."""
    notebook, out, image = build(tmp_path, predict=LOAD_ONCE)
    nest_label_map(out)
    results = {r.name: r for r in card_cnn.grade_notebook(notebook, out, image).results}
    assert results["label map"].status == "PASS", results["label map"].detail
    assert results["inference"].status == "PASS", results["inference"].detail
    assert "1 statement(s) in its cell skipped" in results["inference"].detail  # the test_df demo line


def test_a_nested_table_must_still_hold_the_52_cards(tmp_path):
    notebook, out, image = build(tmp_path)
    table = json.loads((out / "label_map.json").read_text())
    del table["0"]
    (out / "label_map.json").write_text(json.dumps({"label_map": table, "ranks": card_cnn.RANKS}))
    result = next(r for r in card_cnn.grade_notebook(notebook, out, image).results if r.name == "label map")
    assert result.status == "FAIL" and "51 entries" in result.detail


def test_a_failed_load_in_the_predict_cell_is_named(tmp_path):
    notebook, out, image = build(tmp_path, predict=LOAD_ONCE)
    nest_label_map(out)
    (out / "card_cnn.pt").unlink()
    result = next(r for r in card_cnn.grade_notebook(notebook, out, image).results if r.name == "inference")
    assert result.status == "FAIL"
    assert "labels_inf" in result.detail and "skipped in its cell" in result.detail
    assert "FileNotFoundError" in result.detail


def test_the_newest_run_is_graded_by_default(tmp_path, monkeypatch, capsys):
    import os
    notebook, out, image = build(tmp_path)
    for chat, age in (("old", 100), ("new", 0)):
        run = tmp_path / "workspace" / chat / "outputs" / "baseagent-1"
        run.mkdir(parents=True)
        for f in out.iterdir():
            (run / f.name).write_bytes(f.read_bytes())
        stamp = (run / "analysis.ipynb").stat().st_mtime - age
        os.utime(run / "analysis.ipynb", (stamp, stamp))
    newest = tmp_path / "workspace/new/outputs/baseagent-1/analysis.ipynb"
    assert card_cnn.newest_notebook(tmp_path) == newest
    monkeypatch.setattr(card_cnn, "ROOT", tmp_path)
    monkeypatch.setattr(card_cnn, "newest_notebook", lambda: newest)
    assert card_cnn.main(["--image", str(image)]) == 0
    assert "Grading workspace/new/outputs/baseagent-1/analysis.ipynb" in capsys.readouterr().out


def test_run_2_shapes_pass(tmp_path):
    """Run 2 (2026-11-26): rank and suit on the line after the test line, a demo line after a
    test image's path, and a training helper whose default is a lowercase setting."""
    metrics = ("Test exact-card accuracy: 0.9308 on 260 examples\n"
               "Rank accuracy: 0.9462, Suit accuracy: 0.9731")
    helper = "\nwarmup = 0.05\ndef lr_at(elapsed, budget, peak, warmup=warmup):\n    return peak"
    demo = PREDICT.replace("print(predict(example_path))",
                           'print("Demo image: /data/test/ace of clubs/1.jpg")\n'
                           'print("Predicted: ace of clubs (rank=ace, suit=clubs), confidence=0.998")')
    notebook, out, image = build(tmp_path, metrics=metrics, extra_code=helper, predict=demo)
    results = {r.name: r for r in card_cnn.grade_notebook(notebook, out, image).results}
    assert results["metrics"].status == "PASS"
    assert results["metrics"].detail == "exact 0.931, rank 0.946, suit 0.973"
    assert results["inference"].status == "PASS", results["inference"].detail


def test_validation_lines_after_a_test_line_are_not_test_results():
    grade = card_cnn.Grade()
    card_cnn.check_metrics(["test exact accuracy 0.90\nvalidation rank accuracy 0.99\nsuit accuracy 0.97"], grade)
    assert grade.results[0].status == "FAIL" and "rank" in grade.results[0].detail


def test_a_def_whose_default_needs_training_state_is_left_out():
    chunks = card_cnn.definitions(["model = train()\ndef f(m=model):\n    return m\ndef g():\n    return 1"], set())
    assert "def g" in chunks[0] and "def f" not in chunks[0]
