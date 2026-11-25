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
