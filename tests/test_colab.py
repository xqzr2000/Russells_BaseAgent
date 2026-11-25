"""Exercise the Colab runner with a local test CLI. No Google credentials needed."""

from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
from fastapi.testclient import TestClient

from baseagent import AgentSettings, ScriptedLLM
from baseagent.sandbox import probe
from baseagent.sandbox.colab import (
    fingerprint,
    run_colab_notebook,
    save_colab_cell,
    smoke_test_notebook,
    start_colab_notebook,
)
from baseagent.sandbox.kaggle import kaggle_dataset
from baseagent.sandbox.probe import describe_dataset, inspect_dataset
from baseagent.server.app import create_app
from baseagent.tools import ToolContext, ToolError
from baseagent.workspace import Workspace

FAKE_CLI = """
import json
import sys
import time
from pathlib import Path

import nbformat

mode = ROOT.joinpath("mode").read_text()
auth, provider, command, option, name, *args = sys.argv[1:]
assert (auth, provider, option) == ("--auth", "adc", "-s")
with ROOT.joinpath("calls").open("a") as log:
    log.write(json.dumps([command, name, *args]) + "\\n")
if command == "new":
    assert args == ["--gpu", "T4"]
elif command == "exec" and args[1] == "verify.py":
    assert "nvidia-smi" in Path(args[1]).read_text()
    print("BASEAGENT_GPU=" + ("NVIDIA L4" if mode == "wrong_gpu" else "Tesla T4"))
elif command == "upload":
    assert args[1] == "/content/data/train.csv" and Path(args[0]).read_text() == "x,y\\n1,0\\n2,1\\n"
elif command == "exec":
    if mode == "hold":
        time.sleep(60)
    notebook = nbformat.read(args[1], as_version=4)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert "data/train.csv" in cells[1].source
    training = next(cell for cell in cells if cell.metadata.get("baseagent_cell") == 1)
    if mode == "error":
        training.outputs = [nbformat.v4.new_output(
            "error", ename="ValueError", evalue="test training failure", traceback=[],
        )]
    else:
        training.outputs = [nbformat.v4.new_output("stream", name="stdout", text="Validation score: 0.75\\n")]
    cells[-1].outputs = [nbformat.v4.new_output(
        "stream", name="stdout", text="BASEAGENT_NOTEBOOK_COMPLETE\\n",
    )]
    nbformat.write(notebook, "analysis_output.ipynb")
elif command == "download":
    assert args[0] == "/content/submission.csv"
    Path(args[1]).write_text("id,target\\n1,0\\n")
elif command == "stop":
    assert not args
"""


@pytest.fixture
def fake_colab(tmp_path, monkeypatch):
    fake = tmp_path / "colab"
    fake.write_text(f"#!{sys.executable}\nfrom pathlib import Path\nROOT = Path({str(tmp_path)!r})\n" + FAKE_CLI)
    fake.chmod(0o700)
    (tmp_path / "mode").write_text("ok")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    return tmp_path


def calls(directory: Path) -> list[list[str]]:
    path = directory / "calls"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def collect_sse(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def tool_call(name: str, arguments: dict) -> dict:
    return {
        "content": "",
        "tool_calls": [{
            "id": name, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }],
    }


def malformed_cell():
    message = tool_call("save_colab_cell", {})
    message["tool_calls"][0]["function"]["arguments"] = '{"number": 2, "code": "print('
    return message


TRAINING_CELL = {
    "number": 1,
    "note": "## Baseline",
    "code": 'df = pd.read_csv("data/train.csv")\nprint("Validation score: 0.75")',
}
ARGUMENTS = {"inputs": ["data/train.csv"], "outputs": ["submission.csv"]}
EXPORT_CELL = {"number": 2, "note": "## Export", "code": 'df.to_csv(OUTPUT_DIR / "submission.csv", index=False)'}
CARDS_DATASET = "gpiosenka/cards-image-datasetclassification"


def stage_draft(ctx, smoke_passed=True, arguments=ARGUMENTS):
    """A one-cell draft; marked as smoke-tested unless smoke_passed is False."""
    start_colab_notebook.handler("Test notebook", ctx=ctx)
    outcome = save_colab_cell.handler(**TRAINING_CELL, ctx=ctx)
    assert outcome.attachments[0].title == "colab_draft.ipynb"
    draft = ctx.state["colab_draft"]
    if smoke_passed:
        draft.smoke_passed = fingerprint(draft, arguments["inputs"], arguments["outputs"])
    return draft.path


def test_colab_notebook_settings_and_downloads_through_jack(fake_colab, tmp_path):
    llm = ScriptedLLM([
        tool_call("delegate", {"agent": "data_science", "task": "Train on data/train.csv and return a notebook."}),
        tool_call("start_colab_notebook", {"title": "GPU baseline"}),
        tool_call("save_colab_cell", TRAINING_CELL),
        malformed_cell(),
        tool_call("save_colab_cell", EXPORT_CELL),
        tool_call("run_colab_notebook", ARGUMENTS),  # refused: no smoke run yet
        tool_call("smoke_test_notebook", ARGUMENTS),
        tool_call("run_colab_notebook", ARGUMENTS),
        {"content": "Validation score: 0.75."},
        {"content": "The Colab notebook and submission are ready."},
    ])
    app = create_app(llm=llm, log_dir=tmp_path / "runs", workspace_dir=tmp_path / "workspaces")
    with TestClient(app) as client:
        session = client.post("/api/sessions", json={"agent": "coordinator"}).json()
        assert session["settings"]["notebook_runtime"] == "local"
        route = f"/api/sessions/{session['id']}"
        settings = client.patch(f"{route}/settings", json={"notebook_runtime": "colab-t4"})
        assert settings.json()["notebook_runtime"] == "colab-t4"
        assert client.put(f"{route}/files/train.csv", content=b"x,y\n1,0\n2,1\n").status_code == 200

        reply = client.post(f"{route}/messages", json={"content": "Train on data/train.csv."})
        assert reply.status_code == 200
        events = collect_sse(reply.text)
        result = next(event for event in events if event["type"] == "tool_end" and event["name"] == "delegate")
        assert not result["is_error"]
        artifacts = {item["title"]: item["url"] for item in result["attachments"]}
        assert set(artifacts) == {"analysis.ipynb", "submission.csv"}
        notebook = nbformat.reads(client.get(artifacts["analysis.ipynb"]).text, as_version=4)
        assert "Tesla T4" in notebook.cells[0].source
        training = next(cell for cell in notebook.cells if cell.metadata.get("baseagent_cell") == 1)
        assert training.outputs[0].text == "Validation score: 0.75\n"
        assert notebook.cells[-1].source == EXPORT_CELL["code"]
        assert client.get(artifacts["submission.csv"]).text == "id,target\n1,0\n"
        assert any(event["type"] == "tool_progress" and "Tesla T4" in event["message"] for event in events)

    actions = calls(fake_colab)
    assert [action[0] for action in actions] == ["new", "exec", "upload", "exec", "download", "stop"]
    assert len({action[1] for action in actions}) == 1
    log = json.loads(next((tmp_path / "runs").glob("*.data_science.json")).read_text())
    assert any("Run smoke_test_notebook first" in e.get("content", "") for e in log["events"])
    assert any("Smoke run passed" in e.get("content", "") for e in log["events"])


def test_failed_cell_arguments_leave_a_real_draft_through_jack(fake_colab, tmp_path):
    llm = ScriptedLLM([
        tool_call("delegate", {"agent": "data_science", "task": "Train a CNN and save label_map.json."}),
        tool_call("start_colab_notebook", {"title": "Full-card baseline"}),
        tool_call("save_colab_cell", TRAINING_CELL),
        malformed_cell(),
        {"content": "The draft has one saved cell. Training has not run."},
        {"content": "Colab draft saved."},
    ])
    run_dir = tmp_path / "runs"
    workspaces = tmp_path / "workspaces"
    app = create_app(llm=llm, log_dir=run_dir, workspace_dir=workspaces)
    with TestClient(app) as client:
        session = client.post("/api/sessions", json={
            "agent": "coordinator",
            "settings": {"notebook_runtime": "colab-t4"},
        }).json()
        sid = session["id"]
        response = client.post(f"/api/sessions/{sid}/messages", json={"content": "Train a CNN."})
        assert response.status_code == 200
        result = next(e for e in collect_sse(response.text) if e["type"] == "tool_end" and e["name"] == "delegate")
        assert [file["title"] for file in result["attachments"]] == ["colab_draft.ipynb"]
        notebook = nbformat.reads(client.get(result["attachments"][0]["url"]).text, as_version=4)
        assert notebook.cells[-1].source == TRAINING_CELL["code"]
        assert notebook.cells[-1].execution_count is None
        assert not (workspaces / sid / "label_map_eda.ipynb").exists()

    log = json.loads((run_dir / f"{sid}.data_science.json").read_text())
    assert any("Arguments are not valid JSON" in e.get("content", "") for e in log["events"])
    assert not calls(fake_colab)


def tool_context(tmp_path, runtime="colab-t4"):
    workspace = Workspace(tmp_path / "workspace")
    (workspace.ensure() / "data").mkdir()
    (workspace.root / "data" / "train.csv").write_text("x,y\n1,0\n2,1\n")

    async def emit(_):
        pass

    agent = SimpleNamespace(settings=AgentSettings(notebook_runtime=runtime),
                            workspace=workspace, state={}, emit=emit)
    return ToolContext(agent, "call")


@pytest.mark.asyncio
async def test_local_notebook_compute_cannot_start_colab(fake_colab, tmp_path):
    ctx = tool_context(tmp_path, "local")
    with pytest.raises(ToolError, match="Colab is off"):
        await run_colab_notebook.handler(**ARGUMENTS, ctx=ctx)
    assert not calls(fake_colab)


@pytest.mark.asyncio
async def test_colab_rejects_paths_outside_uploaded_data(fake_colab, tmp_path):
    ctx = tool_context(tmp_path)
    arguments = {**ARGUMENTS, "inputs": ["data/../train.csv"]}
    stage_draft(ctx, arguments=arguments)
    with pytest.raises(ToolError, match="uploaded files"):
        await run_colab_notebook.handler(**arguments, ctx=ctx)
    assert not calls(fake_colab)


@pytest.mark.asyncio
async def test_colab_checks_the_actual_gpu_before_uploading_data(fake_colab, tmp_path):
    (fake_colab / "mode").write_text("wrong_gpu")
    ctx = tool_context(tmp_path)
    stage_draft(ctx)
    with pytest.raises(ToolError, match="did not report a T4"):
        await run_colab_notebook.handler(**ARGUMENTS, ctx=ctx)
    assert [action[0] for action in calls(fake_colab)] == ["new", "exec", "stop"]


@pytest.mark.asyncio
async def test_colab_notebook_error_releases_runtime_and_keeps_draft(fake_colab, tmp_path):
    (fake_colab / "mode").write_text("error")
    ctx = tool_context(tmp_path)
    draft = stage_draft(ctx)
    with pytest.raises(ToolError, match=r"Colab cell 4 \(authored cell 1\) failed: ValueError"):
        await run_colab_notebook.handler(**ARGUMENTS, ctx=ctx)
    assert calls(fake_colab)[-1][0] == "stop"
    assert [item.title for item in ctx.state["colab_outputs"]] == ["colab_draft.ipynb"]
    notebook = nbformat.read(draft, as_version=4)
    assert "have not run" in notebook.cells[0].source
    assert notebook.cells[-1].source == TRAINING_CELL["code"]
    assert notebook.cells[-1].execution_count is None
    assert not list(draft.parent.parent.glob("baseagent-*"))


@pytest.mark.asyncio
async def test_stopping_colab_notebook_releases_runtime(fake_colab, tmp_path):
    (fake_colab / "mode").write_text("hold")
    ctx = tool_context(tmp_path)
    draft = stage_draft(ctx)
    task = asyncio.create_task(run_colab_notebook.handler(**ARGUMENTS, ctx=ctx))
    try:
        async with asyncio.timeout(10):
            while not any("analysis.ipynb" in action for action in calls(fake_colab)):
                await asyncio.sleep(0.02)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert calls(fake_colab)[-1][0] == "stop"
    assert draft.is_file()
    assert not list(draft.parent.parent.glob("baseagent-*"))



def test_draft_cells_can_be_replaced_without_running_colab(fake_colab, tmp_path):
    ctx = tool_context(tmp_path)
    draft = stage_draft(ctx)
    save_colab_cell.handler(2, "print('evaluation')", ctx=ctx)
    with pytest.raises(ToolError, match="number from 1 to 3"):
        save_colab_cell.handler(4, "print('gap')", ctx=ctx)
    save_colab_cell.handler(1, "print('revised training')", ctx=ctx)

    notebook = nbformat.read(draft, as_version=4)
    authored = [cell for cell in notebook.cells if cell.metadata.get("baseagent_cell")]
    assert [cell.source for cell in authored] == ["print('revised training')", "print('evaluation')"]
    assert not calls(fake_colab)
    assert "cells" not in run_colab_notebook.schema()["function"]["parameters"]["properties"]



def test_kaggle_draft_uses_supported_bootstrap_and_rejects_broken_import(fake_colab, tmp_path):
    ctx = tool_context(tmp_path)
    url = f"https://www.kaggle.com/datasets/{CARDS_DATASET}/data"
    result = start_colab_notebook.handler("Full-card classifier", kaggle_source=url, ctx=ctx)
    assert "dataset_root" in result and "inspect_dataset" in result and "cards_manifest" not in result
    with pytest.raises(ToolError, match="no KaggleHub class"):
        save_colab_cell.handler(1, "from kagglehub import KaggleHub", ctx=ctx)
    with pytest.raises(ToolError, match="invalid Python at line 1"):
        save_colab_cell.handler(1, "def bad(", ctx=ctx)

    save_colab_cell.handler(1, '%matplotlib inline\nprint(dataset_root)', ctx=ctx)
    draft = ctx.state["colab_draft"]
    notebook = nbformat.read(draft.path, as_version=4)
    sources = [c.source for c in notebook.cells if c.cell_type == "code"]
    assert len(sources) == 4  # setup, notebook settings, download, the authored cell
    assert "OUTPUT_DIR" in sources[1] and "SMOKE" in sources[1]
    assert f"kagglehub.dataset_download({CARDS_DATASET!r})" in sources[2]
    assert not any("cards.csv" in source for source in sources)
    assert "print(dataset_root)" in sources[-1]
    assert not calls(fake_colab)


def test_public_dataset_url_survives_delegation_into_the_draft(fake_colab, tmp_path):
    url = f"https://www.kaggle.com/datasets/{CARDS_DATASET}/data"
    llm = ScriptedLLM([
        tool_call("delegate", {"agent": "data_science", "task": f"Classify full images from {url}."}),
        tool_call("start_colab_notebook", {"title": "Cards"}),
        tool_call("save_colab_cell", {"number": 1, "code": "print(dataset_root)"}),
        {"content": "Draft created."},
        {"content": "The Colab draft is ready."},
    ])
    app = create_app(llm=llm, log_dir=tmp_path / "runs", workspace_dir=tmp_path / "workspaces")
    with TestClient(app) as client:
        session = client.post("/api/sessions", json={
            "agent": "coordinator", "settings": {"notebook_runtime": "colab-t4"},
        }).json()
        reply = client.post(f"/api/sessions/{session['id']}/messages", json={"content": f"Use {url}."})
        result = next(e for e in collect_sse(reply.text) if e["type"] == "tool_end" and e["name"] == "delegate")
        notebook = nbformat.reads(client.get(result["attachments"][0]["url"]).text, as_version=4)
        assert any("kagglehub.dataset_download" in c.source for c in notebook.cells)
    assert not calls(fake_colab)


@pytest.mark.parametrize("value", [
    "https://example.com/datasets/x/y",
    "https://www.kaggle.com/competitions/some-contest",
    "owner/../dataset",
])
def test_kaggle_only_accepts_dataset_urls_or_handles(value):
    with pytest.raises(ToolError, match="dataset URL"):
        kaggle_dataset(value)


@pytest.mark.parametrize("code", [
    "!kaggle datasets download x -p /content  # needs ~/.kaggle/kaggle.json",
    "os.environ['KAGGLE_KEY'] = 'abc'",
    "Path('/root/.kaggle/').mkdir()",
])
def test_cells_cannot_use_kaggle_credentials(fake_colab, tmp_path, code):
    ctx = tool_context(tmp_path)
    start_colab_notebook.handler("Cards", kaggle_source=CARDS_DATASET, ctx=ctx)
    with pytest.raises(ToolError, match="Kaggle credentials"):
        save_colab_cell.handler(1, code, ctx=ctx)


def card_like_dataset(root: Path) -> Path:
    """A small copy of the playing-card dataset's layout, including its known oddities."""
    from PIL import Image

    labels = ["ace of clubs", "two of hearts", "joker"]
    rows = []
    for split, count in (("train", 3), ("valid", 1), ("test", 1)):
        for index, label in enumerate(labels):
            for number in range(count):
                relative = Path(split) / label / f"{number + 1:03}.jpg"
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (224, 224), (index * 80, 40, 40)).save(root / relative)
                rows.append([index, relative.as_posix(), label, label.split()[0], split])
    rows.append([0, "train/ace of clubs/output", "ace of clubs", "ace", "train"])  # not an image
    with (root / "cards.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class index", "filepaths", "labels", "card type", "data set"])
        writer.writerows(rows)
    (root / "53cards-53-(200 X 200)-100.00.h5").write_bytes(b"\x89HDF")
    return root


def test_dataset_report_states_the_facts_a_notebook_needs(tmp_path):
    report = describe_dataset(card_like_dataset(tmp_path / "cards"), CARDS_DATASET)
    assert "cards.csv: 16 rows × 5 columns" in report
    assert "'filepaths': str, 16 non-null, 16 unique; e.g. 'train/ace of clubs/001.jpg'" in report
    assert "'data set': str, 16 non-null, 3 unique; values: 'train' (10), 'valid' (3), 'test' (3)" in report
    assert "file paths: 15 of 16 point to existing files; not files: 'train/ace of clubs/output'" in report
    assert "train/  9 files (.jpg 9) in 3 folders" in report and "joker/  1 file (.jpg 1)" in report
    assert "(test/, train/, valid/ have the same 3 subfolders)" in report
    assert "224×224 RGB" in report
    assert "53cards-53-(200 X 200)-100.00.h5" in report and "train your own" in report


@pytest.mark.asyncio
async def test_inspect_dataset_downloads_once_per_chat(tmp_path, monkeypatch):
    root = card_like_dataset(tmp_path / "cards")
    downloads = []
    monkeypatch.setattr(probe, "download_dataset", lambda handle: downloads.append(handle) or root)
    ctx = tool_context(tmp_path)
    first = await inspect_dataset.handler(f"https://www.kaggle.com/datasets/{CARDS_DATASET}/data", ctx=ctx)
    await inspect_dataset.handler(CARDS_DATASET, ctx=ctx)
    assert downloads == [CARDS_DATASET]
    assert "dataset_root" in first and "cards.csv" in first
    uploaded = await inspect_dataset.handler("data", ctx=ctx)
    assert "train.csv: 2 rows × 2 columns" in uploaded


SMOKE_ARGUMENTS = {"inputs": ["data/train.csv"], "outputs": ["model.txt"]}


def smoke_draft(ctx, *cells: str, kaggle_source: str | None = None):
    start_colab_notebook.handler("Smoke test", kaggle_source=kaggle_source, ctx=ctx)
    for number, code in enumerate(cells, start=1):
        save_colab_cell.handler(number, code, note=f"## Step {number}", ctx=ctx)
    return ctx.state["colab_draft"]


@pytest.mark.asyncio
async def test_smoke_run_passes_and_unlocks_colab_until_the_draft_changes(fake_colab, tmp_path):
    ctx = tool_context(tmp_path)
    draft = smoke_draft(
        ctx,
        'df = pd.read_csv("data/train.csv")\nrows = df.head(1) if SMOKE else df\nprint("rows:", len(rows))',
        '!pip install -q some-package-colab-has\n(OUTPUT_DIR / "model.txt").write_text(f"smoke={SMOKE}")',
    )
    result = await smoke_test_notebook.handler(**SMOKE_ARGUMENTS, ctx=ctx)
    assert "Smoke run passed" in result.content and "model.txt" in result.content
    assert "rows: 1" in result.content
    assert (ctx.agent.workspace.root / "outputs" / "smoke" / "model.txt").read_text() == "smoke=True"
    assert draft.smoke_passed == fingerprint(draft, **SMOKE_ARGUMENTS)

    with pytest.raises(ToolError, match="Run smoke_test_notebook first"):  # other outputs than tested
        await run_colab_notebook.handler(inputs=["data/train.csv"], outputs=["other.txt"], ctx=ctx)
    save_colab_cell.handler(1, 'df = pd.read_csv("data/train.csv")', ctx=ctx)
    with pytest.raises(ToolError, match="Run smoke_test_notebook first"):  # changed since the smoke run
        await run_colab_notebook.handler(**SMOKE_ARGUMENTS, ctx=ctx)
    assert not calls(fake_colab)


@pytest.mark.asyncio
async def test_smoke_failure_names_the_cell_and_shows_what_came_before(fake_colab, tmp_path):
    ctx = tool_context(tmp_path)
    draft = smoke_draft(
        ctx,
        'manifest = pd.read_csv("data/train.csv")\nprint("columns:", list(manifest.columns))',
        'print(manifest["rank"])',
    )
    with pytest.raises(ToolError) as failure:
        await smoke_test_notebook.handler(**SMOKE_ARGUMENTS, ctx=ctx)
    message = str(failure.value)
    assert "Smoke run failed at authored cell 2: KeyError" in message
    assert "columns: ['x', 'y']" in message  # the earlier output that explains the failure
    assert draft.smoke_passed is None


@pytest.mark.asyncio
async def test_smoke_run_requires_every_declared_output(fake_colab, tmp_path):
    ctx = tool_context(tmp_path)
    smoke_draft(ctx, 'Path("model.txt").write_text("written to the wrong folder")')
    with pytest.raises(ToolError, match="OUTPUT_DIR has no model.txt"):
        await smoke_test_notebook.handler(**SMOKE_ARGUMENTS, ctx=ctx)


@pytest.mark.asyncio
async def test_smoke_run_reuses_the_inspected_kaggle_copy(fake_colab, tmp_path, monkeypatch):
    root = card_like_dataset(tmp_path / "cards")
    monkeypatch.setattr(probe, "download_dataset", lambda handle: root)
    ctx = tool_context(tmp_path)
    smoke_draft(
        ctx,
        'cards = pd.read_csv(dataset_root / "cards.csv")\nprint("root:", dataset_root, len(cards))\n'
        '(OUTPUT_DIR / "model.txt").write_text("ok")',
        kaggle_source=CARDS_DATASET,
    )
    result = await smoke_test_notebook.handler(inputs=[], outputs=["model.txt"], ctx=ctx)
    assert f"root: {root} 16" in result.content
