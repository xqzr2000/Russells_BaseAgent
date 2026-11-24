"""Exercise the Colab runner with a local test CLI. No Google credentials needed."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
from fastapi.testclient import TestClient

from baseagent import AgentSettings, ScriptedLLM
from baseagent.sandbox.colab import run_colab_notebook, save_colab_cell, start_colab_notebook
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
    if mode == "error":
        cells[2].outputs = [nbformat.v4.new_output(
            "error", ename="ValueError", evalue="test training failure", traceback=[],
        )]
    else:
        cells[2].outputs = [nbformat.v4.new_output("stream", name="stdout", text="Validation score: 0.75\\n")]
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


def stage_draft(ctx):
    start_colab_notebook.handler("Test notebook", ctx=ctx)
    outcome = save_colab_cell.handler(**TRAINING_CELL, ctx=ctx)
    assert outcome.attachments[0].title == "colab_draft.ipynb"
    return ctx.state["colab_draft"].path


def test_colab_notebook_settings_and_downloads_through_jack(fake_colab, tmp_path):
    llm = ScriptedLLM([
        tool_call("delegate", {"agent": "data_science", "task": "Train on data/train.csv and return a notebook."}),
        tool_call("start_colab_notebook", {"title": "GPU baseline"}),
        tool_call("save_colab_cell", TRAINING_CELL),
        malformed_cell(),
        tool_call("save_colab_cell", {
            "number": 2, "note": "## Export", "code": 'df.to_csv("submission.csv", index=False)',
        }),
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
        assert notebook.cells[-1].source == 'df.to_csv("submission.csv", index=False)'
        assert client.get(artifacts["submission.csv"]).text == "id,target\n1,0\n"
        assert any(event["type"] == "tool_progress" and "Tesla T4" in event["message"] for event in events)

    actions = calls(fake_colab)
    assert [action[0] for action in actions] == ["new", "exec", "upload", "exec", "download", "stop"]
    assert len({action[1] for action in actions}) == 1


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
    stage_draft(ctx)
    arguments = {**ARGUMENTS, "inputs": ["data/../train.csv"]}
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
    with pytest.raises(ToolError, match=r"Colab cell 3 \(authored cell 1\) failed: ValueError"):
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
