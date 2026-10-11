"""Tests for the Jupyter kernel, the data science agent, and delegation.

Offline: a real kernel runs, but the model is scripted (no API calls).
"""

from __future__ import annotations

import json

import nbformat
import pytest

from baseagent import AgentSettings, ScriptedLLM
from baseagent.agents.coordinator import CoordinatorAgent
from baseagent.agents.data_science import DataScienceAgent, notebook_filename
from baseagent.sandbox.kernel import KernelSandbox
from baseagent.workspace import Workspace

CSV = "name,kind,age\nRex,dog,3\nTom,cat,\nRex,dog,3\nIvy,fish,1\n"


def call(name: str, args: dict, call_id: str = "call_0") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def act(*calls: dict) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


def reply(text: str) -> dict:
    return {"role": "assistant", "content": text}


@pytest.fixture
def workspace(tmp_path):
    ws = Workspace(tmp_path / "ws", url_prefix="/api/sessions/s1/files")
    (ws.ensure() / "data").mkdir()
    (ws.root / "data" / "pets.csv").write_text(CSV)
    return ws


def recorder():
    events: list[dict] = []

    async def emit(event):
        events.append(event.model_dump())

    return events, emit


async def test_kernel_keeps_state_and_recovers(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    kernel = KernelSandbox()
    await kernel.start(tmp_path)
    try:
        first = await kernel.execute("import os\nx = 41\nprint(os.environ.get('OPENAI_API_KEY'))")
        assert first.ok and first.text() == "None"  # secrets are not passed to the kernel
        assert (await kernel.execute("x + 1")).text() == "42"

        failed = await kernel.execute("1 / 0")
        assert failed.status == "error" and failed.error.startswith("ZeroDivisionError")
        assert "ZeroDivisionError" in failed.text() and "\x1b[" not in failed.text()

        slow = await kernel.execute("import time; time.sleep(30)", timeout=1.5)
        assert slow.status == "timeout" and "interrupted" in slow.text()
        assert (await kernel.execute("x")).text() == "41"  # state survives an interrupt

        crashed = await kernel.execute("import os; os._exit(1)", timeout=20)
        assert crashed.status == "restarted"
        assert (await kernel.execute("'back'")).text() == "'back'"
    finally:
        await kernel.shutdown()


def test_notebook_filename():
    assert notebook_filename("Do basic EDA on data/titanic.csv") == "titanic_eda.ipynb"
    assert notebook_filename("EDA on data/Sales_2024-Q1.xlsx please") == "Sales_2024-Q1_eda.ipynb"
    assert notebook_filename("look at my data") == "analysis.ipynb"


async def test_data_science_agent_records_a_notebook(workspace):
    llm = ScriptedLLM([
        act(call("run_cell", {"code": "df = pd.read_csv('data/pets.csv')\nprint(df.shape)",
                              "note": "## Load"}, "a"),
            call("run_cell", {"code": "df.nunique()", "note": "## Unique values"}, "b")),
        act(call("run_cell", {"code": "df.not_a_method()"}, "c")),
        reply("**4 rows, 3 columns.** One duplicate row."),
    ])
    events, emit = recorder()
    agent = DataScienceAgent(llm, AgentSettings(model="m"), emit=emit, workspace=workspace)
    try:
        result = await agent.run("Do basic EDA on data/pets.csv")
    finally:
        await agent.close()

    assert result.stop_reason == "final"
    tool_ends = [e for e in events if e["type"] == "tool_end"]
    assert tool_ends[0]["content"] == "(4, 3)"
    assert "name    3" in tool_ends[1]["content"]  # nunique output, as text
    assert tool_ends[2]["is_error"] and "AttributeError" in tool_ends[2]["content"]
    assert tool_ends[0]["attachments"][0]["url"] == "/api/sessions/s1/files/pets_eda.ipynb"
    assert "tools" in llm.requests[0] and llm.requests[0]["tools"][0]["function"]["name"] == "run_cell"

    path = workspace.root / "pets_eda.ipynb"
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    kinds = [(c.cell_type, c.source.splitlines()[0]) for c in nb.cells]
    assert kinds[0] == ("markdown", "# Exploratory data analysis")
    assert kinds[1] == ("code", "import numpy as np")
    assert ("markdown", "## Request") in kinds and ("markdown", "## Unique values") in kinds
    assert kinds[-1] == ("markdown", "## Findings")
    assert nb.cells[4].outputs[0]["text"] == "(4, 3)\n"  # outputs are kept in the notebook
    assert agent.output_files()[0].title == "pets_eda.ipynb"


async def test_coordinator_delegates_and_specialist_keeps_its_kernel(workspace):
    llm = ScriptedLLM([
        # turn 1: coordinator -> data_science -> one cell -> summary -> coordinator answer
        act(call("delegate", {"agent": "data_science", "task": "Do basic EDA on data/pets.csv"})),
        act(call("run_cell", {"code": "df = pd.read_csv('data/pets.csv')\nlen(df)", "note": "## Size"})),
        reply("4 rows."),
        reply("Your file has 4 rows. The notebook is ready below."),
        # turn 2: a follow-up reuses the same specialist and its loaded `df`
        act(call("delegate", {"agent": "data_science", "task": "How many dogs are in df?"})),
        act(call("run_cell", {"code": "(df.kind == 'dog').sum()"})),
        reply("2 dogs."),
        reply("There are 2 dogs."),
    ])
    events, emit = recorder()
    coordinator = CoordinatorAgent(llm, AgentSettings(model="gpt-test", reasoning_effort="low"),
                                   emit=emit, session_id="s1", workspace=workspace)
    try:
        first = await coordinator.run("do EDA on it\n\nAttached file: data/pets.csv")
        second = await coordinator.run("how many dogs?")
        specialist = coordinator.subagents.agents["data_science"]
        assert specialist.settings.model == "gpt-test" and specialist.settings.reasoning_effort == "low"
        assert specialist.settings.max_steps == DataScienceAgent.default_settings["max_steps"]
    finally:
        await coordinator.close()

    assert first.final_text.startswith("Your file has 4 rows") and second.final_text == "There are 2 dogs."
    delegate_ends = [e for e in events if e["type"] == "tool_end" and e["name"] == "delegate"]
    assert "4 rows." in delegate_ends[0]["content"] and "pets_eda.ipynb" in delegate_ends[0]["content"]
    assert delegate_ends[0]["attachments"][0]["title"] == "pets_eda.ipynb"
    assert "2 dogs." in delegate_ends[1]["content"]  # df survived between delegations
    progress = [e["message"] for e in events if e["type"] == "tool_progress"]
    assert progress[0] == "Data scientist: step 1 · Size"

    # Only the coordinator's own turns are in its transcript; the specialist's are separate.
    assert all(e.get("name") != "run_cell" for e in coordinator.transcript)
    roster = llm.requests[0]["tools"][-1]["function"]
    assert roster["name"] == "delegate" and "- data_science:" in roster["description"]
    assert "coordinator" not in roster["description"].split("Specialists:")[1]


async def test_delegate_rejects_unknown_agent(workspace):
    llm = ScriptedLLM([act(call("delegate", {"agent": "nope", "task": "x"})), reply("ok")])
    events, emit = recorder()
    coordinator = CoordinatorAgent(llm, AgentSettings(model="m"), emit=emit, workspace=workspace)
    await coordinator.run("hi")
    end = next(e for e in events if e["type"] == "tool_end")
    assert end["is_error"] and "Unknown agent 'nope'" in end["content"]


async def test_stop_during_a_delegated_cell_frees_the_kernel(workspace):
    import asyncio

    llm = ScriptedLLM([
        act(call("delegate", {"agent": "data_science", "task": "EDA on data/pets.csv"})),
        act(call("run_cell", {"code": "import time\ntime.sleep(60)", "note": "## Slow"})),
        # after Stop, a new turn still works in the same kernel
        act(call("delegate", {"agent": "data_science", "task": "quick check"})),
        act(call("run_cell", {"code": "'kernel is free'"})),
        reply("free"),
        reply("done"),
    ])
    events, emit = recorder()
    coordinator = CoordinatorAgent(llm, AgentSettings(model="m"), emit=emit, workspace=workspace)
    try:
        task = asyncio.create_task(coordinator.run("do EDA on it"))
        for _ in range(100):  # wait until the slow cell is running
            await asyncio.sleep(0.1)
            if any(e["type"] == "tool_progress" for e in events):
                break
        await asyncio.sleep(0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert events[-1]["type"] == "turn_end" and events[-1]["stop_reason"] == "cancelled"

        result = await asyncio.wait_for(coordinator.run("check again"), timeout=30)
        assert result.final_text == "done"
        free = [e for e in events if e["type"] == "tool_end" and e["name"] == "delegate"][-1]
        assert "free" in free["content"]
    finally:
        await coordinator.close()

    nb = nbformat.read(workspace.root / "pets_eda.ipynb", as_version=4)
    stopped = next(c for c in nb.cells if c.cell_type == "code" and "sleep(60)" in c.source)
    assert stopped.outputs[0]["text"].startswith("Stopped before this cell finished")
