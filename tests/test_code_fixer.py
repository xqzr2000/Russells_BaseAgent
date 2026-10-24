"""The whole code-fix pipeline in real containers, with a scripted model.

OpenCode runs in the baseagent-opencode image exactly as in production; only
the model is swapped for a scripted one (tests/fake_openai.py), reached from
the container via host.docker.internal. run_code goes through the real bridge
to real runner containers.

    make sandbox-images && make test-docker

BASEAGENT_OPENCODE_IMAGE / BASEAGENT_RUNNER_IMAGE pick other images.
"""

from __future__ import annotations

import asyncio
import os
import subprocess

import pytest

from baseagent.codeagent.__main__ import start_bridge
from baseagent.codeagent.answer import build_answer
from baseagent.codeagent.bridge import RunBridge
from baseagent.codeagent.opencode_client import CodeTask, OpenCodeFixer
from baseagent.codeagent.runner import CodeRunner

pytestmark = pytest.mark.docker

OPENCODE_IMAGE = os.environ.get("BASEAGENT_OPENCODE_IMAGE", "baseagent-opencode:1")
RUNNER_IMAGE = os.environ.get("BASEAGENT_RUNNER_IMAGE", "baseagent-runner:1")
W = "/workspace/main.py"
LOOP = "i = 0\nwhile i < 3:\n    print(i)\n"


def fixer_for(fake, bridge: RunBridge, port: int, **kw) -> OpenCodeFixer:
    from fake_openai import MODEL, provider_config

    def scripted(config: dict) -> dict:
        config["enabled_providers"] = ["opencode", "fake"]
        config["provider"] = provider_config(f"http://host.docker.internal:{fake.port}/v1")
        config["model"] = config["small_model"] = f"fake/{MODEL}"
        return config

    return OpenCodeFixer(bridge, bridge_url=f"http://host.docker.internal:{port}/internal/code-runs",
                         image=OPENCODE_IMAGE, available_models=["deepseek-v4.1-flash-free"],
                         configure=scripted, **kw)


def leftover_containers() -> list[str]:
    out = subprocess.run(["docker", "ps", "-aq", "--filter", "label=org.agent-sandbox-runtime.managed=true"],
                         capture_output=True, text=True).stdout
    return out.split()


def fix_with(script, task, **kw):
    from fake_openai import FakeOpenAI

    bridge = RunBridge(CodeRunner(image=RUNNER_IMAGE))
    port = start_bridge(bridge)
    progress: list[str] = []

    async def note(message: str) -> None:
        progress.append(message)

    with FakeOpenAI(script) as fake:
        result = asyncio.run(fixer_for(fake, bridge, port, **kw).fix(task, note))
        return result, progress, fake


def test_endless_loop_is_fixed_and_verified():
    script = [
        {"tool_calls": [("edit", {"filePath": W, "oldString": "    print(i)\n",
                                  "newString": "    print(i)\n    i += 1  # FIX 1: advance i so the loop ends\n"})]},
        {"tool_calls": [("run_code", {"path": "main.py", "mode": "check", "stdin": ""})]},
        {"tool_calls": [("run_code", {"path": "main.py", "mode": "run", "stdin": ""})]},
        {"tool_calls": [("StructuredOutput", {
            "issues": [{"line": 2, "problem": "i never changes, so the loop never ends.", "fix": "Added i += 1."}],
            "summary": "The loop now stops after printing 0, 1 and 2.", "not_run_reason": ""})]},
    ]
    fix, progress, fake = fix_with(script, CodeTask(LOOP, request="Why doesn't this stop?"))
    assert fix.answered and fix.stopped is None, fix
    assert fix.fixed == LOOP + "    i += 1  # FIX 1: advance i so the loop ends\n"
    assert [(r.mode, r.status) for r in fix.runs] == [("check", "ok"), ("run", "ok")]
    answer = build_answer(fix)
    assert "1. Line 2: i never changes" in answer
    assert "**Ran:** printed \n```text\n0\n1\n2\n```" in answer
    assert "running main.py" in progress and "run result: ok" in progress
    # The task reached the model with the user's question and code.
    first = fake.requests[0]["messages"]
    assert any("Why doesn't this stop?" in str(m.get("content")) for m in first)
    assert leftover_containers() == []


def test_review_mode_cannot_run_code():
    script = [
        {"tool_calls": [("run_code", {"path": "main.py", "mode": "run", "stdin": ""})]},
        {"tool_calls": [("StructuredOutput", {"issues": [], "summary": "Looks fine.",
                                              "not_run_reason": "Reviewed only, as asked."})]},
    ]
    fix, _, fake = fix_with(script, CodeTask("print('hi')\n", mode="review"))
    assert "run_code" not in fake.tool_names(0)
    assert fix.runs == [] and fix.answered
    assert "**Reviewed only:** not run, as you asked." in build_answer(fix)


def test_time_limit_aborts_and_still_answers():
    script = [{"tool_calls": [("read", {"filePath": W})]}, {"delay": 30, "text": "too late"}]
    fix, progress, _ = fix_with(script, CodeTask(LOOP), time_limit=6)
    assert fix.stopped == "time limit"
    assert any("limit was reached" in p for p in progress)
    assert build_answer(fix).startswith("*OpenCode ran out of time (3-minute limit).*")
    assert leftover_containers() == []


def test_chat_message_runs_the_code_agent_end_to_end(tmp_path, monkeypatch):
    """The real chat path: HTTP message -> Jack (scripted) -> fix_code -> OpenCode container
    (scripted model) -> the server's own /internal/code-runs -> runner container -> reply."""
    import json as _json
    import socket
    import threading
    import time

    import httpx
    import uvicorn
    from fake_openai import FakeOpenAI

    from baseagent import ScriptedLLM
    from baseagent.codeagent import tool as code_tool
    from baseagent.server.app import create_app

    jack = ScriptedLLM([{"role": "assistant", "content": "", "tool_calls": [{
        "id": "call_0", "type": "function",
        "function": {"name": "fix_code", "arguments": _json.dumps({"mode": "auto"})}}]}])
    bridge = RunBridge(CodeRunner(image=RUNNER_IMAGE))
    app = create_app(llm=jack, log_dir=tmp_path / "runs", workspace_dir=tmp_path / "ws", code_bridge=bridge)
    with socket.socket() as s:
        s.bind(("0.0.0.0", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)

    script = [
        {"tool_calls": [("edit", {"filePath": W, "oldString": "    print(i)\n",
                                  "newString": "    print(i)\n    i += 1  # FIX 1: advance i\n"})]},
        {"tool_calls": [("run_code", {"path": "main.py", "mode": "run", "stdin": ""})]},
        {"tool_calls": [("StructuredOutput", {
            "issues": [{"line": 2, "problem": "i never changes.", "fix": "Added i += 1."}],
            "summary": "Fixed the endless loop.", "not_run_reason": ""})]},
    ]
    try:
        with FakeOpenAI(script) as fake:
            # Production wiring (the server's bridge, reached on its own port); only the model is scripted.
            monkeypatch.setattr(code_tool, "_fixer", fixer_for(fake, bridge, port))
            base = f"http://127.0.0.1:{port}"
            with httpx.Client(base_url=base, timeout=300, trust_env=False) as client:
                sid = client.post("/api/sessions", json={"agent": "coordinator"}).json()["id"]
                events = []
                with client.stream("POST", f"/api/sessions/{sid}/messages",
                                   json={"content": f"Why doesn't this stop?\n```python\n{LOOP}```"}) as stream:
                    for line in stream.iter_lines():
                        if line.startswith("data: "):
                            events.append(_json.loads(line[6:]))
    finally:
        server.should_exit = True
    end = next(e for e in events if e["type"] == "turn_end")
    assert end["stop_reason"] == "final", end
    assert end["final_text"].startswith("**What was wrong**\n1. Line 2: i never changes.")
    assert "**Ran:** printed \n```text\n0\n1\n2\n```" in end["final_text"]
    assert any(e["type"] == "tool_progress" and e["message"] == "running main.py" for e in events)
    assert len(jack.requests) == 1  # Jack never retyped or summarised the answer
    assert leftover_containers() == []
