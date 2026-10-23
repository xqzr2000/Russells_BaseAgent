"""The code agent's OpenCode config and free-model choice.

The last test runs a real OpenCode binary (opt-in: OPENCODE_BIN=/path/to/opencode
pytest -m opencode) against a scripted fake model, and checks what the fixer is
actually allowed to do. No Docker needed: the runner behind run_code is faked.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import pytest

from baseagent.codeagent import CodeRunner, RunBridge
from baseagent.codeagent.opencode_config import (
    ANSWER_SCHEMA, IMAGE_ENV, PERMISSION, ZERO_RETENTION_MODELS, NoFreeModel, build_config, parse_model_list,
    pick_model,
)

ROOT = Path(__file__).resolve().parents[1]

LISTED = ["big-pickle", "cat-jam-free", "deepseek-v4.1-flash-free", "longcat-2.5-preview-free",
          "mimo-v2.6-flash-free", "nemotron-3-ultra-free", "jev-1.13-free"]


def test_parse_model_list():
    out = "opencode/big-pickle\nopencode/deepseek-v4.1-flash-free\nsome warning line\n"
    assert parse_model_list(out) == ["big-pickle", "deepseek-v4.1-flash-free"]


def test_zero_retention_models_come_first():
    assert pick_model(LISTED, allow_training=False).model == "deepseek-v4.1-flash-free"
    second = [m for m in LISTED if m != "deepseek-v4.1-flash-free"]
    choice = pick_model(second, allow_training=False)
    assert (choice.model, choice.zero_retention) == ("longcat-2.5-preview-free", True)


def test_training_models_need_the_opt_in(monkeypatch):
    others = [m for m in LISTED if m not in ZERO_RETENTION_MODELS]
    monkeypatch.delenv("OPENCODE_ALLOW_TRAINING_MODELS", raising=False)
    with pytest.raises(NoFreeModel, match="OPENCODE_ALLOW_TRAINING_MODELS"):
        pick_model(others)
    monkeypatch.setenv("OPENCODE_ALLOW_TRAINING_MODELS", "1")
    choice = pick_model(others)
    assert (choice.model, choice.zero_retention) == ("big-pickle", False)


def test_preferred_model_must_be_listed_and_allowed():
    assert pick_model(LISTED, preferred="opencode/longcat-2.5-preview-free",
                      allow_training=False).model == "longcat-2.5-preview-free"
    # Not allowed without the opt-in, so the default is used instead.
    assert pick_model(LISTED, preferred="big-pickle", allow_training=False).model == "deepseek-v4.1-flash-free"
    # Not listed (e.g. a paid model): ignored.
    assert pick_model(LISTED, preferred="gpt-5", allow_training=True).model == "deepseek-v4.1-flash-free"


def test_non_coding_models_are_never_picked():
    with pytest.raises(NoFreeModel):
        pick_model(["jev-1.13-free"], allow_training=True)


def test_config_locks_the_fixer_down():
    config = build_config("deepseek-v4.1-flash-free")
    assert config["model"] == "opencode/deepseek-v4.1-flash-free"
    assert config["enabled_providers"] == ["opencode"]
    assert config["share"] == "disabled" and config["formatter"] is False and config["lsp"] is False
    for rules in (config["permission"], config["agent"]["fixer"]["permission"]):
        keys = list(rules)
        assert keys[0] == "*" and rules["*"] == "deny"  # first, so later allows win
        assert rules["StructuredOutput"] == rules["run_code"] == "allow"
        assert all(rules[k] == "deny" for k in ("external_directory", "doom_loop"))
        assert "bash" not in rules and "webfetch" not in rules  # denied by "*"
    assert config["agent"]["fixer"]["steps"] > 0
    assert not any("key" in k.lower() for k in json.dumps(config).split('"'))  # no API key


def test_dockerfile_sets_the_image_env():
    dockerfile = (ROOT / "docker/opencode/Dockerfile").read_text()
    for key, value in IMAGE_ENV.items():
        assert f"{key}={value}" in dockerfile, key
    assert "XDG_CONFIG_HOME=/opt/baseagent/config" in dockerfile


def test_answer_schema_requires_issues_and_summary():
    assert ANSWER_SCHEMA["required"] == ["issues", "summary"]
    item = ANSWER_SCHEMA["properties"]["issues"]["items"]
    assert item["required"] == ["line", "problem", "fix"]


# --- Real OpenCode against a scripted model (opt-in) ------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Sandbox:
    """Stands in for the runner container: 'runs' Python by printing 45."""

    def __init__(self, image, **_):
        pass

    def write_file(self, path, text):
        self.code = text

    def execute(self, command, **_):
        ok = "+=" in self.code
        out = "45\n" if ok and "python" in command else ""
        return {"stdout": out, "stderr": "", "output": out, "returncode": 0 if ok else 1}

    def stop(self):
        pass


@pytest.mark.opencode
def test_real_opencode_fixer_tools_and_run_code(tmp_path):
    binary = os.environ.get("OPENCODE_BIN") or shutil.which("opencode")
    if not binary:
        pytest.skip("set OPENCODE_BIN to an OpenCode 1.19.2 binary")
    if os.geteuid() == 0:  # OpenCode will run as another user: use a folder it can enter
        tmp_path = Path(tempfile.mkdtemp(prefix="opencode-test-"))
        os.chmod(tmp_path, 0o755)
    import httpx
    import uvicorn
    from fake_openai import MODEL, FakeOpenAI, provider_config

    from baseagent.server.app import create_app

    # The image's layout: read-only global config folder holding run_code.js.
    xdg = tmp_path / "xdg"
    shutil.copytree(ROOT / "docker/opencode/config", xdg / "opencode")
    (xdg / "opencode/.gitignore").write_text("node_modules\npackage.json\npackage-lock.json\nbun.lock\n.gitignore")
    ws, home = tmp_path / "workspace", tmp_path / "home"
    ws.mkdir()
    home.mkdir()
    main = ws / "main.py"
    main.write_text("def add_up(n)\n    total = 0\n    for i in range(n):\n        total =+ i\n    return total\n\n"
                    "print(add_up(10))\n")
    run_as: list[str] = []
    if os.geteuid() == 0:  # root can write anything; run OpenCode as an unprivileged user
        user = os.environ.get("OPENCODE_TEST_USER", "nobody")
        for path in (ws, home, main):
            shutil.chown(path, user)
        os.chmod(tmp_path, 0o755)
        run_as = ["runuser", "-u", user, "--"]
    subprocess.run(["chmod", "-R", "a+rX,a-w", str(xdg)], check=True)

    bridge = RunBridge(CodeRunner(image="fake", sandbox_factory=_Sandbox))
    ticket = bridge.open()
    bridge_port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(code_bridge=bridge), host="127.0.0.1",
                                           port=bridge_port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()

    script = [
        {"tool_calls": [("bash", {"command": "env"})]},
        {"tool_calls": [("read", {"filePath": "/etc/passwd"})]},
        {"tool_calls": [("edit", {"filePath": str(main), "oldString": "def add_up(n)\n",
                                  "newString": "def add_up(n):  # FIX 1: added ':'\n"})]},
        {"tool_calls": [("edit", {"filePath": str(main), "oldString": "total =+ i\n",
                                  "newString": "total += i  # FIX 2: '=+' -> '+='\n"})]},
        {"tool_calls": [("run_code", {"path": "main.py", "mode": "run", "stdin": ""})]},
        {"tool_calls": [("StructuredOutput", {
            "issues": [{"line": 1, "problem": "missing ':'", "fix": "added ':'"},
                       {"line": 4, "problem": "'=+'", "fix": "'+='"}],
            "summary": "fixed"})]},
    ]
    with FakeOpenAI(script) as fake:
        config = build_config("deepseek-v4.1-flash-free")
        config["enabled_providers"] = ["opencode", "fake"]
        config["provider"] = provider_config(fake.base_url)
        config["model"] = config["small_model"] = f"fake/{MODEL}"
        port, password = _free_port(), "test-password"
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(home), "XDG_CONFIG_HOME": str(xdg),
            **IMAGE_ENV,
            "OPENCODE_CONFIG_CONTENT": json.dumps(config), "OPENCODE_SERVER_PASSWORD": password,
            "BASEAGENT_RUN_URL": f"http://127.0.0.1:{bridge_port}/internal/code-runs",
            "BASEAGENT_RUN_TOKEN": ticket.token, "NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1",
        }
        log = (tmp_path / "opencode.log").open("w")
        proc = subprocess.Popen([*run_as, "env", "-i", *[f"{k}={v}" for k, v in env.items()], binary, "serve",
                                 "--hostname", "127.0.0.1", "--port", str(port), "--print-logs"],
                                cwd=ws, stdout=log, stderr=subprocess.STDOUT)
        client = httpx.Client(base_url=f"http://127.0.0.1:{port}", auth=("opencode", password),
                              timeout=120, trust_env=False)
        try:
            for _ in range(240):
                try:
                    # Short timeout: a request sent while OpenCode is still starting can
                    # be accepted and never answered.
                    if client.get("/global/health", timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.25)
            else:
                pytest.fail("OpenCode did not start; see " + str(tmp_path / "opencode.log"))
            if proc.poll() is not None:
                pytest.fail("OpenCode exited: " + (tmp_path / "opencode.log").read_text()[-2000:])
            assert httpx.get(f"http://127.0.0.1:{port}/global/health", trust_env=False).status_code == 401
            session = client.post("/session", json={"title": "Fix main.py"}).json()
            reply = client.post(f"/session/{session['id']}/message", json={
                "agent": "fixer", "parts": [{"type": "text", "text": "Fix main.py."}],
                "format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
            }).json()
        finally:
            proc.terminate()
            proc.wait(10)
            server.should_exit = True

        # Exactly these tools, nothing else (no shell, web, sub-agents, questions...).
        assert fake.tool_names(0) == ["StructuredOutput", "edit", "glob", "grep", "read", "run_code", "write"]
        results = fake.tool_results(len(fake.requests) - 1)
        assert "unavailable tool 'bash'" in results[0]
        assert "prevents you from using this specific tool call" in results[1]  # outside the workspace
        assert "Ran without errors" in results[4] and "45" in results[4]
    assert reply["info"].get("structured", {}).get("summary") == "fixed"
    assert "total += i  # FIX 2" in main.read_text()
    assert [(r.mode, r.status) for r in ticket.runs] == [("run", "ok")]
    assert sorted(p.name for p in (xdg / "opencode").iterdir()) == [".gitignore", "tools"]  # no npm install
