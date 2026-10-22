"""The code agent's runner and runner bridge (offline: a fake sandbox stands in for Docker)."""

from __future__ import annotations

import os

import httpx
import pytest

from baseagent.codeagent import CodeRunner, RunBridge, RunLimits, RunRequest
from baseagent.codeagent.bridge import BridgeError
from baseagent.codeagent.runner import classify, trim
from baseagent.llm import ScriptedLLM
from baseagent.server.app import create_app


class FakeSandbox:
    """Records how it was started and answers each step from a script."""

    instances: list["FakeSandbox"] = []
    script: list[dict] = []

    def __init__(self, image, **options):
        self.image, self.options = image, options
        self.files: dict[str, str] = {}
        self.commands: list[tuple[list[str], str | None]] = []
        self.stopped = False
        FakeSandbox.instances.append(self)

    def write_file(self, path, text):
        self.files[path] = text

    def execute(self, command, *, shell, timeout, stdin=None, output_limit=None):
        self.commands.append((command, stdin))
        self.output_limit = output_limit
        step = FakeSandbox.script.pop(0) if FakeSandbox.script else {}
        out = {"stdout": step.get("stdout", ""), "stderr": step.get("stderr", ""),
               "returncode": step.get("returncode", 0), "output_limited": step.get("limited", False)}
        out["output"] = out["stdout"] + out["stderr"]
        return out

    def stop(self):
        self.stopped = True


@pytest.fixture(autouse=True)
def reset_fake():
    FakeSandbox.instances, FakeSandbox.script = [], []
    yield


def runner() -> CodeRunner:
    return CodeRunner(image="test-runner", sandbox_factory=FakeSandbox)


def test_run_python_is_locked_down_and_cleaned_up():
    FakeSandbox.script = [{"stdout": "45\n"}]
    result = runner().run("py", "print(sum(range(10)))", stdin="hi")
    assert (result.status, result.stdout, result.language) == ("ok", "45\n", "python")
    sandbox = FakeSandbox.instances[0]
    options = sandbox.options
    assert options["network"] == "none"
    assert options["memory"] == "512m" and options["pids_limit"] == 64
    assert options["read_only"] is True and options["user"] == "1000:1000"
    assert sandbox.files == {"/work/main.py": "print(sum(range(10)))"}
    assert sandbox.commands == [(["python", "-I", "-B", "main.py"], "hi")]
    assert sandbox.stopped


def test_no_secrets_reach_the_runner(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    runner().run("python", "print(1)")
    sandbox = FakeSandbox.instances[0]
    assert "sk-should-not-leak" not in repr(sandbox.options)
    assert "env" not in sandbox.options


def test_check_stops_at_the_first_failing_step():
    FakeSandbox.script = [{"returncode": 1, "stderr": "SyntaxError: expected ':'"}]
    result = runner().run("python", "for i in range(3)\n  pass", mode="check")
    assert result.status == "error" and "SyntaxError" in result.stderr
    assert len(FakeSandbox.instances[0].commands) == 1  # ruff never ran
    assert FakeSandbox.instances[0].commands[0][1] is None  # no stdin for checks


def test_check_passes_both_steps():
    result = runner().run("javascript", "console.log(1)", mode="check")
    assert result.status == "ok" and result.note == "No problems found."
    assert FakeSandbox.instances[0].files == {"/work/main.js": "console.log(1)"}


def test_timeout_is_reported():
    FakeSandbox.script = [{"returncode": 124}]
    result = runner().run("python", "while True: pass")
    assert result.status == "timeout"
    assert "10 s" in result.note


def test_too_much_output_is_reported():
    FakeSandbox.script = [{"returncode": -9, "stdout": "0\n" * 1000, "limited": True}]
    result = runner().run("python", "while True: print(0)")
    assert result.status == "too_much_output" and "1 MB" in result.note
    assert FakeSandbox.instances[0].output_limit == 1_000_000
    assert len(result.stdout) < 4200  # trimmed for the model


def test_out_of_memory_is_reported():
    FakeSandbox.script = [{"returncode": 1, "stderr": "Traceback ...\nMemoryError\n"}]
    assert runner().run("python", "x = [0] * 10**10").status == "out_of_memory"
    FakeSandbox.script = [{"returncode": 137}]
    assert runner().run("python", "x = bytearray(10**10)").status == "out_of_memory"


def test_unsupported_language_is_not_run():
    result = runner().run("rust", "fn main() {}")
    assert result.status == "not_runnable" and "Python" in result.note
    assert FakeSandbox.instances == []


def test_classify_and_trim():
    assert classify(0, 0.1, 10, "")[0] == "ok"
    assert classify(137, 10.2, 10, "")[0] == "timeout"
    assert classify(2, 0.1, 10, "")[0] == "error"
    long = "a" * 50 + "b" * 50
    cut = trim(long, 30)
    assert cut.startswith("a" * 20) and cut.endswith("b" * 10) and "cut" in cut


async def test_bridge_tickets_cap_and_log():
    bridge = RunBridge(runner(), max_runs=2)
    ticket = bridge.open("chat-1")
    request = RunRequest(language="python", code="print(1)")
    await bridge.handle(ticket.token, request)
    await bridge.handle(ticket.token, request)
    with pytest.raises(BridgeError) as limit:
        await bridge.handle(ticket.token, request)
    assert limit.value.status == 429
    with pytest.raises(BridgeError) as bad:
        await bridge.handle("not-a-token", request)
    assert bad.value.status == 401
    log = bridge.close(ticket.token)
    assert len(log) == 2
    with pytest.raises(BridgeError):  # closed tickets stop working
        await bridge.handle(ticket.token, request)


async def test_code_runs_endpoint(tmp_path):
    bridge = RunBridge(runner())
    app = create_app(ScriptedLLM([]), log_dir=tmp_path / "runs",
                     workspace_dir=tmp_path / "ws", code_bridge=bridge)
    ticket = bridge.open()
    FakeSandbox.script = [{"stdout": "hello\n"}]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        body = {"language": "python", "code": "print('hello')"}
        denied = await client.post("/internal/code-runs", json=body)
        assert denied.status_code == 401
        ok = await client.post("/internal/code-runs", json=body,
                               headers={"Authorization": f"Bearer {ticket.token}"})
        assert ok.status_code == 200
        assert ok.json()["status"] == "ok" and ok.json()["stdout"] == "hello\n"


# --- Real Docker (opt-in): make sandbox-images, then pytest -m docker -------------------------

docker_runner = pytest.mark.docker


def real_runner() -> CodeRunner:
    image = os.environ.get("BASEAGENT_RUNNER_IMAGE", "baseagent-runner:1")
    return CodeRunner(image=image, limits=RunLimits(run_timeout=5))


@docker_runner
def test_docker_python_runs_and_reads_stdin():
    result = real_runner().run("python", "name = input()\nprint('hi', name)", stdin="Jack\n")
    assert result.status == "ok", result
    assert result.stdout.strip() == "hi Jack"


@docker_runner
def test_docker_endless_loop_is_killed():
    result = real_runner().run("python", "while True:\n    pass\n")
    assert result.status == "timeout", result
    assert result.duration < 10


@docker_runner
def test_docker_printing_loop_is_cut_off():
    result = real_runner().run("python", "i = 0\nwhile i < 10:\n    print(i)\n")
    assert result.status == "too_much_output", result
    assert result.duration < 5


@docker_runner
def test_docker_memory_hog_is_killed():
    result = real_runner().run("python", "x = bytearray(2 * 1024**3)\nprint(len(x))")
    assert result.status == "out_of_memory", result


@docker_runner
def test_docker_no_network():
    code = ("import socket\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 80), timeout=2)\n"
            "    print('ONLINE')\n"
            "except OSError as e:\n"
            "    print('offline:', type(e).__name__)\n")
    result = real_runner().run("python", code)
    assert "offline" in result.stdout and "ONLINE" not in result.stdout, result


@docker_runner
def test_docker_fork_bomb_is_contained():
    code = ("import os\n"
            "n = 0\n"
            "try:\n"
            "    while True:\n"
            "        if os.fork() == 0:\n"
            "            import time; time.sleep(30)\n"
            "        n += 1\n"
            "except OSError:\n"
            "    print('limit hit after', n)\n")
    result = real_runner().run("python", code)
    assert "limit hit" in result.stdout or result.status in ("timeout", "error"), result


@docker_runner
def test_docker_read_only_and_no_secrets():
    code = ("import os\n"
            "try:\n"
            "    open('/etc/x', 'w')\n"
            "    print('WROTE')\n"
            "except OSError:\n"
            "    print('read-only')\n"
            "print('keys:', [k for k in os.environ if 'KEY' in k or 'TOKEN' in k])\n"
            "print('user:', os.getuid())\n")
    result = real_runner().run("python", code)
    assert "read-only" in result.stdout and "keys: []" in result.stdout, result
    assert "user: 1000" in result.stdout


@docker_runner
def test_docker_python_check_finds_syntax_error():
    result = real_runner().run("python", "def f(n)\n    return n\n", mode="check")
    assert result.status == "error" and "SyntaxError" in result.stderr, result


@docker_runner
def test_docker_javascript_runs():
    result = real_runner().run("javascript", "console.log([1,2,3].map(x => x * 2).join(','))")
    assert result.status == "ok" and result.stdout.strip() == "2,4,6", result
