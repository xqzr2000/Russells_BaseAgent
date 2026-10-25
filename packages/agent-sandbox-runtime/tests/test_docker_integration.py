import pytest

from agent_sandbox_runtime import Sandbox
from agent_sandbox_runtime.process import docker_ready

pytestmark = pytest.mark.docker


@pytest.fixture(autouse=True)
def docker_available():
    ready, detail = docker_ready()
    if not ready:
        pytest.fail(f"Docker integration requires a running daemon: {detail}")


def test_smoke_and_cleanup():
    with Sandbox("python:3.12-slim", deployment_timeout=90) as sandbox:
        assert sandbox.is_alive()
        result = sandbox.execute("echo hello", timeout=10)
        assert result["output"].strip() == "hello"
        assert result["returncode"] == 0
        assert sandbox.execute("exit 17")["returncode"] == 17
        assert sandbox.execute("sleep 3", timeout=1)["returncode"] != 0
    assert not sandbox.is_alive()
