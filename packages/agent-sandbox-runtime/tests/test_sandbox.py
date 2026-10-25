from unittest.mock import patch
import subprocess

import pytest

from agent_sandbox_runtime.sandbox import Sandbox, SandboxError


def result(out="", err="", code=0):
    return subprocess.CompletedProcess(["docker"], code, out, err)


def test_sandbox_commands_and_ports():
    commands = []
    def fake(*args, **kwargs):
        commands.append(args)
        if args[0] == "run":
            return result("the-container\n")
        if args[0] == "inspect":
            return result("true\n")
        if args[0] == "port":
            return result("127.0.0.1:32123\n")
        if args[0] == "exec":
            return result("hi\n", "warning\n")
        return result()
    with patch("agent_sandbox_runtime.sandbox.docker", side_effect=fake):
        with Sandbox("python:3.12", ports=[8000], deployment_timeout=100) as sb:
            assert sb.tunnel_url(8000) == "http://127.0.0.1:32123"
            output = sb.execute("echo hi", timeout=10, env={"FOO": "bar"})
            assert output["output"] == "hi\nwarning\n"
            assert output["returncode"] == 0
        assert any(c[0] == "rm" for c in commands)
        run = next(c for c in commands if c[0] == "run")
        assert "--cap-drop" in run and "--publish" in run
        assert "127.0.0.1::8000" in run
        assert "--rm" in run
        command = next(c for c in commands if c[0] == "exec")
        assert "--env" in command
        assert "timeout" in command


def test_no_remote_public_host_ports():
    with patch("agent_sandbox_runtime.sandbox.docker", return_value=result("abc")) as fake:
        with Sandbox("python:3.12", ports=[8080]) as sb:
            with pytest.raises(SandboxError, match="loopback"):
                sb.tunnel_url(8080)


def test_rejects_unsafe_argv():
    with pytest.raises(ValueError):
        Sandbox("python:3.12", ports=[99999])


def test_spawn_keeps_secrets_off_the_command_line():
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == "run":
            return result("cid\n")
        if args[0] == "inspect":
            return result("true\n")
        return result()

    with patch("agent_sandbox_runtime.sandbox.docker", side_effect=fake):
        with Sandbox("img", deployment_timeout=100) as sb:
            sb.spawn(["server", "--port", "1"], env={"PASSWORD": "s3cret"}, cwd="/w")
    args, kwargs = next(c for c in calls if c[0][0] == "exec")
    assert "--detach" in args and "PASSWORD" in args
    assert not any("s3cret" in a for a in args)
    assert kwargs["env"] == {"PASSWORD": "s3cret"}
    assert args[-3:] == ("server", "--port", "1")
