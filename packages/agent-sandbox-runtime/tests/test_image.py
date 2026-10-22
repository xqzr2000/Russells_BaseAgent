from pathlib import Path

import pytest

from agent_sandbox_runtime.image import DockerImage, _copy_safe


def test_ignores_secrets_and_symlinks(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "dest"
    source.mkdir()
    (source / "app.py").write_text("print('ok')")
    (source / ".env").write_text("OPENAI_API_KEY=secret")
    (source / ".git").mkdir()
    (source / ".git" / "config").write_text("token")
    (source / "link").symlink_to(tmp_path / "nonexistent")
    from agent_sandbox_runtime.image import IGNORED
    _copy_safe(source, dest, IGNORED)
    assert (dest / "app.py").is_file()
    assert not (dest / ".env").exists()
    assert not (dest / ".git").exists()
    assert not (dest / "link").exists()


def test_image_add_file_immutable(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim")
    first = DockerImage.from_dockerfile(dockerfile, tmp_path)
    second = first.add_local_file(dockerfile, "/opt/assignment/example")
    assert not first.local_files
    assert len(second.local_files) == 1
    with pytest.raises(ValueError):
        first.add_local_file(dockerfile, "/etc/passwd")


def test_build_generates_base_and_final_layers(tmp_path):
    import subprocess
    from unittest.mock import patch

    source = tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text("print('ok')")
    (source / ".env.test").write_text("not-for-build")
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\nCOPY . /testbed\n")
    asset = tmp_path / "asset.py"
    asset.write_text("print('asset')")
    image = DockerImage.from_dockerfile(dockerfile, source, pins=["httpx==0.28.1"])
    image = image.add_local_file(asset, "/opt/assignment/asset.py")
    executed = []

    def fake(*args, **kwargs):
        executed.append(args)
        if args[0] == "image":
            return subprocess.CompletedProcess(["docker"], 1, "", "missing")
        if args[0] == "build" and "--file" in args:
            assert not Path(args[-1], ".env.test").exists()
        if args[0] == "build" and "--file" not in args:
            assert "httpx==0.28.1" in (Path(args[-1]) / "requirements-agent.txt").read_text()
            assert "COPY asset-0 /opt/assignment/asset.py" in (Path(args[-1]) / "Dockerfile").read_text()
        return subprocess.CompletedProcess(["docker"], 0, "ok", "")

    with patch("agent_sandbox_runtime.image.docker", side_effect=fake):
        tag = image.build()
    assert tag.startswith("agent-sandbox-task:")
    assert sum(args[0] == "build" for args in executed) == 2
