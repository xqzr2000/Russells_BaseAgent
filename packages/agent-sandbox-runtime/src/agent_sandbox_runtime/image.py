"""Build immutable Docker task images from safe local build contexts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from .process import DockerError, docker

# Never send credentials or Git metadata to the daemon's build context.
IGNORED = frozenset({
    ".git", ".venv", ".env", ".env.local", ".pytest_cache", "__pycache__",
    ".DS_Store", ".idea", ".vscode", ".ssh", ".aws", ".config", ".npmrc",
    ".pypirc", ".netrc", ".git-credentials", "id_rsa", "id_ed25519", "credentials.json",
})


def _copy_safe(source: Path, destination: Path, ignore_names: frozenset[str]) -> None:
    """Copy a checkout without following symlinks outside the source or secret files."""
    if not source.is_dir():
        raise FileNotFoundError(f"Docker build context does not exist: {source}")
    for base, dirs, files in os.walk(source, followlinks=False):
        rel = Path(base).relative_to(source)
        dirs[:] = sorted(d for d in dirs if d not in ignore_names and not (Path(base) / d).is_symlink())
        target = destination / rel
        target.mkdir(parents=True, exist_ok=True)
        for filename in sorted(files):
            candidate = Path(base) / filename
            if (filename in ignore_names or filename.startswith(".env")
                or filename.endswith((".pyc", ".pem", ".key", ".p12", ".pfx"))
                or candidate.is_symlink()):
                continue
            shutil.copy2(candidate, target / filename)


def _fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()[:20]


@dataclass(frozen=True)
class DockerImage:
    """Recipe which resolves to a tagged local image when a Sandbox starts.

    Build is intentionally deferred: a caller can append runtime script files
    before building. Docker layer caching keeps repeated builds practical.
    """

    dockerfile: Path
    context_dir: Path
    pins: tuple[str, ...] = ()
    local_files: tuple[tuple[Path, str], ...] = ()
    ignored_names: frozenset[str] = IGNORED
    force_build: bool = False

    @classmethod
    def from_dockerfile(
        cls,
        dockerfile: str | Path,
        context_dir: str | Path,
        *,
        pins: tuple[str, ...] | list[str] = (),
        force_build: bool = False,
    ) -> "DockerImage":
        return cls(Path(dockerfile).resolve(), Path(context_dir).resolve(), tuple(pins), force_build=force_build)

    def add_local_file(self, source: str | Path, destination: str, copy: bool = True) -> "DockerImage":
        source = Path(source).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if not destination.startswith("/opt/assignment/") or ".." in Path(destination).parts or any(c.isspace() for c in destination):
            raise ValueError("Only simple absolute /opt/assignment/ paths are allowed for injected runtime files")
        return replace(self, local_files=(*self.local_files, (source, destination)))

    def build(self) -> str:
        if not self.dockerfile.is_file():
            raise FileNotFoundError(self.dockerfile)
        with tempfile.TemporaryDirectory(prefix="agent-image-") as temp:
            stage = Path(temp)
            ctx = stage / "context"
            _copy_safe(self.context_dir, ctx, self.ignored_names)
            # Dockerfile may live outside the project source tree; do not
            # inject build metadata into the source copied to /testbed.
            for index, (source, _) in enumerate(self.local_files):
                target = stage / "extras" / str(index)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            fingerprint = hashlib.sha256()
            fingerprint.update(_fingerprint(ctx).encode())
            fingerprint.update(self.dockerfile.read_bytes())
            fingerprint.update(_fingerprint(stage / "extras").encode() if self.local_files else b"no-extras")
            fingerprint.update(json.dumps({"pins": self.pins, "paths": [d for _, d in self.local_files]}).encode())
            key = fingerprint.hexdigest()[:20]
            tag = f"agent-sandbox-task:{key}"
            if not self.force_build and docker("image", "inspect", tag, check=False).returncode == 0:
                return tag
            base_digest = hashlib.sha256(_fingerprint(ctx).encode() + self.dockerfile.read_bytes()).hexdigest()[:20]
            base_tag = f"agent-sandbox-base:{base_digest}"
            if self.force_build or docker("image", "inspect", base_tag, check=False).returncode != 0:
                argv = ["build", "--tag", base_tag, "--file", str(self.dockerfile)]
                if self.force_build:
                    argv.append("--no-cache")
                docker(*argv, str(ctx), timeout=1800)
            if not self.pins and not self.local_files:
                # Keep the base cached but also expose the intended final tag.
                docker("tag", base_tag, tag)
                return tag
            layer = stage / "layer"
            layer.mkdir()
            lines = [f"FROM {base_tag}"]
            if self.local_files:
                for index, (_, dest) in enumerate(self.local_files):
                    shutil.copy2(stage / "extras" / str(index), layer / f"asset-{index}")
                    lines.append(f"COPY asset-{index} {dest}")
            if self.pins:
                (layer / "requirements-agent.txt").write_text("\n".join(self.pins) + "\n")
                lines += ["COPY requirements-agent.txt /tmp/requirements-agent.txt", "RUN python -m pip install --no-cache-dir -r /tmp/requirements-agent.txt"]
            (layer / "Dockerfile").write_text("\n".join(lines) + "\n")
            argv = ["build", "--tag", tag]
            if self.force_build:
                argv.append("--no-cache")
            docker(*argv, str(layer), timeout=1800)
            return tag
