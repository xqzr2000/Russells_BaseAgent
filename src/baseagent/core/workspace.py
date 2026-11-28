"""A chat's workspace: the folder its agents read and write files in.

The server gives every chat session its own folder (``workspace/<session id>/``)
and serves files from it at ``url_prefix``, so a tool can hand the user a
download link. Uploaded files go to ``data/``. Specialist agents a coordinator
delegates to share the coordinator's workspace, so they see the same uploads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

DATA_DIR = "data"


@dataclass
class Workspace:
    root: Path
    url_prefix: str | None = None  # e.g. "/api/sessions/<id>/files"; None = not served

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()

    def ensure(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def resolve(self, relative: str | Path) -> Path:
        """A path inside the workspace; raises ``ValueError`` if it would escape it."""
        path = (self.root / relative).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError(f"{relative!s} is outside the workspace")
        return path

    def relative(self, path: str | Path) -> str:
        return Path(path).resolve().relative_to(self.root).as_posix()

    def url(self, path: str | Path) -> str | None:
        """The download URL for a file in the workspace, if the server serves it."""
        if self.url_prefix is None:
            return None
        return f"{self.url_prefix}/{quote(self.relative(path))}"
