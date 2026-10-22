"""Isolated, disposable Docker execution without Modal credentials."""

from .image import DockerImage
from .sandbox import Sandbox, SandboxError

__all__ = ["DockerImage", "Sandbox", "SandboxError"]
__version__ = "0.1.0"
