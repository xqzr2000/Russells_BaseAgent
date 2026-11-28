"""Locked-down containers for running code (Docker, via ``agent-sandbox-runtime``).

* ``runner``: runs one snippet in a throwaway container with no network, a
  read-only file system and tight limits

Kept from the removed OpenCode code agent for the planned Claude Code sandbox.
"""

from baseagent.sandbox.runner import CodeRunner, RunLimits, RunnerUnavailable, RunResult

__all__ = ["CodeRunner", "RunLimits", "RunResult", "RunnerUnavailable"]
