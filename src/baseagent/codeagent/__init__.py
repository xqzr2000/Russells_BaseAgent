"""The code agent's machinery (OpenCode in a sandbox; it is not a BaseAgent).

* ``runner``: runs a snippet in a throwaway container with no network
* ``bridge``: the token-protected door OpenCode's ``run_code`` tool calls to
  reach the runner, with a per-request run log and run cap
* ``opencode_config``: what OpenCode's ``fixer`` agent may do, the answer
  schema, and the free-model choice
"""

from baseagent.codeagent.bridge import BridgeError, RunBridge, RunRequest, Ticket
from baseagent.codeagent.runner import CodeRunner, RunLimits, RunnerUnavailable, RunResult

__all__ = ["BridgeError", "CodeRunner", "RunBridge", "RunLimits", "RunRequest", "RunResult",
           "RunnerUnavailable", "Ticket"]
