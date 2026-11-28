"""Russell's BaseAgent: a small, subclassable ReAct agent core."""

from baseagent.core.agent import AgentSettings, BaseAgent, TurnResult
from baseagent.agents import AGENTS, discover, get_agent_class, register_agent
from baseagent.core.events import Attachment
from baseagent.core.llm import FAKE_MODEL, LLMClient, ModelSettings, OpenAIClient, ScriptedLLM
from baseagent.core.tools import Tool, ToolContext, ToolError, ToolResult, tool

__all__ = [
    "AGENTS", "AgentSettings", "Attachment", "BaseAgent", "FAKE_MODEL", "LLMClient",
    "ModelSettings", "OpenAIClient", "ScriptedLLM", "Tool", "ToolContext", "ToolError",
    "ToolResult", "TurnResult", "discover", "get_agent_class", "register_agent", "tool",
]
