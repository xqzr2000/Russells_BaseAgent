"""Russell's BaseAgent: a small, subclassable ReAct agent core."""

from baseagent.agent import AgentSettings, BaseAgent, TurnResult
from baseagent.agents import AGENTS, discover, get_agent_class, register_agent
from baseagent.events import Attachment
from baseagent.llm import FAKE_MODEL, LLMClient, ModelSettings, OpenAIClient, ScriptedLLM
from baseagent.tools import Tool, ToolContext, ToolError, ToolResult, tool

__all__ = [
    "AGENTS", "AgentSettings", "Attachment", "BaseAgent", "FAKE_MODEL", "LLMClient",
    "ModelSettings", "OpenAIClient", "ScriptedLLM", "Tool", "ToolContext", "ToolError",
    "ToolResult", "TurnResult", "discover", "get_agent_class", "register_agent", "tool",
]
