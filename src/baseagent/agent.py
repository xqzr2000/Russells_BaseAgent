"""BaseAgent: the shared ReAct loop every specialized agent inherits.

A subclass describes *what* the agent is; the base class owns *how* it runs.

Override these (all optional):

    name / title / description   class attributes shown in the UI
    examples                     starter prompts for an empty chat
    default_settings             model/step defaults for this agent
    skills_dir                   folder of SKILL.md skills for this agent
    system_prompt()              standing instructions
    get_tools()                  the tools it may call
    async setup() / close()      async resources (a Python kernel, a DB...)
    is_final(message)            when a turn is finished
    format_observation(result)   how a tool result is shown to the model

Rule for subclasses: ``__init__`` only stores options. Anything that needs
I/O or depends on subclass state belongs in ``setup()``, which the loop awaits
once before the first turn.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, ClassVar

from pydantic import BaseModel

from baseagent import events as ev
from baseagent.llm import LLMClient, ModelSettings
from baseagent.memory import Memory, rough_tokens, truncate
from baseagent.skills import SkillLibrary
from baseagent.tools import Tool, ToolContext, ToolError, ToolResult

Emit = Callable[[BaseModel], Awaitable[None]]

NO_TOOL_CALL_NUDGE = (
    "Your last response neither called a tool nor finished the task. "
    "Continue by calling one of the available tools."
)


class AgentSettings(ModelSettings):
    """Everything the chat room's Settings panel can change."""

    max_steps: int = 20
    compact_threshold_tokens: int | None = None  # None = never compact
    compaction_keep_recent_steps: int = 2
    compaction_max_tokens: int = 1500
    tool_timeout: float = 120.0


@dataclass
class TurnResult:
    final_text: str
    steps: int
    stop_reason: str
    usage: ev.Usage


async def _ignore(_: BaseModel) -> None:
    return None


class BaseAgent:
    name: ClassVar[str] = "base"
    title: ClassVar[str] = "Base agent"
    description: ClassVar[str] = "A general assistant with no tools."
    default_settings: ClassVar[dict[str, Any]] = {}
    examples: ClassVar[list[str]] = []  # starter prompts shown in the chat room
    skills_dir: ClassVar[str | Path | None] = None

    def __init__(
        self,
        llm: LLMClient,
        settings: AgentSettings | None = None,
        emit: Emit | None = None,
        log_dir: str | Path | None = None,
        session_id: str = "local",
    ) -> None:
        # Store options only; see the module docstring.
        self.llm = llm
        self.settings = settings or self.make_settings()
        self.emit_callback: Emit = emit or _ignore
        self.log_dir = Path(log_dir) if log_dir else None
        self.session_id = session_id

        self.memory = Memory()
        self.state: dict[str, Any] = {}  # free scratch space for tools/subclasses
        self.transcript: list[dict[str, Any]] = []  # every event, for reloads and logs
        self.compactions: list[dict[str, Any]] = []
        self.turn = 0
        self.total_usage = ev.Usage()
        self._tools: dict[str, Tool] = {}
        self._skills = SkillLibrary()
        self._ready = False
        self._last_prompt_tokens: tuple[int, int] | None = None  # (actual, rough)

    # ------------------------------------------------------------------ hooks

    @classmethod
    def make_settings(cls, **overrides: Any) -> AgentSettings:
        return AgentSettings(**{**cls.default_settings, **overrides})

    def system_prompt(self) -> str:
        return (
            "You are a helpful, precise assistant. Use the available tools when they "
            "help you answer accurately, and answer directly when they don't. "
            "When you have the answer, reply to the user in plain text."
        )

    def get_tools(self) -> list[Tool]:
        return []

    async def setup(self) -> None:
        """Acquire async resources. Called once, before the first turn."""

    async def close(self) -> None:
        """Release resources acquired in ``setup``."""

    def is_final(self, message: dict[str, Any]) -> bool:
        """A turn ends when the model replies without calling a tool."""
        return not message.get("tool_calls")

    def format_observation(self, result: ToolResult) -> str:
        text = f"Error: {result.content}" if result.is_error else result.content
        return truncate(text)

    # ------------------------------------------------------------- lifecycle

    async def ensure_ready(self) -> None:
        if self._ready:
            return
        await self.setup()
        self._skills = SkillLibrary.load(self.skills_dir) if self.skills_dir else SkillLibrary()
        tools = list(self.get_tools())
        if self._skills:
            tools.append(self._skills.as_tool())
        self._tools = {}
        for t in tools:
            if t.name in self._tools:
                raise ValueError(f"Duplicate tool name {t.name!r} in {type(self).__name__}")
            self._tools[t.name] = t
        self._ready = True

    @property
    def tools(self) -> dict[str, Tool]:
        return self._tools

    def full_system_prompt(self) -> str:
        prompt = self.system_prompt()
        if self._skills:
            prompt += "\n\n" + self._skills.catalog_prompt()
        return prompt

    def build_prompt(self) -> list[dict[str, Any]]:
        return self.memory.messages(self.full_system_prompt())

    async def emit(self, event: BaseModel) -> None:
        if getattr(event, "type", None) != "text_delta":
            self.transcript.append(ev.event_to_dict(event))
        await self.emit_callback(event)

    # ---------------------------------------------------------- the loop

    async def run(self, user_message: str) -> TurnResult:
        """Handle one user message: loop model → tools → model until final."""

        await self.ensure_ready()
        self.turn += 1
        turn = self.turn
        await self.emit(ev.TurnStart(turn=turn, user_message=user_message,
                                     model=self.settings.model, agent=self.name))
        self.memory.add({"role": "user", "content": user_message})

        steps, usage, final_text, stop_reason = 0, ev.Usage(), "", "final"
        tool_schemas = [t.schema() for t in self._tools.values()] or None
        try:
            while True:
                if steps >= self.settings.max_steps:
                    stop_reason = "step_limit"
                    final_text = f"(Stopped after reaching the {self.settings.max_steps}-step limit.)"
                    break
                await self.maybe_compact(turn)
                steps += 1
                await self.emit(ev.StepStart(turn=turn, step=steps))

                async def on_text(delta: str, step: int = steps) -> None:
                    await self.emit(ev.TextDelta(turn=turn, step=step, delta=delta))

                messages = self.build_prompt()
                response = await self.llm.chat(messages, self.settings, tool_schemas, on_text)
                self._last_prompt_tokens = (response.usage.prompt_tokens, rough_tokens(messages))
                for warning in response.warnings:
                    await self.emit(ev.Warning(turn=turn, message=warning))
                usage = usage + response.usage
                await self.emit(ev.UsageEvent(turn=turn, step=steps, usage=response.usage))

                action = self._clean_action(response.message)
                self.memory.add(action)
                calls = action.get("tool_calls", [])
                await self.emit(ev.AssistantMessage(
                    turn=turn, step=steps, content=action["content"],
                    tool_calls=[ev.ToolCallInfo(id=c["id"], name=c["function"]["name"],
                                                arguments=c["function"]["arguments"]) for c in calls],
                ))

                if self.is_final(action):
                    final_text = action["content"]
                    break
                if not calls:
                    self.memory.add({"role": "user", "content": NO_TOOL_CALL_NUDGE})
                    continue
                for observation in await self.execute_tool_calls(calls, turn, steps):
                    self.memory.add(observation)
        except asyncio.CancelledError:
            self.memory.repair("The user stopped the turn before this call ran.")
            stop_reason, final_text = "cancelled", "(Stopped by the user.)"
            await self._finish(turn, final_text, steps, stop_reason, usage)
            raise
        except Exception as exc:  # noqa: BLE001 - report, keep the session usable
            self.memory.repair("The turn failed before this call ran.")
            stop_reason, final_text = "error", f"{type(exc).__name__}: {exc}"
            await self.emit(ev.Error(turn=turn, message=final_text))
        await self._finish(turn, final_text, steps, stop_reason, usage)
        return TurnResult(final_text, steps, stop_reason, usage)

    async def _finish(self, turn, final_text, steps, stop_reason, usage) -> None:
        self.total_usage = self.total_usage + usage
        await self.emit(ev.TurnEnd(turn=turn, final_text=final_text, steps=steps,
                                   stop_reason=stop_reason, usage=usage))
        self.save_log()

    @staticmethod
    def _clean_action(message: dict[str, Any]) -> dict[str, Any]:
        action: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
        calls = [c for c in message.get("tool_calls") or [] if isinstance(c, dict)]
        for i, call in enumerate(calls):
            call.setdefault("type", "function")
            call["id"] = call.get("id") or f"call_{i}"
            call.setdefault("function", {}).setdefault("arguments", "")
        if calls:
            action["tool_calls"] = calls
        return action

    # ---------------------------------------------------------- tools

    async def execute_tool_calls(
        self, calls: list[dict[str, Any]], turn: int, step: int
    ) -> list[dict[str, Any]]:
        """Run each call in order and return one linked ``tool`` message per call."""

        observations = []
        exclusive_used = False
        for call in calls:
            fn = call.get("function", {})
            name, arguments = fn.get("name", ""), fn.get("arguments", "")
            await self.emit(ev.ToolStart(turn=turn, step=step, call_id=call["id"],
                                         name=name, arguments=arguments))
            started = time.monotonic()
            tool = self._tools.get(name)
            if tool is not None and tool.exclusive:
                if exclusive_used:
                    result = ToolResult(
                        f"`{name}` was not run: only one `{name}`-type call runs per response "
                        "because it changes shared state. Check the latest result and call again.",
                        is_error=True)
                else:
                    exclusive_used = True
                    result = await self.run_tool(tool, arguments, call["id"])
            elif tool is None:
                result = ToolResult(f"Unknown tool {name!r}. Available tools: "
                                    f"{', '.join(self._tools) or '(none)'}.", is_error=True)
            else:
                result = await self.run_tool(tool, arguments, call["id"])
            content = self.format_observation(result)
            await self.emit(ev.ToolEnd(
                turn=turn, step=step, call_id=call["id"], name=name, content=content,
                is_error=result.is_error, attachments=result.attachments,
                duration_ms=int((time.monotonic() - started) * 1000),
            ))
            observations.append({"role": "tool", "tool_call_id": call["id"], "content": content})
        return observations

    async def run_tool(self, tool: Tool, arguments: str, call_id: str) -> ToolResult:
        try:
            parsed = tool.parse(arguments)
            timeout = tool.timeout or self.settings.tool_timeout
            return await asyncio.wait_for(tool(parsed, ToolContext(self, call_id)), timeout)
        except ToolError as exc:
            return ToolResult(str(exc), is_error=True)
        except asyncio.TimeoutError:
            return ToolResult(f"`{tool.name}` timed out.", is_error=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a broken tool must not end the run
            return ToolResult(f"{type(exc).__name__}: {exc}", is_error=True)

    # ---------------------------------------------------------- compaction

    def estimate_prompt_tokens(self) -> int:
        rough = rough_tokens(self.build_prompt())
        if self._last_prompt_tokens and self._last_prompt_tokens[0]:
            actual, rough_then = self._last_prompt_tokens
            return actual + max(0, rough - rough_then)
        return rough

    async def maybe_compact(self, turn: int) -> bool:
        threshold = self.settings.compact_threshold_tokens
        if not threshold or self.estimate_prompt_tokens() < threshold:
            return False
        split = self.memory.split_for_compaction(self.settings.compaction_keep_recent_steps)
        if split is None:
            return False
        old, kept = split
        before = rough_tokens(self.build_prompt())
        request = self.memory.compaction_request(old)
        summary_settings = self.settings.model_copy(
            update={"max_output_tokens": self.settings.compaction_max_tokens})
        response = await self.llm.chat(request, summary_settings)
        summary = (response.message.get("content") or "").strip()
        if not summary:  # fallback keeps the old context retired, not lost silently
            summary = (self.memory.working_memory or "") + "\n(Earlier steps omitted; summary unavailable.)"
        self.memory.apply_compaction(summary, kept)
        after = rough_tokens(self.build_prompt())
        self._last_prompt_tokens = None
        self.compactions.append({"turn": turn, "before": before, "after": after,
                                 "request": request, "summary": summary})
        await self.emit(ev.Compaction(turn=turn, tokens_before=before, tokens_after=after,
                                      summary=summary))
        return True

    # ---------------------------------------------------------- logging

    def save_log(self) -> None:
        if not self.log_dir:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"{self.session_id}.json"
        path.write_text(json.dumps({
            "session": self.session_id,
            "agent": self.name,
            "settings": self.settings.model_dump(),
            "working_memory": self.memory.working_memory,
            "history": self.memory.history,
            "compactions": self.compactions,
            "events": self.transcript,
            "usage": self.total_usage.model_dump(),
        }, indent=2, ensure_ascii=False, default=str))
