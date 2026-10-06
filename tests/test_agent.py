"""Offline tests for the BaseAgent loop, using a scripted stand-in model."""

from __future__ import annotations

import asyncio
import json

import pytest

from baseagent import AgentSettings, BaseAgent, ScriptedLLM, ToolResult, tool
from baseagent.agents import discover, get_agent_class
from baseagent.agents.general import GeneralAgent
from baseagent.events import Attachment


def call(name: str, args: dict | str, call_id: str = "call_0") -> dict:
    arguments = args if isinstance(args, str) else json.dumps(args)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def act(*calls: dict, content: str = "") -> dict:
    return {"role": "assistant", "content": content, "tool_calls": list(calls)}


def reply(text: str) -> dict:
    return {"role": "assistant", "content": text}


@tool
def echo(text: str) -> str:
    """Echo text back.

    Args:
        text: what to echo
    """
    return f"echo:{text}"


@tool(exclusive=True)
def move(square: str) -> str:
    """Change shared state."""
    return f"moved {square}"


@tool
def chart() -> ToolResult:
    """Return a fake chart."""
    return ToolResult("chart made", attachments=[Attachment(kind="image", mime="image/png", data="AAA")])


class EchoAgent(BaseAgent):
    name = "echo-test"

    def get_tools(self):
        return [echo, move, chart]


def make(script, summaries=None, **settings):
    llm = ScriptedLLM(script, summaries)
    events = []

    async def emit(event):
        events.append(event.model_dump())

    agent = EchoAgent(llm, AgentSettings(**settings), emit=emit)
    return agent, llm, events


async def test_tool_round_trip_and_final_reply():
    agent, llm, events = make([act(call("echo", {"text": "hi"})), reply("done")])
    result = await agent.run("say hi")

    assert result.final_text == "done" and result.stop_reason == "final" and result.steps == 2
    second = llm.requests[1]["messages"]
    assert second[0]["role"] == "system"
    assert {"role": "user", "content": "say hi"} in second
    tool_msg = next(m for m in second if m["role"] == "tool")
    assert tool_msg == {"role": "tool", "tool_call_id": "call_0", "content": "echo:hi"}
    types = [e["type"] for e in events]
    assert types[0] == "turn_start" and types[-1] == "turn_end"
    assert "tool_start" in types and "tool_end" in types


async def test_schema_comes_from_type_hints_and_docstring():
    schema = echo.schema()["function"]
    assert schema["name"] == "echo"
    assert schema["description"] == "Echo text back."
    assert schema["parameters"]["properties"]["text"] == {"type": "string", "description": "what to echo"}
    assert schema["parameters"]["required"] == ["text"]
    assert schema["parameters"]["additionalProperties"] is False


@pytest.mark.parametrize("bad", ["{not json", json.dumps([1]), json.dumps({}),
                                 json.dumps({"text": "a", "extra": 1})])
async def test_bad_arguments_are_recoverable(bad):
    agent, llm, _ = make([act(call("echo", bad)), reply("ok")])
    result = await agent.run("x")
    assert result.stop_reason == "final"
    tool_msg = next(m for m in llm.requests[1]["messages"] if m["role"] == "tool")
    assert tool_msg["content"].startswith("Error:")


async def test_unknown_tool_and_tool_exception_are_recoverable():
    @tool
    def boom() -> str:
        """Fails."""
        raise RuntimeError("kaboom")

    class BoomAgent(BaseAgent):
        def get_tools(self):
            return [boom]

    llm = ScriptedLLM([act(call("nope", {}), call("boom", {}, "call_1")), reply("ok")])
    agent = BoomAgent(llm)
    await agent.run("x")
    tools = [m for m in llm.requests[1]["messages"] if m["role"] == "tool"]
    assert "Unknown tool" in tools[0]["content"]
    assert "RuntimeError: kaboom" in tools[1]["content"]


async def test_repeated_call_ids_are_linked_per_step():
    agent, llm, _ = make([act(call("echo", {"text": "a"})), act(call("echo", {"text": "b"})), reply("ok")])
    await agent.run("x")
    tools = [m for m in llm.requests[2]["messages"] if m["role"] == "tool"]
    assert [t["content"] for t in tools] == ["echo:a", "echo:b"]


async def test_only_one_exclusive_call_runs_per_response():
    agent, llm, _ = make([act(call("move", {"square": "e4"}), call("move", {"square": "d4"}, "c2")), reply("ok")])
    await agent.run("x")
    tools = [m for m in llm.requests[1]["messages"] if m["role"] == "tool"]
    assert tools[0]["content"] == "moved e4"
    assert tools[1]["content"].startswith("Error:")


async def test_step_limit_stops_the_turn():
    agent, llm, _ = make([act(call("echo", {"text": str(i)})) for i in range(5)], max_steps=3)
    result = await agent.run("loop")
    assert result.stop_reason == "step_limit" and len(llm.requests) == 3


async def test_long_observation_is_truncated():
    @tool
    def big() -> str:
        """Big output."""
        return "a" * 6000 + "z" * 6000

    class BigAgent(BaseAgent):
        def get_tools(self):
            return [big]

    llm = ScriptedLLM([act(call("big", {})), reply("ok")])
    await BigAgent(llm).run("x")
    content = next(m for m in llm.requests[1]["messages"] if m["role"] == "tool")["content"]
    assert len(content) <= 10_000 and content.startswith("a") and content.endswith("z")


async def test_conversation_persists_across_turns():
    agent, llm, _ = make([reply("first"), reply("second")])
    await agent.run("one")
    await agent.run("two")
    contents = [m["content"] for m in llm.requests[1]["messages"]]
    assert "one" in contents and "first" in contents and "two" in contents


async def test_attachments_reach_events_not_the_model():
    agent, llm, events = make([act(call("chart", {})), reply("ok")])
    await agent.run("plot")
    end = next(e for e in events if e["type"] == "tool_end")
    assert end["attachments"][0]["mime"] == "image/png"
    assert "AAA" not in json.dumps(llm.requests[1]["messages"])


async def test_cancel_leaves_history_valid():
    @tool
    async def slow() -> str:
        """Slow."""
        await asyncio.sleep(10)
        return "never"

    class SlowAgent(BaseAgent):
        def get_tools(self):
            return [slow]

    agent = SlowAgent(ScriptedLLM([act(call("slow", {}))]))
    task = asyncio.create_task(agent.run("x"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    history = agent.memory.history
    assert history[-1]["role"] == "tool" and history[-1]["tool_call_id"] == "call_0"
    assert agent.transcript[-1]["stop_reason"] == "cancelled"


async def test_compaction_summarizes_old_steps_and_keeps_latest_request():
    script = [act(call("echo", {"text": "x" * 4000}), content="look")] * 3 + [reply("final")]
    agent, llm, events = make(script, summaries=["MEMORY: echoed big text"] * 3,
                              compact_threshold_tokens=1500, compaction_keep_recent_steps=1)
    result = await agent.run("please echo")
    assert result.final_text == "final"
    assert any(e["type"] == "compaction" for e in events)
    summary_request = next(r for r in llm.requests if not r["tools"])
    assert "x" * 100 in json.dumps(summary_request["messages"])
    last_prompt = json.dumps(llm.requests[-1]["messages"])
    assert "MEMORY: echoed big text" in last_prompt
    assert "please echo" in last_prompt  # the current request is kept verbatim
    event = next(e for e in events if e["type"] == "compaction")
    assert event["tokens_after"] < event["tokens_before"]
    # every tool message still follows its call
    msgs = llm.requests[-1]["messages"]
    for i, m in enumerate(msgs):
        if m["role"] == "tool":
            prev = next(p for p in reversed(msgs[:i]) if p["role"] != "tool")
            assert m["tool_call_id"] in {c["id"] for c in prev.get("tool_calls", [])}


async def test_general_agent_is_registered_with_skills():
    agents = discover()
    assert get_agent_class("general") is GeneralAgent and "general" in agents
    llm = ScriptedLLM([act(call("calculator", {"expression": "12*(3+4)"})),
                       act(call("invoke_skill", {"name": "tutor-explanations"})), reply("84")])
    agent = GeneralAgent(llm)
    await agent.run("what is 12*(3+4)? teach me")
    prompt = llm.requests[0]["messages"][0]["content"]
    assert "tutor-explanations" in prompt and "check-your-understanding" not in prompt
    tools = [m["content"] for m in llm.requests[2]["messages"] if m["role"] == "tool"]
    assert tools[0] == "84" and "check-your-understanding" in tools[1]


async def test_setup_runs_once_and_subclass_state_is_ready():
    class Stateful(BaseAgent):
        setups = 0

        def __init__(self, *a, mode="x", **kw):
            super().__init__(*a, **kw)
            self.mode = mode

        async def setup(self):
            type(self).setups += 1
            self.resource = f"ready-{self.mode}"

        def system_prompt(self):
            return f"mode={self.mode} {self.resource}"

    llm = ScriptedLLM([reply("a"), reply("b")])
    agent = Stateful(llm, mode="eda")
    await agent.run("1")
    await agent.run("2")
    assert Stateful.setups == 1
    assert llm.requests[0]["messages"][0]["content"] == "mode=eda ready-eda"


async def test_fake_model_uses_calculator():
    from baseagent.llm import FAKE_MODEL, FakeLLM

    agent = GeneralAgent(FakeLLM(), GeneralAgent.make_settings(model=FAKE_MODEL))
    result = await agent.run("compute 12*(3+4) please")
    assert "84" in result.final_text


async def test_template_agent_works_when_copied():
    from baseagent.agents import AGENTS, discover
    from baseagent.agents._template import TemplateAgent

    assert "template" not in discover()  # underscore files are not auto-registered
    llm = ScriptedLLM([act(call("example_tool", {"query": "titanic"}),
                           call("example_rich_tool", {"title": "T"}, "c2")), reply("ok")])
    events = []

    async def emit(e):
        events.append(e.model_dump())

    agent = TemplateAgent(llm, emit=emit, workdir="/tmp")
    result = await agent.run("go")
    assert result.final_text == "ok" and agent.resource == "ready" and agent.state["calls"] == 1
    ends = [e for e in events if e["type"] == "tool_end"]
    assert ends[0]["content"] == "Looked up 'titanic' (limit 5)."
    assert ends[1]["attachments"][0]["kind"] == "table"
    assert "template" not in AGENTS
