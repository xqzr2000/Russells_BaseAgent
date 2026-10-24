"""Jack's fix_code tool and the coordinator's code handling (offline: scripted Jack, fake fixer)."""

from __future__ import annotations

import json

import pytest

from baseagent import AgentSettings, ScriptedLLM
from baseagent.agents.chess_coach import ChessCoachAgent
from baseagent.agents.coordinator import IDENTITY_REPLY, SYSTEM_PROMPT, CoordinatorAgent
from baseagent.codeagent import tool as code_tool
from baseagent.codeagent.opencode_client import CodeAgentError, CodeFix, CodeTask
from baseagent.codeagent.tool import code_blocks, latest_user_message, question_from
from baseagent.workspace import Workspace

BUGGY = "def add_up(n)\n    total = 0\n    for i in range(n):\n        total =+ i\n    return total\n"


def call(name: str, args: dict, call_id: str = "call_0") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def act(*calls: dict) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


class FakeFixer:
    def __init__(self, error: str | None = None):
        self.tasks: list[CodeTask] = []
        self.error = error

    async def fix(self, task: CodeTask, progress=None) -> CodeFix:
        self.tasks.append(task)
        if progress:
            await progress("starting the OpenCode sandbox")
            await progress("running main.py")
        if self.error:
            raise CodeAgentError(self.error)
        return CodeFix(filename=task.filename or "main.py", fence=task.language or "python", runnable=True,
                       mode=task.mode, original=task.code, fixed=task.code.replace("=+", "+="),
                       issues=[{"line": 4, "problem": "'=+' sets total to +i.", "fix": "Use '+='."}],
                       summary="Fixed.", answered=True, model="deepseek-v4.1-flash-free")


@pytest.fixture
def fixer(monkeypatch):
    fake = FakeFixer()
    monkeypatch.setattr(code_tool, "_fixer", fake)
    return fake


def jack(script, tmp_path):
    llm = ScriptedLLM(script)
    events: list[dict] = []

    async def emit(event):
        events.append(event.model_dump())

    agent = CoordinatorAgent(llm, AgentSettings(model="m"), emit=emit,
                             workspace=Workspace(tmp_path / "ws", url_prefix="/files"))
    return agent, llm, events


def test_code_blocks_question_and_latest_message():
    text = f"Why is this wrong?\n```python\n{BUGGY}```\nand this\n```js\nconsole.log(1)\n```"
    assert code_blocks(text) == [("python", BUGGY), ("js", "console.log(1)\n")]
    assert question_from(text + "\n\nAttached file: data/x.py") == "Why is this wrong? and this"
    history = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "ok"},
               {"role": "user", "content": "second"},
               {"role": "user", "content": "Your last response neither called a tool nor finished the task."}]
    assert latest_user_message(history) == "second"


async def test_jack_sends_the_code_block_and_shows_the_answer_verbatim(fixer, tmp_path):
    agent, llm, events = jack([act(call("fix_code", {"mode": "test", "question": "Does this work?"}))], tmp_path)
    result = await agent.run(f"Does this work?\n```python\n{BUGGY}```")
    task = fixer.tasks[0]
    assert task.code == BUGGY and task.language == "python" and task.mode == "test"
    assert task.request == "Does this work?"
    # The code agent's Markdown is the reply, with no second model call.
    assert len(llm.requests) == 1
    end = next(e for e in events if e["type"] == "turn_end")
    assert end["final_text"] == result.final_text
    assert result.final_text.startswith("**What was wrong**\n1. Line 4:")
    assert "total += i" in result.final_text
    progress = [e["message"] for e in events if e["type"] == "tool_progress"]
    assert progress == ["starting the OpenCode sandbox", "running main.py"]


async def test_attached_file_is_read_from_the_workspace(fixer, tmp_path):
    agent, _, _ = jack([act(call("fix_code", {"file": "data/main.py"}))], tmp_path)
    path = tmp_path / "ws" / "data" / "main.py"
    path.parent.mkdir(parents=True)
    path.write_text(BUGGY)
    await agent.run("Fix this please\n\nAttached file: data/main.py")
    task = fixer.tasks[0]
    assert (task.code, task.filename, task.mode) == (BUGGY, "main.py", "auto")
    assert task.request == "Fix this please"


async def test_missing_code_is_a_recoverable_tool_error(fixer, tmp_path):
    agent, llm, _ = jack([act(call("fix_code", {})), {"role": "assistant", "content": "Please paste it."}],
                         tmp_path)
    result = await agent.run("Can you fix my code?")
    assert fixer.tasks == []
    tool_reply = llm.requests[1]["messages"][-1]["content"]
    assert "no ``` code block" in tool_reply
    assert result.final_text == "Please paste it."


async def test_code_agent_failures_go_back_to_jack(monkeypatch, tmp_path):
    monkeypatch.setattr(code_tool, "_fixer", FakeFixer(error="Docker isn't running"))
    agent, llm, _ = jack([act(call("fix_code", {"code": "print(1"})),
                          {"role": "assistant", "content": "The sandbox isn't available right now."}], tmp_path)
    result = await agent.run("print(1 <- fix")
    assert "The code agent couldn't do it: Docker isn't running" in llm.requests[1]["messages"][-1]["content"]
    assert result.final_text == "The sandbox isn't available right now."


async def test_files_outside_the_workspace_are_refused(fixer, tmp_path):
    agent, llm, _ = jack([act(call("fix_code", {"file": "../../etc/passwd"})),
                          {"role": "assistant", "content": "no"}], tmp_path)
    await agent.run("fix ../../etc/passwd")
    assert fixer.tasks == [] and "outside the workspace" in llm.requests[1]["messages"][-1]["content"]


def test_jack_title_identity_and_tool():
    assert CoordinatorAgent.title == "Jack (the coordinator)"
    assert CoordinatorAgent.referred_as() == "Jack" and ChessCoachAgent.referred_as() == "the Chess coach"
    assert IDENTITY_REPLY == "Woof, Jack here, I'm hunting jobs for Russell! You can ask me anything!"
    assert IDENTITY_REPLY in SYSTEM_PROMPT and "fix_code" in SYSTEM_PROMPT
    tool = next(t for t in CoordinatorAgent(ScriptedLLM([]), AgentSettings(model="m")).get_tools()
                if t.name == "fix_code")
    params = tool.schema()["function"]["parameters"]["properties"]
    assert params["mode"]["enum"] == ["review", "auto", "test"]
    assert "code" in params and "file" in params and "block" in params



async def test_identity_questions_skip_the_model(tmp_path):
    agent, llm, events = jack([{"role": "assistant", "content": "Working for Russell."}], tmp_path)
    result = await agent.run("Hey Jack, who are you?")
    assert result.final_text == IDENTITY_REPLY and llm.requests == []
    end = next(e for e in events if e["type"] == "turn_end")
    assert end["final_text"] == IDENTITY_REPLY and end["usage"]["total_tokens"] == 0
    # A longer question isn't a match, and the model sees the earlier exchange.
    await agent.run("Who are you working for?")
    assert len(llm.requests) == 1
    sent = [m["content"] for m in llm.requests[0]["messages"]]
    assert "Hey Jack, who are you?" in sent and IDENTITY_REPLY in sent


def test_identity_matcher():
    from baseagent.agents.coordinator import is_identity_question as q

    assert all(q(m) for m in ["Who are you?", "hi, who are you", "Hey Jack, what’s your name?", "whats your name",
                              "What are you?", "introduce yourself", "who r u", "tell me about yourself"])
    assert not any(q(m) for m in ["who are you working for?", "what are you doing?", "Who are you? Also fix this",
                                  "who are the authors of pandas", "what is your name in chess terms"])


def test_specialist_lists_use_the_short_briefs():
    agent = CoordinatorAgent(ScriptedLLM([]), AgentSettings(model="m"))
    tools = {t.name: t.schema()["function"]["description"] for t in agent.get_tools()}
    assert "- data_science: EDA on a tabular data file" in tools["delegate"]
    assert "- chess_coach: Chess lessons and games" in tools["handoff"]
