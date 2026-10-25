"""The code agent's answer, file naming and progress lines (offline)."""

from __future__ import annotations

from baseagent.codeagent.answer import build_answer, verification
from baseagent.codeagent.bridge import code_sha256
from baseagent.codeagent.opencode_client import CodeFix, CodeTask, describe_tool, resolve_file, task_message
from baseagent.codeagent.runner import RunResult

ORIGINAL = "def add_up(n)\n    total = 0\n    for i in range(n):\n        total =+ i\n    return total\n"
FIXED = ("def add_up(n):  # FIX 1: added ':'\n    total = 0\n    for i in range(n):\n"
         "        total += i  # FIX 2: '=+' -> '+='\n    return total\nprint(add_up(10))\n")
# Russell's first chat-room test: JavaScript pasted without ``` fences.
CHAR_COUNT = """// ===== YOUR TASK =====
// Show how many characters remain as the user types in the username field.
const inputField = document.querySelector('input#username-field');
const display = document.querySelector('p#char-count');

const maxChars = parseInt(inputField.getAttribute('maxlength'));

function updateRemainingCharacters() {
    display.textContent = "Remaining characters:", maxChars - inputField.value.length;
}
inputField.addEventListener('input', updateRemainingCharacters);
"""
ISSUES = [{"line": 1, "problem": "Missing ':' after the def line.", "fix": "Added ':'."},
          {"line": 4, "problem": "'=+' sets total to +i.", "fix": "Changed it to '+='."}]


def run(mode: str, status: str, code: str, stdout: str = "", stderr: str = "") -> RunResult:
    return RunResult("python", mode, status, exit_code=0 if status == "ok" else 1, stdout=stdout,
                     stderr=stderr, meta={"code_sha256": code_sha256(code), "timeout": 10})


def fix(**kw) -> CodeFix:
    base = dict(filename="main.py", fence="python", runnable=True, mode="auto", original=ORIGINAL,
                fixed=FIXED, issues=ISSUES, summary="Two bugs fixed.", answered=True,
                model="deepseek-v4.1-flash-free")
    base.update(kw)
    return CodeFix(**base)


def test_full_answer_layout():
    answer = build_answer(fix(runs=[run("check", "ok", FIXED), run("run", "ok", FIXED, "45\n")]))
    assert answer.startswith("**What was wrong**\n1. Line 1: Missing ':' after the def line. Fix: Added ':'.")
    assert "2. Line 4:" in answer
    assert "**Fixed code** (`main.py`)\n```python\ndef add_up(n):  # FIX 1" in answer
    assert "**Ran:** printed `45`" in answer
    assert answer.rstrip().endswith("*Fixed by OpenCode with the free model deepseek-v4.1-flash-free.*")
    assert answer.index("What was wrong") < answer.index("Fixed code") < answer.index("Ran:")


def test_checked_only_gives_the_reason():
    line = verification(fix(runs=[run("check", "ok", FIXED)], not_run_reason="It reads data.csv."))
    assert line == "**Checked only:** no syntax or type errors. Not run: It reads data.csv."


def test_runs_on_other_versions_dont_count():
    line = verification(fix(runs=[run("run", "ok", ORIGINAL, "oops")]))
    assert line == "**Reviewed only:** an earlier version was run, but not the final one."


def test_review_mode_and_unrunnable_languages():
    assert verification(fix(mode="review")) == "**Reviewed only:** not run, as you asked."
    line = verification(fix(runnable=False, fence="java", filename="Main.java"))
    assert line.startswith("**Reviewed only:** java can't be run here")


def test_fixed_code_that_still_fails_or_loops():
    failing = verification(fix(runs=[run("run", "error", FIXED, stderr="Traceback\nNameError: name 'x'")]))
    assert failing == "**Ran:** it still fails: `NameError: name 'x'`"
    looping = verification(fix(runs=[run("run", "timeout", FIXED)]))
    assert "stopped after 10 s" in looping


def test_multi_line_output_and_no_original_code_note():
    runs = [run("run", "too_much_output", ORIGINAL), run("run", "ok", FIXED, "0\n1\n2\n")]
    answer = build_answer(fix(runs=runs))
    assert "**Ran:** printed \n```text\n0\n1\n2\n```" in answer
    assert "original code" not in answer.lower()


def test_no_bugs_found():
    answer = build_answer(fix(fixed=ORIGINAL, issues=[], summary="The code is correct.",
                              runs=[run("run", "ok", ORIGINAL, "ok")]))
    assert answer.startswith("**No bugs found.** The code is correct.")
    assert "Fixed code" not in answer and "**Ran:** printed `ok`" in answer


def test_time_limit_and_training_models_are_flagged():
    answer = build_answer(fix(stopped="time limit", zero_retention=False, model="big-pickle"))
    assert answer.startswith("*OpenCode hit the 3-minute limit")
    assert "may use what it's sent for training" in answer


def test_resolve_file():
    assert resolve_file(CodeTask("x", filename="My Script.py")) == ("My_Script.py", "python", True)
    assert resolve_file(CodeTask("x", filename="../../etc/passwd.js")) == ("passwd.js", "javascript", True)
    assert resolve_file(CodeTask("x", language="ts")) == ("main.ts", "typescript", True)
    assert resolve_file(CodeTask("x", language="java")) == ("main.java", "java", False)
    assert resolve_file(CodeTask("x")) == ("main.txt", "", False)  # unknown: never assume Python
    assert resolve_file(CodeTask(CHAR_COUNT)) == ("main.js", "javascript", True)
    assert resolve_file(CodeTask(ORIGINAL)) == ("main.py", "python", True)


def test_task_message_includes_the_code_and_request():
    message = task_message(CodeTask("print(1", request="Why does this fail?"), "main.py", "python", True)
    assert message.startswith("The user asked: Why does this fail?")
    assert "saved as main.py" in message and "```python\nprint(1\n```" in message
    assert "can't be run here" in task_message(CodeTask("x"), "a.go", "go", False)


def test_progress_lines():
    def part(tool, status, **state):
        return {"tool": tool, "state": {"status": status, **state}}

    assert describe_tool(part("read", "running", input={"filePath": "/workspace/main.py"}), "main.py") \
        == "reading main.py"
    assert describe_tool(part("read", "pending"), "main.py") is None
    assert describe_tool(part("run_code", "running", input={"path": "main.py", "mode": "run"}), "main.py") \
        == "running main.py"
    done = part("run_code", "completed", input={"mode": "run"}, metadata={"status": "timeout"})
    assert describe_tool(done, "main.py") == "run result: stopped: too slow (10 s limit)"
    assert describe_tool(part("StructuredOutput", "running"), "main.py") == "writing the explanation"
    assert describe_tool(part("bash", "running"), "main.py") is None


def test_failure_without_answer_says_nothing_was_done():
    answer = build_answer(fix(fixed=ORIGINAL, issues=[], answered=False, summary="",
                              stopped="Cannot connect to API"))
    assert answer == ("*OpenCode couldn't finish (Cannot connect to API).* Your code wasn't changed. "
                      "Free models are sometimes busy, so try again in a minute.")


def test_unchanged_code_is_credited_as_checked():
    answer = build_answer(fix(fixed=ORIGINAL, issues=[], summary="Fine.", runs=[run("run", "ok", ORIGINAL, "x")]))
    assert answer.rstrip().endswith("*Checked by OpenCode with the free model deepseek-v4.1-flash-free.*")


def test_tips_are_listed_after_the_verification_line():
    answer = build_answer(fix(tips=["`=+` is `=` followed by unary `+`.", "`range(n)` stops before n."],
                              runs=[run("run", "ok", FIXED, "45\n")]))
    assert answer.index("**Ran:**") < answer.index("**Tips**\n- `=+` is `=` followed by unary `+`.\n- `range(n)`")
    assert "**Tips**" not in build_answer(fix())


def test_language_guesses():
    from baseagent.codeagent.language import guess_language as g

    assert g(CHAR_COUNT) == "javascript"
    assert g(ORIGINAL) == "python"  # even with the missing ':'
    assert g("i = 0\nwhile i < 3:\n    print(i)\n") == "python"
    assert g("function add(a: number, b: number): number {\n  return a + b;\n}") == "typescript"
    assert g('public class Main { public static void main(String[] args) { System.out.println(1); } }') == "java"
    assert g("#include <stdio.h>\nint main(void) { printf(\"hi\"); }") == "c"
    assert g("package main\nfunc main() { x := 1 }") == "go"
    assert g("check this please, something is wrong with my loop") is None
