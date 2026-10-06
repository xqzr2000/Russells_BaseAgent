"""Small, safe tools that any agent can reuse."""

from __future__ import annotations

import ast
import math
import operator
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from baseagent.tools import ToolContext, ToolError, tool

_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    name: getattr(math, name)
    for name in ("sqrt", "log", "log10", "log2", "exp", "sin", "cos", "tan",
                 "asin", "acos", "atan", "floor", "ceil", "factorial")
} | {"abs": abs, "round": round, "min": min, "max": max}
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10_000:
            raise ToolError("Exponent too large.")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS and not node.keywords):
        return _FUNCTIONS[node.func.id](*(_evaluate(a) for a in node.args))
    raise ToolError(f"Unsupported expression element: {ast.dump(node)[:60]}")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly. Use it instead of mental math.

    Args:
        expression: Python-style arithmetic, e.g. "2 * (3 + 4) ** 2" or "sqrt(2) * pi".
            Supports + - * / // % **, sqrt, log, exp, trig, floor, ceil, abs, round,
            min, max, factorial, pi, e.
    """
    try:
        tree = ast.parse(expression.replace("^", "**"), mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"Could not parse {expression!r}: {exc.msg}") from exc
    try:
        value = _evaluate(tree)
    except ZeroDivisionError:
        raise ToolError("Division by zero.") from None
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        value = int(value)
    return str(value)


@tool
def current_time(timezone: str = "America/Vancouver") -> str:
    """Get the current date and time.

    Args:
        timezone: IANA timezone name, e.g. "America/Vancouver", "UTC", "Asia/Tokyo".
    """
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ToolError(f"Unknown timezone {timezone!r}.") from exc
    return datetime.now(zone).strftime("%A %Y-%m-%d %H:%M:%S %Z (UTC%z)")


@tool
def save_note(note: str, ctx: ToolContext) -> str:
    """Save a short note to remember for the rest of this chat session.

    Args:
        note: The fact or reminder to keep.
    """
    notes = ctx.state.setdefault("notes", [])
    notes.append(note)
    return f"Saved note #{len(notes)}."


@tool
def list_notes(ctx: ToolContext) -> str:
    """List the notes saved in this chat session."""
    notes = ctx.state.get("notes", [])
    return "\n".join(f"{i}. {n}" for i, n in enumerate(notes, 1)) or "(no notes yet)"
