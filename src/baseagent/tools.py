"""Tools: a schema the model reads plus a handler the agent runs.

Write a tool as a plain (sync or async) function with type hints and a
docstring, and decorate it with ``@tool``::

    @tool
    def calculator(expression: str) -> str:
        \"\"\"Evaluate an arithmetic expression.

        Args:
            expression: e.g. "2 * (3 + 4)".
        \"\"\"

The decorator builds a Pydantic model from the signature. That one model gives
the JSON Schema sent to the model AND validates the model's arguments before
the function runs, so the schema and the handler cannot drift apart.

A parameter annotated ``ToolContext`` is injected by the agent and hidden from
the model. Return a ``str`` or a ``ToolResult`` (which can carry UI
attachments such as charts).
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from baseagent.events import Attachment

if TYPE_CHECKING:
    from baseagent.agent import BaseAgent


@dataclass
class ToolContext:
    """What a tool may use from the agent that is running it."""

    agent: "BaseAgent"
    call_id: str

    @property
    def state(self) -> dict[str, Any]:
        """Per-agent scratch space that persists across turns."""
        return self.agent.state


@dataclass
class ToolResult:
    content: str  # what the model sees
    attachments: list[Attachment] = field(default_factory=list)  # what the UI shows
    is_error: bool = False


class ToolError(Exception):
    """Raise inside a tool to report a recoverable problem to the model."""


Handler = Callable[..., Any | Awaitable[Any]]


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Handler
    context_param: str | None = None
    # At most one exclusive tool runs per model response (e.g. a tool that
    # changes shared state, like playing a chess move).
    exclusive: bool = False
    timeout: float | None = None

    def schema(self) -> dict[str, Any]:
        params = self.args_model.model_json_schema()
        params.pop("title", None)
        for prop in params.get("properties", {}).values():
            prop.pop("title", None)
        params.setdefault("properties", {})
        params["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }

    def parse(self, raw_arguments: str | dict | None) -> BaseModel:
        """Decode and validate arguments; raises ToolError with a readable reason."""

        if isinstance(raw_arguments, dict):
            data = raw_arguments
        elif raw_arguments is None or not str(raw_arguments).strip():
            data = {}
        else:
            try:
                data = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise ToolError(f"Arguments are not valid JSON ({exc.msg}).") from exc
        if not isinstance(data, dict):
            raise ToolError("Arguments must be a JSON object.")
        try:
            return self.args_model.model_validate(data)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or 'arguments'}: {err['msg']}"
                for err in exc.errors()
            )
            raise ToolError(f"Invalid arguments: {problems}.") from exc

    async def __call__(self, arguments: BaseModel, ctx: ToolContext) -> ToolResult:
        kwargs = arguments.model_dump()
        if self.context_param:
            kwargs[self.context_param] = ctx
        result = self.handler(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, ToolResult):
            return result
        if result is None:
            return ToolResult(content="(no output)")
        if isinstance(result, str):
            return ToolResult(content=result)
        return ToolResult(content=json.dumps(result, default=str, ensure_ascii=False))


_ARGS_HEADER = re.compile(r"^\s*(Args|Arguments|Parameters):\s*$", re.IGNORECASE)


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into a description and per-arg help."""

    lines = inspect.cleandoc(doc or "").splitlines()
    description: list[str] = []
    args: dict[str, str] = {}
    current: str | None = None
    in_args = False
    for line in lines:
        if _ARGS_HEADER.match(line):
            in_args = True
            continue
        if in_args:
            if re.match(r"^\S", line) and line.rstrip().endswith(":"):
                in_args = False  # e.g. "Returns:"
                continue
            match = re.match(r"^\s*(\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)$", line)
            if match:
                current = match.group(1)
                args[current] = match.group(2).strip()
            elif current and line.strip():
                args[current] += " " + line.strip()
            continue
        description.append(line)
    return "\n".join(description).strip(), args


def tool(
    func: Handler | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    exclusive: bool = False,
    timeout: float | None = None,
):
    """Turn a typed function into a ``Tool``. Usable as ``@tool`` or ``@tool(...)``."""

    def build(fn: Handler) -> Tool:
        signature = inspect.signature(fn)
        doc_description, arg_help = _parse_docstring(fn.__doc__ or "")
        fields: dict[str, Any] = {}
        context_param = None
        hints = _resolve_hints(fn)
        for param_name, param in signature.parameters.items():
            annotation = hints.get(param_name, param.annotation)
            if annotation is ToolContext or getattr(annotation, "__name__", "") == "ToolContext":
                context_param = param_name
                continue
            if annotation is inspect.Parameter.empty:
                annotation = str
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[param_name] = (
                annotation,
                Field(default=default, description=arg_help.get(param_name)),
            )
        args_model = create_model(
            f"{fn.__name__}_args",
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )
        return Tool(
            name=name or fn.__name__,
            description=description or doc_description or fn.__name__,
            args_model=args_model,
            handler=fn,
            context_param=context_param,
            exclusive=exclusive,
            timeout=timeout,
        )

    return build(func) if func is not None else build


def _resolve_hints(fn: Handler) -> dict[str, Any]:
    try:
        import typing

        return typing.get_type_hints(fn)
    except Exception:  # noqa: BLE001 - fall back to raw annotations
        return {}
