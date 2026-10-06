"""Typed events an agent emits while it works.

Everything outside the agent (the web UI, the trajectory log, tests) learns
what happened by listening to these. The agent never imports UI code; the
server turns each event into one Server-Sent Event, and the TypeScript UI
mirrors these models in ``web/src/types.ts``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class Attachment(BaseModel):
    """Rich output for the UI (not sent to the model): an image, table, or file.

    The EDA agent will use this for seaborn charts (``image/png``) and
    dataframe previews (``text/markdown`` or ``text/html``).
    """

    kind: Literal["image", "table", "file", "text"]
    mime: str = "text/plain"
    title: str | None = None
    data: str = ""  # base64 for images, text for tables/text
    url: str | None = None  # for files served by the server


class ToolCallInfo(BaseModel):
    id: str
    name: str
    arguments: str


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(**{k: getattr(self, k) + getattr(other, k) for k in Usage.model_fields})


class _Event(BaseModel):
    turn: int = 0


class TurnStart(_Event):
    type: Literal["turn_start"] = "turn_start"
    user_message: str
    model: str
    agent: str


class StepStart(_Event):
    type: Literal["step_start"] = "step_start"
    step: int


class TextDelta(_Event):
    type: Literal["text_delta"] = "text_delta"
    step: int
    delta: str


class AssistantMessage(_Event):
    type: Literal["assistant_message"] = "assistant_message"
    step: int
    content: str
    tool_calls: list[ToolCallInfo] = Field(default_factory=list)


class ToolStart(_Event):
    type: Literal["tool_start"] = "tool_start"
    step: int
    call_id: str
    name: str
    arguments: str


class ToolEnd(_Event):
    type: Literal["tool_end"] = "tool_end"
    step: int
    call_id: str
    name: str
    content: str  # exactly what the model will see
    is_error: bool = False
    duration_ms: int = 0
    attachments: list[Attachment] = Field(default_factory=list)


class UsageEvent(_Event):
    type: Literal["usage"] = "usage"
    step: int
    usage: Usage


class Compaction(_Event):
    type: Literal["compaction"] = "compaction"
    tokens_before: int
    tokens_after: int
    summary: str


class Warning(_Event):  # noqa: A001 - mirrors the event name used in the UI
    type: Literal["warning"] = "warning"
    message: str


class Error(_Event):
    type: Literal["error"] = "error"
    message: str


class TurnEnd(_Event):
    type: Literal["turn_end"] = "turn_end"
    final_text: str
    steps: int
    stop_reason: Literal["final", "step_limit", "cancelled", "error"]
    usage: Usage


AgentEvent = Annotated[
    TurnStart
    | StepStart
    | TextDelta
    | AssistantMessage
    | ToolStart
    | ToolEnd
    | UsageEvent
    | Compaction
    | Warning
    | Error
    | TurnEnd,
    Field(discriminator="type"),
]


def event_to_dict(event: BaseModel) -> dict[str, Any]:
    return event.model_dump(mode="json")
