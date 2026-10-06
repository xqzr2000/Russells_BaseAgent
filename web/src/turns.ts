// Folds the flat event stream into turns and steps the UI can render.
import type { AgentEvent, Attachment, Usage } from "./types";

export interface ToolRun {
  id: string;
  name: string;
  args: string;
  result?: string;
  isError?: boolean;
  durationMs?: number;
  attachments: Attachment[];
}

export interface Step {
  index: number;
  text: string;
  tools: ToolRun[];
  hasToolCalls: boolean;
  done: boolean;
  usage?: Usage;
}

export interface Turn {
  index: number;
  userMessage: string;
  model: string;
  steps: Step[];
  notes: { kind: "warning" | "error" | "compaction"; text: string; detail?: string }[];
  finalText?: string;
  stopReason?: string;
  usage?: Usage;
  ended: boolean;
}

export function buildTurns(events: AgentEvent[]): Turn[] {
  const turns: Turn[] = [];
  const byTurn = new Map<number, Turn>();
  const stepOf = (turn: Turn, index: number): Step => {
    let step = turn.steps.find((s) => s.index === index);
    if (!step) {
      step = { index, text: "", tools: [], hasToolCalls: false, done: false };
      turn.steps.push(step);
    }
    return step;
  };

  for (const event of events) {
    if (event.type === "turn_start") {
      const turn: Turn = {
        index: event.turn,
        userMessage: event.user_message,
        model: event.model,
        steps: [],
        notes: [],
        ended: false,
      };
      turns.push(turn);
      byTurn.set(event.turn, turn);
      continue;
    }
    const turn = byTurn.get(event.turn);
    if (!turn) continue;
    switch (event.type) {
      case "step_start":
        stepOf(turn, event.step);
        break;
      case "text_delta":
        stepOf(turn, event.step).text += event.delta;
        break;
      case "assistant_message": {
        const step = stepOf(turn, event.step);
        step.text = event.content;
        step.done = true;
        step.hasToolCalls = event.tool_calls.length > 0;
        for (const call of event.tool_calls) {
          if (!step.tools.some((t) => t.id === call.id)) {
            step.tools.push({ id: call.id, name: call.name, args: call.arguments, attachments: [] });
          }
        }
        break;
      }
      case "tool_start": {
        const step = stepOf(turn, event.step);
        step.hasToolCalls = true;
        if (!step.tools.some((t) => t.id === event.call_id)) {
          step.tools.push({ id: event.call_id, name: event.name, args: event.arguments, attachments: [] });
        }
        break;
      }
      case "tool_end": {
        const step = stepOf(turn, event.step);
        // ids can repeat across steps but not within one, so match inside this step
        const run = step.tools.find((t) => t.id === event.call_id && t.result === undefined);
        if (run) {
          run.result = event.content;
          run.isError = event.is_error;
          run.durationMs = event.duration_ms;
          run.attachments = event.attachments;
        }
        break;
      }
      case "usage":
        stepOf(turn, event.step).usage = event.usage;
        break;
      case "compaction":
        turn.notes.push({
          kind: "compaction",
          text: `Context compacted: about ${event.tokens_before.toLocaleString()} → ${event.tokens_after.toLocaleString()} tokens`,
          detail: event.summary,
        });
        break;
      case "warning":
        turn.notes.push({ kind: "warning", text: event.message });
        break;
      case "error":
        turn.notes.push({ kind: "error", text: event.message });
        break;
      case "turn_end":
        turn.ended = true;
        turn.finalText = event.final_text;
        turn.stopReason = event.stop_reason;
        turn.usage = event.usage;
        break;
    }
  }
  return turns;
}
