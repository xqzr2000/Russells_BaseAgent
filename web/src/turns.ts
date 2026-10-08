// Folds the flat event stream into turns and steps the UI can render.
import type { AgentEvent, Attachment, Usage } from "./types";

export interface ToolRun {
  id: string;
  name: string;
  args: string;
  result?: string;
  isError?: boolean;
  durationMs?: number;
  progress?: string; // latest status line while it runs (e.g. a delegated agent's step)
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
  userFiles: string[]; // files attached to the message, e.g. "data/titanic.csv"
  model: string;
  steps: Step[];
  notes: { kind: "warning" | "error" | "compaction"; text: string; detail?: string }[];
  finalText?: string;
  stopReason?: string;
  usage?: Usage;
  ended: boolean;
}

// The server appends "\n\nAttached file(s): a, b" to a message (see with_attachments in app.py).
const ATTACHED = /\n\nAttached files?: ([^\n]+)$/;

export function splitAttachments(message: string): { text: string; files: string[] } {
  const match = ATTACHED.exec(message);
  if (!match) return { text: message, files: [] };
  return { text: message.slice(0, match.index), files: match[1].split(", ") };
}

/** Every downloadable file the turn's tools produced, newest version of each URL once. */
export function turnFiles(turn: Turn): Attachment[] {
  const byUrl = new Map<string, Attachment>();
  for (const step of turn.steps) {
    for (const run of step.tools) {
      for (const item of run.attachments) {
        if (item.kind === "file" && item.url) byUrl.set(item.url, item);
      }
    }
  }
  return [...byUrl.values()];
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
      const { text, files } = splitAttachments(event.user_message);
      const turn: Turn = {
        index: event.turn,
        userMessage: text,
        userFiles: files,
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
      case "tool_progress": {
        const run = stepOf(turn, event.step).tools.find((t) => t.id === event.call_id && t.result === undefined);
        if (run) run.progress = event.message;
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
