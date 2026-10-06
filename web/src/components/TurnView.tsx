import { useState } from "react";
import type { Attachment } from "../types";
import type { Step, ToolRun, Turn } from "../turns";
import { Markdown } from "./Markdown";

function prettyArgs(raw: string): string {
  try {
    return JSON.stringify(JSON.parse(raw || "{}"), null, 2);
  } catch {
    return raw;
  }
}

function argsPreview(raw: string): string {
  try {
    const parsed = JSON.parse(raw || "{}");
    const parts = Object.entries(parsed).map(([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`);
    const text = parts.join(", ");
    return text.length > 90 ? `${text.slice(0, 87)}...` : text;
  } catch {
    return raw.slice(0, 90);
  }
}

function AttachmentView({ item }: { item: Attachment }) {
  if (item.kind === "image") {
    const src = item.url ?? `data:${item.mime};base64,${item.data}`;
    return (
      <figure className="attachment">
        <img src={src} alt={item.title ?? "Chart produced by the tool"} />
        {item.title && <figcaption>{item.title}</figcaption>}
      </figure>
    );
  }
  if (item.kind === "table") {
    return item.mime === "text/html" ? (
      <div className="attachment table-html" dangerouslySetInnerHTML={{ __html: item.data }} />
    ) : (
      <div className="attachment">
        <Markdown text={item.data} />
      </div>
    );
  }
  if (item.kind === "file" && item.url) {
    return (
      <a className="attachment file" href={item.url} download>
        Download {item.title ?? "file"}
      </a>
    );
  }
  return <pre className="attachment code">{item.data}</pre>;
}

function ToolRow({ run }: { run: ToolRun }) {
  const [open, setOpen] = useState(false);
  const running = run.result === undefined;
  const status = running ? "running" : run.isError ? "error" : "ok";
  return (
    <div className={`tool tool-${status}`}>
      <button className="tool-head" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <span className="tool-name">{run.name}</span>
        <span className="tool-args">{argsPreview(run.args)}</span>
        <span className="tool-status">
          {running ? "running" : run.isError ? "failed" : `${run.durationMs ?? 0} ms`}
        </span>
      </button>
      {open && (
        <div className="tool-body">
          <div className="tool-label">Arguments</div>
          <pre className="code">{prettyArgs(run.args)}</pre>
          <div className="tool-label">{run.isError ? "Error returned to the model" : "Result returned to the model"}</div>
          <pre className="code">{running ? "Waiting for the tool..." : run.result}</pre>
        </div>
      )}
      {run.attachments.length > 0 && (
        <div className="attachments">
          {run.attachments.map((a, i) => (
            <AttachmentView key={i} item={a} />
          ))}
        </div>
      )}
    </div>
  );
}

function TraceStep({ step, live }: { step: Step; live: boolean }) {
  const active = live && (!step.done || step.tools.some((t) => t.result === undefined));
  return (
    <li className={`trace-step ${active ? "is-active" : ""} ${step.hasToolCalls ? "has-tools" : ""}`}>
      <span className="trace-node" aria-hidden />
      <div className="trace-content">
        <div className="trace-title">Step {step.index}</div>
        {step.text && <div className="trace-note">{step.text}</div>}
        {!step.text && !step.hasToolCalls && active && <div className="trace-note muted">Thinking...</div>}
        {step.tools.map((run, i) => (
          <ToolRow key={`${run.id}-${i}`} run={run} />
        ))}
      </div>
    </li>
  );
}

const STOP_MESSAGES: Record<string, string> = {
  step_limit: "Stopped at the step limit. Raise Max steps in Settings or ask the agent to continue.",
  cancelled: "Stopped by you.",
  error: "The turn failed.",
};

export function TurnView({ turn, live }: { turn: Turn; live: boolean }) {
  const lastStep = turn.steps[turn.steps.length - 1];
  // Steps that called tools form the trace; a step without tool calls is the answer.
  const traced = turn.steps.filter((s) => s.hasToolCalls || (s !== lastStep && s.text));
  const answerStep = lastStep && !lastStep.hasToolCalls ? lastStep : undefined;
  const answer = turn.ended
    ? turn.stopReason === "final"
      ? turn.finalText ?? ""
      : ""
    : answerStep?.text ?? "";
  const showThinkingNode = live && !turn.ended && (!lastStep || (lastStep.done && lastStep.hasToolCalls));
  const steps = turn.usage ? turn.steps.length : undefined;

  return (
    <article className="turn">
      <div className="user-msg">
        <div className="user-bubble">{turn.userMessage}</div>
      </div>
      <div className="agent-msg">
        {(traced.length > 0 || showThinkingNode) && (
          <ol className="trace">
            {traced.map((step) => (
              <TraceStep key={step.index} step={step} live={live && !turn.ended} />
            ))}
            {showThinkingNode && (
              <li className="trace-step is-active">
                <span className="trace-node" aria-hidden />
                <div className="trace-content">
                  <div className="trace-note muted">Waiting for the model...</div>
                </div>
              </li>
            )}
          </ol>
        )}
        {turn.notes.map((note, i) => (
          <div key={i} className={`note note-${note.kind}`}>
            {note.text}
            {note.detail && (
              <details>
                <summary>Show working memory</summary>
                <Markdown text={note.detail} />
              </details>
            )}
          </div>
        ))}
        {answer && (
          <div className={`answer ${!turn.ended ? "is-streaming" : ""}`}>
            <Markdown text={answer} />
          </div>
        )}
        {turn.ended && turn.stopReason !== "final" && (
          <div className={`note note-${turn.stopReason === "error" ? "error" : "warning"}`}>
            {STOP_MESSAGES[turn.stopReason ?? ""] ?? turn.finalText}
            {turn.stopReason === "error" && turn.finalText && <div className="muted">{turn.finalText}</div>}
          </div>
        )}
        {turn.ended && turn.usage && (
          <div className="turn-meta">
            {steps} {steps === 1 ? "step" : "steps"}
            {turn.usage.total_tokens > 0 && `, ${turn.usage.total_tokens.toLocaleString()} tokens`}
            {turn.usage.reasoning_tokens > 0 && ` (${turn.usage.reasoning_tokens.toLocaleString()} reasoning)`}, {turn.model}
          </div>
        )}
      </div>
    </article>
  );
}
