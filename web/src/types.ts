// Mirrors src/baseagent/events.py and the server's JSON. Keep the two in sync.

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cached_tokens: number;
  reasoning_tokens: number;
}

export interface Attachment {
  kind: "image" | "table" | "file" | "text";
  mime: string;
  title: string | null;
  data: string;
  url: string | null;
}

export interface ToolCallInfo {
  id: string;
  name: string;
  arguments: string;
}

type Base = { turn: number };

export type AgentEvent =
  | (Base & { type: "turn_start"; user_message: string; model: string; agent: string })
  | (Base & { type: "step_start"; step: number })
  | (Base & { type: "text_delta"; step: number; delta: string })
  | (Base & { type: "assistant_message"; step: number; content: string; tool_calls: ToolCallInfo[] })
  | (Base & { type: "tool_start"; step: number; call_id: string; name: string; arguments: string })
  | (Base & { type: "tool_progress"; step: number; call_id: string; message: string })
  | (Base & {
      type: "tool_end";
      step: number;
      call_id: string;
      name: string;
      content: string;
      is_error: boolean;
      duration_ms: number;
      attachments: Attachment[];
    })
  | (Base & { type: "usage"; step: number; usage: Usage })
  | (Base & { type: "compaction"; tokens_before: number; tokens_after: number; summary: string })
  | (Base & { type: "warning"; message: string })
  | (Base & { type: "error"; message: string })
  | (Base & {
      type: "turn_end";
      final_text: string;
      steps: number;
      stop_reason: "final" | "step_limit" | "cancelled" | "error";
      usage: Usage;
    });

export interface AgentSettings {
  model: string;
  reasoning_effort: string | null;
  temperature: number | null;
  max_output_tokens: number;
  max_steps: number;
  compact_threshold_tokens: number | null;
  compaction_keep_recent_steps: number;
  compaction_max_tokens: number;
  tool_timeout: number;
}

export interface AgentInfo {
  name: string;
  title: string;
  description: string;
  examples: string[];
  default_settings: AgentSettings;
}

export interface SessionSummary {
  id: string;
  agent: string;
  title: string;
  created: number;
  updated: number;
  busy: boolean;
}

export interface SessionDetail extends SessionSummary {
  settings: AgentSettings;
  events: AgentEvent[];
  usage: Usage;
}

export interface ServerConfig {
  openai_key_configured: boolean;
  base_url_host: string;
  default_model: string;
  default_agent: string;
  fake_model: string;
  suggested_models: string[];
}

/** A file uploaded into the chat's workspace, waiting to be sent with a message. */
export interface UploadedFile {
  name: string;
  path: string; // workspace-relative, e.g. "data/titanic.csv"
  size: number;
}
