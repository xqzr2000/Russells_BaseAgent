import type {
  AgentEvent,
  AgentInfo,
  AgentSettings,
  ServerConfig,
  SessionDetail,
  SessionSummary,
} from "./types";

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep statusText */
    }
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  config: () => json<ServerConfig>("/api/config"),
  agents: () => json<AgentInfo[]>("/api/agents"),
  models: (refresh = false) =>
    json<{ models: string[]; error: string | null }>(`/api/models${refresh ? "?refresh=true" : ""}`),
  sessions: () => json<SessionSummary[]>("/api/sessions"),
  session: (id: string) => json<SessionDetail>(`/api/sessions/${id}`),
  createSession: (agent: string, settings: Partial<AgentSettings>) =>
    json<SessionDetail>("/api/sessions", { method: "POST", body: JSON.stringify({ agent, settings }) }),
  updateSettings: (id: string, changes: Partial<AgentSettings>) =>
    json<AgentSettings>(`/api/sessions/${id}/settings`, { method: "PATCH", body: JSON.stringify(changes) }),
  deleteSession: (id: string) => json<{ ok: boolean }>(`/api/sessions/${id}`, { method: "DELETE" }),
  stop: (id: string) => json<{ stopped: boolean }>(`/api/sessions/${id}/stop`, { method: "POST" }),
};

/** Send a message and call onEvent for each Server-Sent Event until the turn ends. */
export async function streamTurn(
  sessionId: string,
  content: string,
  onEvent: (event: AgentEvent) => void,
): Promise<void> {
  const response = await fetch(`/api/sessions/${sessionId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  });
  if (!response.ok || !response.body) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data: "))
        .map((line) => line.slice(6))
        .join("");
      if (data) onEvent(JSON.parse(data) as AgentEvent);
    }
  }
}
