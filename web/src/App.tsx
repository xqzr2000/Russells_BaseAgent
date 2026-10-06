import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, streamTurn } from "./api";
import { SettingsDrawer } from "./components/SettingsDrawer";
import { TurnView } from "./components/TurnView";
import { buildTurns } from "./turns";
import type { AgentEvent, AgentInfo, AgentSettings, ServerConfig, SessionDetail, SessionSummary } from "./types";

const DEFAULTS_KEY = "baseagent.defaultSettings";

function loadDefaults(): Partial<AgentSettings> {
  try {
    return JSON.parse(localStorage.getItem(DEFAULTS_KEY) ?? "{}");
  } catch {
    return {};
  }
}

function saveDefaults(settings: AgentSettings) {
  try {
    localStorage.setItem(DEFAULTS_KEY, JSON.stringify(settings));
  } catch {
    /* storage unavailable: defaults just won't persist */
  }
}

function timeAgo(seconds: number): string {
  const diff = Date.now() / 1000 - seconds;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} h ago`;
  return new Date(seconds * 1000).toLocaleDateString();
}

export default function App() {
  const [config, setConfig] = useState<ServerConfig | null>(null);
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [agentName, setAgentName] = useState("general");
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [session, setSession] = useState<SessionDetail | null>(null);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [models, setModels] = useState<string[]>([]);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [pendingSettings, setPendingSettings] = useState<Partial<AgentSettings>>(loadDefaults);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const scroller = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);

  const agent = agents.find((a) => a.name === agentName) ?? agents[0];
  const settings: AgentSettings | null = session?.settings
    ? session.settings
    : agent
      ? { ...agent.default_settings, ...pendingSettings }
      : null;
  const turns = useMemo(() => buildTurns(events), [events]);

  const refreshSessions = useCallback(() => api.sessions().then(setSessions).catch(() => {}), []);
  const loadModels = useCallback((refresh = false) => {
    api
      .models(refresh)
      .then((r) => {
        setModels(r.models);
        setModelsError(r.error);
      })
      .catch((e) => setModelsError(String(e)));
  }, []);

  useEffect(() => {
    Promise.all([api.config(), api.agents()])
      .then(([cfg, list]) => {
        setConfig(cfg);
        setAgents(list);
        if (list.length && !list.some((a) => a.name === "general")) setAgentName(list[0].name);
      })
      .catch(() => setError("Can't reach the agent server. Start it with `make dev`, then reload this page."));
    refreshSessions();
    loadModels();
  }, [refreshSessions, loadModels]);

  useEffect(() => {
    const el = scroller.current;
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [events]);

  const onScroll = () => {
    const el = scroller.current;
    if (el) stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  };

  const newChat = (name = agentName) => {
    setSession(null);
    setEvents([]);
    setAgentName(name);
    setError(null);
    setSidebarOpen(false);
  };

  const openSession = async (id: string) => {
    if (busy) return;
    try {
      const detail = await api.session(id);
      setSession(detail);
      setEvents(detail.events);
      setAgentName(detail.agent);
      setError(null);
      setSidebarOpen(false);
      stickToBottom.current = true;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const deleteSession = async (id: string) => {
    await api.deleteSession(id).catch(() => {});
    if (session?.id === id) newChat();
    refreshSessions();
  };

  const send = async (text: string) => {
    const content = text.trim();
    if (!content || busy || !agent) return;
    setError(null);
    setBusy(true);
    setDraft("");
    stickToBottom.current = true;
    try {
      let current = session;
      if (!current) {
        current = await api.createSession(agent.name, { ...pendingSettings });
        setSession(current);
      }
      const id = current.id;
      await streamTurn(id, content, (event) => setEvents((prev) => [...prev, event]));
      const detail = await api.session(id);
      setSession(detail);
      setEvents(detail.events);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setDraft((d) => d || content);
    } finally {
      setBusy(false);
      refreshSessions();
    }
  };

  const stop = () => {
    if (session) api.stop(session.id).catch(() => {});
  };

  const saveSettings = async (next: AgentSettings) => {
    if (session) {
      const updated = await api.updateSettings(session.id, next);
      setSession({ ...session, settings: updated });
    }
    setPendingSettings(next);
    saveDefaults(next);
  };

  const keyMissing = config && !config.openai_key_configured;

  return (
    <div className={`app ${sidebarOpen ? "sidebar-open" : ""}`}>
      <aside className="sidebar">
        <div className="brand">
          <svg viewBox="0 0 32 32" width="26" height="26" aria-hidden>
            <rect width="32" height="32" rx="7" fill="var(--agent)" />
            <path d="M10 6v20" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" />
            <circle cx="10" cy="11" r="3.2" fill="#fff" />
            <circle cx="10" cy="21" r="3.2" fill="#F2B544" />
            <path d="M15 11h8M15 21h6" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" />
          </svg>
          <div>
            <div className="brand-name">BaseAgent</div>
            <div className="brand-sub">Russell's chat room</div>
          </div>
        </div>
        <button className="btn primary block" onClick={() => newChat()} disabled={busy}>
          New chat
        </button>
        <nav className="session-list" aria-label="Chats">
          {sessions.length === 0 && <p className="muted small">Your chats appear here.</p>}
          {sessions.map((s) => (
            <div key={s.id} className={`session-item ${session?.id === s.id ? "current" : ""}`}>
              <button className="session-open" onClick={() => openSession(s.id)}>
                <span className="session-title">{s.title}</span>
                <span className="session-meta">
                  {agents.find((a) => a.name === s.agent)?.title ?? s.agent}, {timeAgo(s.updated)}
                </span>
              </button>
              <button className="icon-btn subtle" onClick={() => deleteSession(s.id)} aria-label={`Delete chat ${s.title}`}>
                ✕
              </button>
            </div>
          ))}
        </nav>
        <p className="sidebar-foot muted small">Chats live in server memory and are logged to runs/.</p>
      </aside>

      <main className="main">
        <header className="topbar">
          <button className="icon-btn menu" onClick={() => setSidebarOpen((v) => !v)} aria-label="Show chats">
            ☰
          </button>
          <label className="agent-picker">
            <span className="sr-only">Agent</span>
            <select
              value={agent?.name ?? ""}
              onChange={(e) => newChat(e.target.value)}
              disabled={busy}
              aria-label="Agent"
            >
              {agents.map((a) => (
                <option key={a.name} value={a.name}>
                  {a.title}
                </option>
              ))}
            </select>
          </label>
          <button className="model-chip" onClick={() => setSettingsOpen(true)} title="Change model and settings">
            <span className={`dot ${keyMissing && settings?.model !== config?.fake_model ? "warn" : ""}`} aria-hidden />
            {settings?.model ?? "..."}
            {settings?.reasoning_effort && <span className="chip-extra">{settings.reasoning_effort}</span>}
          </button>
          <button className="btn ghost" onClick={() => setSettingsOpen(true)}>
            Settings
          </button>
        </header>

        <div className="conversation" ref={scroller} onScroll={onScroll}>
          <div className="column">
            {turns.length === 0 && agent && (
              <section className="empty">
                <h1>{agent.title}</h1>
                <p>{agent.description}</p>
                {keyMissing && (
                  <p className="note note-warning">
                    No OpenAI key found, so chats use the offline <code>{config?.fake_model}</code> model. Add{" "}
                    <code>OPENAI_API_KEY</code> as a Codespaces secret and rebuild the container to use real models.
                  </p>
                )}
                {agent.examples.length > 0 && (
                  <div className="examples">
                    {agent.examples.map((ex) => (
                      <button key={ex} className="example" onClick={() => send(ex)} disabled={busy}>
                        {ex}
                      </button>
                    ))}
                  </div>
                )}
              </section>
            )}
            {turns.map((turn, i) => (
              <TurnView key={turn.index} turn={turn} live={busy && i === turns.length - 1} />
            ))}
          </div>
        </div>

        <footer className="composer-wrap">
          <div className="column">
            {error && (
              <p className="form-error" role="alert">
                {error}
              </p>
            )}
            <form
              className="composer"
              onSubmit={(e) => {
                e.preventDefault();
                send(draft);
              }}
            >
              <textarea
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                    e.preventDefault();
                    send(draft);
                  }
                }}
                placeholder={`Message ${agent?.title ?? "the agent"}`}
                rows={Math.min(8, Math.max(1, draft.split("\n").length))}
                aria-label="Message"
              />
              {busy ? (
                <button type="button" className="btn stop" onClick={stop}>
                  Stop
                </button>
              ) : (
                <button type="submit" className="btn primary" disabled={!draft.trim()}>
                  Send
                </button>
              )}
            </form>
            <p className="composer-hint muted small">Enter to send, Shift+Enter for a new line.</p>
          </div>
        </footer>
      </main>

      <SettingsDrawer
        open={settingsOpen}
        settings={settings}
        config={config}
        models={models}
        modelsError={modelsError}
        onRefreshModels={() => loadModels(true)}
        onSave={saveSettings}
        onClose={() => setSettingsOpen(false)}
      />
    </div>
  );
}
