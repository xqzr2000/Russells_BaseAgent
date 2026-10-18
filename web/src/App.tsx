import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { DragEvent } from "react";
import { api, streamTurn } from "./api";
import { ChessPanel } from "./components/ChessPanel";
import { FileIcon, formatSize } from "./components/FileIcon";
import { Meadow, firstImpression, jackForced } from "./components/Meadow";
import { SettingsPanel } from "./components/SettingsPanel";
import { TurnView } from "./components/TurnView";
import { buildTurns } from "./turns";
import type {
  AgentEvent,
  AgentInfo,
  AgentSettings,
  BoardState,
  ChessMoveResult,
  ServerConfig,
  SessionDetail,
  SessionSummary,
  UploadedFile,
} from "./types";

const DEFAULTS_KEY = "baseagent.defaultSettings";
const UPLOAD_ACCEPT = ".csv,.tsv,.txt,.json,.xlsx,.xls,.parquet";

const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer.types).includes("Files");

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
  const [agentName, setAgentName] = useState("");
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [session, setSession] = useState<SessionDetail | null>(null);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [board, setBoard] = useState<BoardState | null>(null); // the chess coach's board, if any
  const [chessNotice, setChessNotice] = useState<string | null>(null);
  const [moving, setMoving] = useState(false); // a board move is on its way to the server
  const [activeAgent, setActiveAgent] = useState<{ name: string; title: string } | null>(null); // after a handoff
  // The chat on screen. Late responses for another chat (after switching) are dropped.
  const shownSession = useRef<string | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [pendingSettings, setPendingSettings] = useState<Partial<AgentSettings>>(loadDefaults);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [attachments, setAttachments] = useState<UploadedFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const dragDepth = useRef(0);
  const fileInput = useRef<HTMLInputElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const composerRef = useRef<HTMLFormElement>(null);
  const stickToBottom = useRef(true);
  /*
   * Jack's first-impression video: "pending" until we know whether the server
   * already has chats, "armed" while waiting for the first keystroke,
   * "playing" until it fades back to the still, then "off" for good.
   */
  const [jack, setJack] = useState<"pending" | "armed" | "playing" | "off">(firstImpression ? "pending" : "off");
  const jackDone = useCallback(() => setJack("off"), []);

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
        const preferred = list.find((a) => a.name === cfg.default_agent) ?? list[0];
        if (preferred) setAgentName(preferred.name);
      })
      .catch(() => setError("Can't reach the agent server. Start it with `make dev`, then reload this page."));
    api
      .sessions()
      .then((list) => {
        setSessions(list);
        // A first impression only: if there are chats already, stay on the still.
        setJack((j) => (j === "pending" ? (list.length && !jackForced ? "off" : "armed") : j));
      })
      .catch(() => setJack((j) => (j === "pending" ? "off" : j)));
    loadModels();
  }, [loadModels]);

  // Opening a chat or sending without typing (an example prompt) spends the first impression.
  useEffect(() => {
    if (jack === "armed" && turns.length > 0) setJack("off");
  }, [jack, turns.length]);

  const closePanels = useCallback(() => {
    setSidebarOpen(false);
    setSettingsOpen(false);
  }, []);

  useEffect(() => {
    if (!sidebarOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setSidebarOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [sidebarOpen]);

  // A file dropped outside the drop zone must not make the browser navigate away.
  useEffect(() => {
    const block = (e: globalThis.DragEvent) => e.preventDefault();
    window.addEventListener("dragover", block);
    window.addEventListener("drop", block);
    return () => {
      window.removeEventListener("dragover", block);
      window.removeEventListener("drop", block);
    };
  }, []);

  useEffect(() => {
    const el = scroller.current;
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [events]);

  useEffect(() => {
    if (!chessNotice) return;
    const timer = window.setTimeout(() => setChessNotice(null), 4000);
    return () => window.clearTimeout(timer);
  }, [chessNotice]);

  const onScroll = () => {
    const el = scroller.current;
    if (el) stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  };

  const showSession = (detail: SessionDetail | null) => {
    shownSession.current = detail?.id ?? null;
    setSession(detail);
    setBoard(detail?.board ?? null);
    setActiveAgent(detail?.active_agent ?? null);
  };

  const newChat = (name = agentName) => {
    if (moving) return;
    showSession(null);
    setEvents([]);
    setChessNotice(null);
    setAgentName(name);
    setAttachments([]);
    setError(null);
    setSidebarOpen(false);
  };

  const openSession = async (id: string) => {
    if (busy || uploading || moving) return;
    try {
      const detail = await api.session(id);
      showSession(detail);
      setEvents(detail.events);
      setChessNotice(null);
      setAgentName(detail.agent);
      setAttachments([]);
      setError(null);
      setSidebarOpen(false);
      setJack((j) => (j === "armed" || j === "pending" ? "off" : j));
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

  /** The open chat, created on first use (sending a message or attaching a file). */
  const ensureSession = async (): Promise<SessionDetail> => {
    if (session) return session;
    if (!agent) throw new Error("No agent is available.");
    const created = await api.createSession(agent.name, { ...pendingSettings });
    showSession(created);
    return created;
  };

  const uploadFiles = async (files: File[]) => {
    if (!files.length || uploading || !agent) return;
    setError(null);
    setUploading(true);
    try {
      const current = await ensureSession();
      for (const file of files) {
        const uploaded = await api.uploadFile(current.id, file);
        setAttachments((prev) => [...prev.filter((a) => a.path !== uploaded.path), uploaded]);
      }
    } catch (e) {
      setError(`Upload failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setUploading(false);
      refreshSessions();
    }
  };

  const send = async (text: string) => {
    const content = text.trim();
    if (!content || busy || uploading || !agent) return;
    const pending = attachments;
    setError(null);
    setBusy(true);
    setDraft("");
    setAttachments([]);
    stickToBottom.current = true;
    try {
      const current = await ensureSession();
      const id = current.id;
      await streamTurn(id, content, pending.map((a) => a.path), (event) => {
        setEvents((prev) => [...prev, event]);
        if (event.type === "board") setBoard(event.board);
        if (event.type === "handoff") {
          // The new agent's panels (e.g. its board) appear or disappear at once.
          setBoard(event.ui_state.board ?? null);
          setActiveAgent({ name: event.to_agent, title: event.title });
        }
      });
      const detail = await api.session(id);
      if (shownSession.current === id) {
        showSession(detail);
        setEvents(detail.events);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setDraft((d) => d || content);
      setAttachments((a) => (a.length ? a : pending));
    } finally {
      setBusy(false);
      refreshSessions();
    }
  };

  /** Apply a board change from the server, if the user is still looking at that chat. */
  const applyChess = async (sessionId: string, request: () => Promise<ChessMoveResult>): Promise<boolean> => {
    if (busy || moving) return false;
    setChessNotice(null);
    setMoving(true);
    try {
      const before = board?.history.length ?? 0;
      const result = await request();
      // Let the user's own move settle before the bot's reply animates in.
      if (result.board.history.length > before + 1) await new Promise((r) => setTimeout(r, 380));
      if (shownSession.current !== sessionId) return true; // the user switched chats meanwhile
      setBoard(result.board);
      refreshSessions();
      // Lesson moves and the end of a game are reported to the coach as a chat message.
      if (result.needs_coach) setTimeout(() => send(result.message), 0);
      return true;
    } catch (e) {
      if (shownSession.current === sessionId) setChessNotice(e instanceof Error ? e.message : String(e));
      return false;
    } finally {
      setMoving(false);
    }
  };

  /** The user dragged a piece (UCI, e.g. "e2e4", or "e7e8n" for a promotion). */
  const chessMove = (uci: string) => (session ? applyChess(session.id, () => api.chessMove(session.id, uci)) : Promise.resolve(false));
  const claimDraw = () => {
    if (session) applyChess(session.id, () => api.claimDraw(session.id));
  };
  const rewindBoard = (to: "back" | "start") => {
    if (session) applyChess(session.id, () => api.chessRewind(session.id, to));
  };

  const dropHandlers = {
    onDragEnter: (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      dragDepth.current += 1;
      setDragging(true);
    },
    onDragOver: (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
    },
    onDragLeave: (e: DragEvent) => {
      if (!hasFiles(e)) return;
      dragDepth.current = Math.max(0, dragDepth.current - 1);
      if (dragDepth.current === 0) setDragging(false);
    },
    onDrop: (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      dragDepth.current = 0;
      setDragging(false);
      uploadFiles(Array.from(e.dataTransfer.files));
    },
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
  const hasChat = turns.length > 0;
  const showDialog = hasChat || board !== null;
  // After a handoff (e.g. to the chess coach), the pill and placeholder name the agent now talking.
  const talkingTo = activeAgent?.title ?? agent?.title;

  return (
    <div className="app" {...dropHandlers}>
      <Meadow armed={jack === "armed" || jack === "playing"} play={jack === "playing"} onDone={jackDone} />
      {dragging && (
        <div className="drop-overlay" aria-hidden>
          <div className="drop-card">
            <FileIcon />
            Drop a data file to attach it
            <span className="muted small">CSV, TSV, Excel, JSON or Parquet</span>
          </div>
        </div>
      )}

      {/* Two frosted pills float over the meadow: chats (top left) and model + settings (top right). */}
      <header className="hud">
        <button
          className={`pill chats-pill ${sidebarOpen ? "is-open" : ""}`}
          onClick={() => {
            setSettingsOpen(false);
            setSidebarOpen((v) => !v);
          }}
          aria-expanded={sidebarOpen}
          aria-label={`Chats and agents (talking to: ${talkingTo ?? "none"})`}
        >
          <svg className="chevron" viewBox="0 0 16 16" width="16" height="16" aria-hidden>
            <path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span className="pill-text">{talkingTo ?? "BaseAgent"}</span>
        </button>
        <button
          className={`pill model-pill ${settingsOpen ? "is-open" : ""}`}
          onClick={() => {
            setSidebarOpen(false);
            setSettingsOpen((v) => !v);
          }}
          aria-expanded={settingsOpen}
          title="Change model and settings"
        >
          <span className={`dot ${keyMissing && settings?.model !== config?.fake_model ? "warn" : ""}`} aria-hidden />
          <span className="pill-text model-name">{settings?.model ?? "..."}</span>
          {settings?.reasoning_effort && <span className="chip-extra">{settings.reasoning_effort}</span>}
          <svg className="gear" viewBox="0 0 20 20" width="17" height="17" aria-hidden>
            <path
              d="M10 12.6a2.6 2.6 0 1 0 0-5.2 2.6 2.6 0 0 0 0 5.2Zm6.3-1.4.1-1.2-.1-1.2 1.6-1.3-1.6-2.8-2 .7a6.3 6.3 0 0 0-2-1.2L12 2.1H8.8l-.4 2.1a6.3 6.3 0 0 0-2 1.2l-2-.7-1.6 2.8 1.6 1.3-.1 1.2.1 1.2-1.6 1.3 1.6 2.8 2-.7c.6.5 1.3.9 2 1.2l.4 2.1H12l.4-2.1c.7-.3 1.4-.7 2-1.2l2 .7 1.6-2.8-1.7-1.3Z"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.4"
              strokeLinejoin="round"
            />
          </svg>
          <span className="sr-only">Settings</span>
        </button>
      </header>

      {/* The frosted dialog panel: only once there's a conversation (or a chess board). */}
      <div className={`dialog ${board ? "has-board" : ""} ${showDialog ? "is-shown" : ""}`} ref={dialogRef} aria-hidden={!showDialog}>
        {board && (
          <ChessPanel
            board={board}
            busy={busy || moving}
            onMove={chessMove}
            onClaimDraw={claimDraw}
            onRewind={rewindBoard}
            onAsk={(message) => send(message)}
            notice={chessNotice}
          />
        )}
        <div className="conversation" ref={scroller} onScroll={onScroll}>
          <div className="column">
            {turns.map((turn, i) => (
              <TurnView
                key={turn.index}
                turn={turn}
                live={busy && i === turns.length - 1}
                agentTitle={agents.find((a) => a.name === turn.agent)?.title}
              />
            ))}
          </div>
        </div>
      </div>

      <footer className="composer-wrap">
        <div className="column">
          {!hasChat && agent && (
            <div className={`starter ${draft || busy ? "is-quiet" : ""}`}>
              {keyMissing && (
                <p className="note note-warning starter-note">
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
            </div>
          )}
          {error && (
            <p className="form-error" role="alert">
              {error}
            </p>
          )}
          {(attachments.length > 0 || uploading) && (
            <div className="pending-files">
              {attachments.map((file) => (
                <span key={file.path} className="pending-file">
                  <FileIcon />
                  <span className="pending-name">{file.name}</span>
                  <span className="muted">{formatSize(file.size)}</span>
                  <button
                    type="button"
                    className="chip-x"
                    onClick={() => setAttachments((prev) => prev.filter((a) => a.path !== file.path))}
                    aria-label={`Remove ${file.name}`}
                  >
                    ✕
                  </button>
                </span>
              ))}
              {uploading && <span className="pending-file is-uploading">Uploading...</span>}
            </div>
          )}
          <form
            ref={composerRef}
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              send(draft);
            }}
          >
            <button
              type="button"
              className="icon-btn attach"
              onClick={() => fileInput.current?.click()}
              disabled={uploading || !agent}
              aria-label="Attach a data file"
              title="Attach a data file (or drag it into the chat)"
            >
              <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden>
                <path
                  d="M13.5 6.5 8 12a1.6 1.6 0 0 0 2.3 2.3l5.6-5.6a3.2 3.2 0 0 0-4.5-4.5L5.6 10a4.8 4.8 0 0 0 6.8 6.8l4.1-4.1"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                />
              </svg>
            </button>
            <input
              ref={fileInput}
              type="file"
              multiple
              accept={UPLOAD_ACCEPT}
              hidden
              onChange={(e) => {
                uploadFiles(Array.from(e.target.files ?? []));
                e.target.value = "";
              }}
            />
            <textarea
              value={draft}
              onChange={(e) => {
                setDraft(e.target.value);
                // Jack's first impression starts on the first keystroke of a fresh chat.
                if (jack === "armed" && e.target.value && !hasChat) setJack("playing");
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault();
                  send(draft);
                }
              }}
              placeholder={
                attachments.length ? "Ask about the attached file, e.g. Do EDA on it" : `Message ${talkingTo ?? "the agent"}`
              }
              rows={Math.min(8, Math.max(1, draft.split("\n").length))}
              aria-label="Message"
            />
            {busy ? (
              <button type="button" className="btn stop" onClick={stop}>
                Stop
              </button>
            ) : (
              <button type="submit" className="btn primary" disabled={!draft.trim() || uploading}>
                Send
              </button>
            )}
          </form>
          <p className="composer-hint muted small">
            Enter to send, Shift+Enter for a new line. Drag a CSV into the chat to attach it.
          </p>
        </div>
      </footer>

      {sidebarOpen && (
        <>
          <div className="float-backdrop" onClick={closePanels} aria-hidden />
          <aside className="float-panel chats-panel" aria-label="Chats">
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
            <label className="field agent-field">
              <span className="label">Agent</span>
              <select value={agent?.name ?? ""} onChange={(e) => newChat(e.target.value)} disabled={busy}>
                {agents.map((a) => (
                  <option key={a.name} value={a.name}>
                    {a.title}
                  </option>
                ))}
              </select>
              {agent?.description && <span className="hint">{agent.description}</span>}
            </label>
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
            <p className="panel-note muted small">Chats live in server memory and are logged to runs/.</p>
          </aside>
        </>
      )}

      <SettingsPanel
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
