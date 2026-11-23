import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { DragEvent } from "react";
import { api, streamTurn } from "./api";
import { fenceIfCode } from "./codeLanguage";
import { ChatSidebar } from "./components/ChatSidebar";
import { ChessPanel } from "./components/ChessPanel";
import { FileIcon, formatSize } from "./components/FileIcon";
import { Icon } from "./components/Icon";
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
const UPLOAD_ACCEPT =
  ".csv,.tsv,.txt,.json,.xlsx,.xls,.parquet," +
  ".py,.js,.mjs,.cjs,.ts,.java,.c,.h,.cpp,.hpp,.cs,.go,.rs,.rb,.php,.sh,.sql,.r,.kt,.swift,.html,.css,.lua,.bas";

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
  const [changingSession, setChangingSession] = useState(false);
  const [attachments, setAttachments] = useState<UploadedFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const dragDepth = useRef(0);
  const fileInput = useRef<HTMLInputElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const sidebarToggle = useRef<HTMLButtonElement>(null);
  const sidebarClose = useRef<HTMLButtonElement>(null);
  const stickToBottom = useRef(true);

  const agent = agents.find((a) => a.name === agentName) ?? agents[0];
  const settings = useMemo(
    () => session?.settings ?? (agent ? { ...agent.default_settings, ...pendingSettings } : null),
    [session?.settings, agent, pendingSettings],
  );
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
    refreshSessions();
    loadModels();
  }, [loadModels, refreshSessions]);

  const closeSidebar = useCallback(() => {
    setSidebarOpen(false);
    requestAnimationFrame(() => sidebarToggle.current?.focus());
  }, []);

  useEffect(() => {
    if (!sidebarOpen) return;
    sidebarClose.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeSidebar();
    };
    const mobile = window.matchMedia("(max-width: 760px)");
    const onResize = () => {
      if (!mobile.matches) setSidebarOpen(false);
    };
    window.addEventListener("keydown", onKey);
    mobile.addEventListener("change", onResize);
    return () => {
      window.removeEventListener("keydown", onKey);
      mobile.removeEventListener("change", onResize);
    };
  }, [sidebarOpen, closeSidebar]);

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
  }, [events, board, busy]);

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

  const sessionLocked = busy || uploading || moving || changingSession;
  const newChat = (name = agentName) => {
    if (sessionLocked) return;
    showSession(null);
    setEvents([]);
    setChessNotice(null);
    setAgentName(name);
    setAttachments([]);
    setError(null);
    if (sidebarOpen) closeSidebar();
  };

  const openSession = async (id: string) => {
    if (sessionLocked) return;
    setChangingSession(true);
    try {
      const detail = await api.session(id);
      showSession(detail);
      setEvents(detail.events);
      setChessNotice(null);
      setAgentName(detail.agent);
      setAttachments([]);
      setError(null);
      if (sidebarOpen) closeSidebar();
      stickToBottom.current = true;
    } catch (e) {
      if (sidebarOpen) closeSidebar();
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setChangingSession(false);
    }
  };

  const deleteSession = async (id: string) => {
    if (sessionLocked) return;
    setChangingSession(true);
    try {
      await api.deleteSession(id);
      if (session?.id === id) newChat();
      await refreshSessions();
    } catch (e) {
      if (sidebarOpen) closeSidebar();
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setChangingSession(false);
    }
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
    if (!files.length || sessionLocked || !agent) return;
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
    if (!content || sessionLocked || !agent) return;
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
    if (sessionLocked) return false;
    setChessNotice(null);
    setMoving(true);
    try {
      const result = await request();
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
  const requestHint = () => {
    if (session) applyChess(session.id, () => api.chessHint(session.id));
  };
  const advanceBoard = () => {
    if (session) applyChess(session.id, () => api.chessAdvance(session.id));
  };
  const claimDraw = () => {
    if (session) applyChess(session.id, () => api.claimDraw(session.id));
  };
  const rewindBoard = (to: "back" | "start") => {
    if (session) applyChess(session.id, () => api.chessRewind(session.id, to));
  };

  const dropHandlers = {
    onDragEnter: (e: DragEvent) => {
      if (!hasFiles(e) || sessionLocked || settingsOpen || !agent) return;
      e.preventDefault();
      dragDepth.current += 1;
      setDragging(true);
    },
    onDragOver: (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = sessionLocked || settingsOpen || !agent ? "none" : "copy";
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
      if (!settingsOpen) uploadFiles(Array.from(e.dataTransfer.files));
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

  const showConversation = turns.length > 0 || board !== null || busy;
  const talkingTo = activeAgent?.title ?? agent?.title;

  return (
    <div className="app" {...dropHandlers}>
      {dragging && (
        <div className="drop-overlay" aria-hidden="true">
          <div className="drop-card">
            <FileIcon />
            Drop files to attach them
          </div>
        </div>
      )}

      {sidebarOpen && <div className="sidebar-backdrop" onClick={closeSidebar} aria-hidden="true" />}
      <aside id="chat-sidebar" className={`sidebar ${sidebarOpen ? "is-open" : ""}`} aria-label="Chat navigation">
        <button ref={sidebarClose} type="button" className="icon-btn sidebar-close" onClick={closeSidebar} aria-label="Close chat navigation">
          <Icon name="close" />
        </button>
        <ChatSidebar
          agents={agents}
          agent={agent}
          sessions={sessions}
          sessionId={session?.id}
          locked={sessionLocked}
          offline={config?.openai_key_configured === false}
          onNew={newChat}
          onOpen={openSession}
          onDelete={deleteSession}
        />
      </aside>

      <main className="workspace" inert={sidebarOpen}>
        <header className="workspace-header">
          <button
            ref={sidebarToggle}
            type="button"
            className="icon-btn sidebar-toggle"
            aria-label="Open chat navigation"
            aria-controls="chat-sidebar"
            aria-expanded={sidebarOpen}
            onClick={() => setSidebarOpen(true)}
          >
            <Icon name="panel" size={20} />
          </button>
          <div className="workspace-label">
            <div className="workspace-title" title={session?.title}>{session?.title || "New conversation"}</div>
            {talkingTo && <div className="workspace-agent">{talkingTo}</div>}
          </div>
          <button
            type="button"
            className="model-button"
            aria-label={settings ? `Settings for ${settings.model}` : "Settings"}
            aria-haspopup="dialog"
            onClick={() => setSettingsOpen(true)}
            disabled={!settings || sessionLocked}
          >
            {settings && <span>{settings.model}</span>}
            <Icon name="sliders" />
          </button>
        </header>

        <div className={`workspace-body ${showConversation ? "with-conversation" : "is-empty"}`}>
          {board && (
            <div className="board-wrap column">
              <ChessPanel
                key={session?.id}
                board={board}
                busy={sessionLocked}
                onMove={chessMove}
                onHint={requestHint}
                onAdvance={advanceBoard}
                onClaimDraw={claimDraw}
                onRewind={rewindBoard}
                onAsk={send}
                notice={chessNotice}
              />
            </div>
          )}
          {showConversation ? (
            <div className="conversation" ref={scroller} onScroll={onScroll} aria-label="Conversation" tabIndex={0}>
              <div className="column">
                {busy && turns.length === 0 && <p className="muted" role="status">Waiting for the agent…</p>}
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
          ) : (
            <div className="welcome column">
              <span className="brand-mark welcome-mark"><Icon name="brand" size={25} /></span>
              <h1>What are we working on?</h1>
            </div>
          )}

          <footer className="composer-wrap column">
            {error && <p className="form-error" role="alert">{error}</p>}
            <form
              className="composer"
              onSubmit={(e) => {
                e.preventDefault();
                send(draft);
              }}
            >
              {(attachments.length > 0 || uploading) && (
                <div className="pending-files">
                  {attachments.map((file) => (
                    <span key={file.path} className="pending-file">
                      <FileIcon />
                      <span className="pending-name" title={file.name}>{file.name}</span>
                      <span className="muted">{formatSize(file.size)}</span>
                      <button
                        type="button"
                        className="icon-btn chip-x"
                        onClick={() => setAttachments((prev) => prev.filter((a) => a.path !== file.path))}
                        aria-label={`Remove ${file.name}`}
                      >
                        <Icon name="close" size={14} />
                      </button>
                    </span>
                  ))}
                  {uploading && <span className="muted" role="status">Uploading…</span>}
                </div>
              )}
              <textarea
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onPaste={(e) => {
                  // Fence pasted code so the code agent sees its original formatting.
                  const fenced = fenceIfCode(e.clipboardData.getData("text/plain"));
                  if (!fenced) return;
                  e.preventDefault();
                  const el = e.currentTarget;
                  const before = draft.slice(0, el.selectionStart);
                  const after = draft.slice(el.selectionEnd);
                  const head = before + (before && !before.endsWith("\n") ? "\n" : "") + fenced + "\n";
                  setDraft(head + after.replace(/^\n/, ""));
                  requestAnimationFrame(() => el.setSelectionRange(head.length, head.length));
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                    e.preventDefault();
                    send(draft);
                  }
                }}
                placeholder={attachments.length ? "Ask about your attachments" : `Message ${talkingTo ?? "the agent"}`}
                rows={Math.min(8, draft.split("\n").length)}
                aria-label="Message"
              />
              <div className="composer-actions">
                <button
                  type="button"
                  className="icon-btn attach"
                  onClick={() => fileInput.current?.click()}
                  disabled={sessionLocked || !agent}
                  aria-label="Attach files"
                  title="Attach files"
                >
                  <Icon name="attach" size={20} />
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
                {busy ? (
                  <button type="button" className="send-button stop-button" onClick={stop} disabled={!session} aria-label="Stop response" title="Stop response">
                    <Icon name="stop" />
                  </button>
                ) : (
                  <button type="submit" className="send-button" disabled={!draft.trim() || sessionLocked || !agent} aria-label="Send message" title="Send message">
                    <Icon name="send" />
                  </button>
                )}
              </div>
            </form>
          </footer>

          {!showConversation && agent && agent.examples.length > 0 && (
            <section className="suggestions column" aria-labelledby="suggestions-heading">
              <h2 id="suggestions-heading">Try a prompt</h2>
              <div className="examples">
                {agent.examples.map((example) => (
                  <button type="button" key={example} className="example" onClick={() => send(example)} disabled={sessionLocked}>
                    <Icon name="chat" size={16} />
                    <span>{example}</span>
                  </button>
                ))}
              </div>
            </section>
          )}
        </div>
      </main>

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
