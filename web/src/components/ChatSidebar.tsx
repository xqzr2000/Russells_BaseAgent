import { Icon, type IconName } from "./Icon";
import { JACK_AGENT, JackAvatar } from "./JackAvatar";
import type { AgentInfo, SessionSummary } from "../types";

interface Props {
  agents: AgentInfo[];
  agent?: AgentInfo;
  sessions: SessionSummary[];
  sessionId?: string;
  locked: boolean;
  offline: boolean;
  onNew: (agentName?: string) => void;
  onOpen: (id: string) => void;
  onDelete: (id: string) => void;
}

const agentIcons: Partial<Record<string, IconName>> = {
  general: "chat",
  data_science: "data",
  chess_coach: "chess",
};

export function ChatSidebar({ agents, agent, sessions, sessionId, locked, offline, onNew, onOpen, onDelete }: Props) {
  return (
    <>
      <div className="sidebar-header">
        <span className="brand-mark"><Icon name="brand" /></span>
        <span className="brand-name">BaseAgent</span>
      </div>
      <button type="button" className="new-chat" onClick={() => onNew()} disabled={locked || !agent}>
        <Icon name="plus" />
        New chat
      </button>
      <div className="sidebar-content">
        {agents.length > 0 && (
          <nav className="sidebar-group" aria-labelledby="agents-heading">
            <h2 id="agents-heading">Agents</h2>
            {agents.map((item) => (
              <button
                key={item.name}
                type="button"
                className="sidebar-link agent-link"
                aria-pressed={item.name === agent?.name}
                title={item.description}
                onClick={() => onNew(item.name)}
                disabled={locked}
              >
                <span className="agent-icon">
                  {item.name === JACK_AGENT ? <JackAvatar /> : <Icon name={agentIcons[item.name] ?? "agents"} />}
                </span>
                {item.name === JACK_AGENT ? (
                  <span className="jack-sidebar-label">
                    <span>Jack</span>
                    <span className="jack-role">Coordinator</span>
                  </span>
                ) : (
                  <span>{item.title}</span>
                )}
              </button>
            ))}
          </nav>
        )}
        <nav className="sidebar-group history" aria-labelledby="history-heading">
          <h2 id="history-heading">History</h2>
          {sessions.length === 0 && <p className="muted">No conversations yet.</p>}
          {sessions.map((item) => (
            <div key={item.id} className="history-item" data-current={sessionId === item.id}>
              <button
                type="button"
                className="sidebar-link history-link"
                aria-current={sessionId === item.id ? "true" : undefined}
                title={item.title}
                onClick={() => onOpen(item.id)}
                disabled={locked || sessionId === item.id}
              >
                <span>{item.title}</span>
              </button>
              <button
                type="button"
                className="icon-btn delete-chat"
                aria-label={`Delete chat ${item.title}`}
                onClick={() => onDelete(item.id)}
                disabled={locked}
              >
                <Icon name="trash" size={16} />
              </button>
            </div>
          ))}
        </nav>
      </div>
      {offline && <div className="sidebar-status"><span aria-hidden="true" />Offline model available</div>}
    </>
  );
}
