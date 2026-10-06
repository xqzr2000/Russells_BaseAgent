import { useEffect, useState } from "react";
import type { AgentSettings, ServerConfig } from "../types";

const EFFORTS: { value: string | null; label: string }[] = [
  { value: null, label: "Default" },
  { value: "minimal", label: "Minimal" },
  { value: "low", label: "Low" },
  { value: "medium", label: "Medium" },
  { value: "high", label: "High" },
];

interface Props {
  open: boolean;
  settings: AgentSettings | null;
  config: ServerConfig | null;
  models: string[];
  modelsError: string | null;
  onRefreshModels: () => void;
  onSave: (settings: AgentSettings) => Promise<void>;
  onClose: () => void;
}

export function SettingsDrawer({ open, settings, config, models, modelsError, onRefreshModels, onSave, onClose }: Props) {
  const [draft, setDraft] = useState<AgentSettings | null>(settings);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setDraft(settings);
      setError(null);
    }
  }, [open, settings]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    if (open) window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open || !draft) return null;
  const set = <K extends keyof AgentSettings>(key: K, value: AgentSettings[K]) =>
    setDraft({ ...draft, [key]: value });
  const options = Array.from(new Set([...(models ?? []), ...(config?.suggested_models ?? [])]));

  const save = async () => {
    if (!draft.model.trim()) {
      setError("Enter a model name.");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await onSave({ ...draft, model: draft.model.trim() });
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer" role="dialog" aria-label="Settings" onClick={(e) => e.stopPropagation()}>
        <header className="drawer-head">
          <h2>Settings</h2>
          <button className="icon-btn" onClick={onClose} aria-label="Close settings">
            ✕
          </button>
        </header>

        <p className={`key-status ${config?.openai_key_configured ? "ok" : "missing"}`}>
          {config?.openai_key_configured
            ? `OpenAI key found (${config.base_url_host}).`
            : "No OPENAI_API_KEY found. Add it as a Codespaces secret and rebuild, or use the fake-echo model to try things offline."}
        </p>

        <section className="field">
          <label htmlFor="model">Model</label>
          <div className="row">
            <input
              id="model"
              list="model-options"
              value={draft.model}
              onChange={(e) => set("model", e.target.value)}
              spellCheck={false}
              autoComplete="off"
            />
            <button className="btn ghost" onClick={onRefreshModels} type="button">
              Refresh list
            </button>
          </div>
          <datalist id="model-options">
            {options.map((m) => (
              <option key={m} value={m} />
            ))}
          </datalist>
          <p className="hint">
            {modelsError ?? `${models.length} models available from your account. Type any model id to use one not listed.`}
          </p>
        </section>

        <section className="field">
          <span className="label">Reasoning effort</span>
          <div className="segmented" role="radiogroup" aria-label="Reasoning effort">
            {EFFORTS.map((e) => (
              <button
                key={e.label}
                role="radio"
                aria-checked={draft.reasoning_effort === e.value}
                className={draft.reasoning_effort === e.value ? "on" : ""}
                onClick={() => set("reasoning_effort", e.value)}
                type="button"
              >
                {e.label}
              </button>
            ))}
          </div>
          <p className="hint">For reasoning models such as gpt-5 and o4-mini. Ignored automatically by models that don't support it.</p>
        </section>

        <section className="field">
          <span className="label">Temperature</span>
          <label className="check">
            <input
              type="checkbox"
              checked={draft.temperature === null}
              onChange={(e) => set("temperature", e.target.checked ? null : 0.7)}
            />
            Use the model's default
          </label>
          {draft.temperature !== null && (
            <div className="row">
              <input
                type="range"
                min={0}
                max={2}
                step={0.1}
                value={draft.temperature}
                onChange={(e) => set("temperature", Number(e.target.value))}
                aria-label="Temperature"
              />
              <span className="value">{draft.temperature.toFixed(1)}</span>
            </div>
          )}
        </section>

        <div className="field-grid">
          <section className="field">
            <label htmlFor="max-steps">Max steps per message</label>
            <input
              id="max-steps"
              type="number"
              min={1}
              max={200}
              value={draft.max_steps}
              onChange={(e) => set("max_steps", Math.max(1, Number(e.target.value)))}
            />
          </section>
          <section className="field">
            <label htmlFor="max-out">Max output tokens</label>
            <input
              id="max-out"
              type="number"
              min={256}
              step={256}
              value={draft.max_output_tokens}
              onChange={(e) => set("max_output_tokens", Math.max(256, Number(e.target.value)))}
            />
          </section>
        </div>

        <section className="field">
          <span className="label">Context compaction</span>
          <label className="check">
            <input
              type="checkbox"
              checked={draft.compact_threshold_tokens !== null}
              onChange={(e) => set("compact_threshold_tokens", e.target.checked ? 8000 : null)}
            />
            Summarize old steps when the prompt grows past a limit
          </label>
          {draft.compact_threshold_tokens !== null && (
            <div className="field-grid">
              <div>
                <label htmlFor="threshold">Token limit</label>
                <input
                  id="threshold"
                  type="number"
                  min={500}
                  step={500}
                  value={draft.compact_threshold_tokens}
                  onChange={(e) => set("compact_threshold_tokens", Math.max(500, Number(e.target.value)))}
                />
              </div>
              <div>
                <label htmlFor="keep">Recent steps kept</label>
                <input
                  id="keep"
                  type="number"
                  min={1}
                  max={10}
                  value={draft.compaction_keep_recent_steps}
                  onChange={(e) => set("compaction_keep_recent_steps", Math.max(1, Number(e.target.value)))}
                />
              </div>
            </div>
          )}
        </section>

        {error && <p className="form-error">{error}</p>}
        <footer className="drawer-foot">
          <p className="hint">Applies to this chat and becomes the default for new chats.</p>
          <button className="btn primary" onClick={save} disabled={saving}>
            {saving ? "Saving..." : "Save settings"}
          </button>
        </footer>
      </aside>
    </div>
  );
}
