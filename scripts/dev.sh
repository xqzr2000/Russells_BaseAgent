#!/usr/bin/env bash
# Run the Python agent server (port 8000, auto-reload) and the Vite UI
# (port 5173, proxies /api to 8000) together. Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")/.."
trap 'kill 0' EXIT INT TERM
# Install UI packages when package-lock.json has changed (e.g. after pulling a branch).
if [[ ! -f web/node_modules/.package-lock.json || web/package-lock.json -nt web/node_modules/.package-lock.json ]]; then
  echo "Installing chat room packages (npm ci)..."
  (cd web && npm ci --no-audit --no-fund)
fi
# `make dev` in a terminal takes over from the auto-started background copy.
if [[ -z "${BASEAGENT_BACKGROUND:-}" ]] && bash scripts/background.sh status >/dev/null 2>&1; then
  bash scripts/background.sh stop --quiet
  echo "Stopped the background chat room; running it here instead."
fi
if [[ -n "${CODESPACE_NAME:-}" && -n "${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-}" ]]; then
  echo "Chat room: https://${CODESPACE_NAME}-5173.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}"
else
  echo "Chat room: open port 5173 (port 8000 is the agent API)."
fi
uv run uvicorn baseagent.server.app:app --host 0.0.0.0 --port 8000 \
  --reload --reload-dir src --reload-dir skills &
(cd web && npm run dev) &
# If either server stops, stop the other too (the EXIT trap), so the chat room is
# never half up: a page that loads but can't chat.
wait -n
