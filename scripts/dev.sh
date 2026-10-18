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
echo "Chat room: open port 5173 (port 8000 is the agent API)."
uv run uvicorn baseagent.server.app:app --host 0.0.0.0 --port 8000 \
  --reload --reload-dir src --reload-dir skills &
(cd web && npm run dev) &
wait
