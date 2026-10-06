#!/usr/bin/env bash
# Run the Python agent server (port 8000, auto-reload) and the Vite UI
# (port 5173, proxies /api to 8000) together. Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")/.."
trap 'kill 0' EXIT INT TERM
uv run uvicorn baseagent.server.app:app --host 0.0.0.0 --port 8000 \
  --reload --reload-dir src --reload-dir skills &
(cd web && npm run dev) &
wait
