#!/usr/bin/env bash
# One-time setup when the codespace / dev container is created.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync                          # Python deps (incl. the data science kernel's) into .venv
(cd web && npm ci)               # UI deps from web/package-lock.json
if [ ! -f .env ] && [ -f .env.example ]; then cp .env.example .env; fi

# The code agent's sandbox images (needs Docker; safe to re-run later).
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  make sandbox-images || echo "Building the sandbox images failed; run 'make sandbox-images' later."
else
  echo "Docker isn't ready yet; run 'make sandbox-images' before using the code agent."
fi

if [ -n "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY found in the environment."
else
  echo "OPENAI_API_KEY not found. Add it as a Codespaces secret, then rebuild the container."
  echo "Until then the chat room works with the offline 'fake-echo' model."
fi
echo "Setup complete. Run: make dev"
