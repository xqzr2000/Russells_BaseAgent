#!/usr/bin/env bash
# One-time setup when the codespace / dev container is created.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# The data scientist's skills (private repo). Optional: without access BaseAgent works without them.
if git submodule update --init knowledge >/dev/null 2>&1; then
  echo "Knowledge skills: $(find knowledge/skills -name SKILL.md 2>/dev/null | wc -l) skills in knowledge/."
else
  echo "Knowledge skills not available (no access to xqzr2000/jack-knows-it-all); continuing without them."
fi

uv sync                          # Python deps (incl. the data science kernel's) into .venv
(cd web && npm ci)               # UI deps from web/package-lock.json
if [ ! -f .env ] && [ -f .env.example ]; then cp .env.example .env; fi

# Colab T4 runs: the Colab CLI here, the Google sign-in persisted in $CLOUDSDK_CONFIG.
make colab-setup || echo "Installing the Colab CLI failed; run 'make colab-setup' later."
if [ -n "${CLOUDSDK_CONFIG:-}" ] && [ ! -d "$CLOUDSDK_CONFIG" ]; then
  sudo mkdir -p "$CLOUDSDK_CONFIG" && sudo chown "$(id -u):$(id -g)" "$CLOUDSDK_CONFIG"
  chmod 700 "$CLOUDSDK_CONFIG"
fi
if [ -f "${CLOUDSDK_CONFIG:-$HOME/.config/gcloud}/application_default_credentials.json" ]; then
  echo "Google credentials for Colab found. Check them with: make colab-check"
else
  echo "For Colab T4 runs, sign in once with: make colab-login"
fi

if [ -n "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY found in the environment."
else
  echo "OPENAI_API_KEY not found. Add it as a Codespaces secret, then rebuild the container."
  echo "Until then the chat room works with the offline 'fake-echo' model."
fi
echo "Setup complete. Run: make dev"
