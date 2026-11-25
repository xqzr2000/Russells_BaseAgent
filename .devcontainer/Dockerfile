FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# The base image ships Yarn's apt repository, whose signing key no longer
# verifies (NO_PUBKEY 62D54FD4003F6525). Any `apt-get update` then fails, which
# breaks the docker-in-docker feature and drops Codespaces into a recovery
# container. Nothing here uses Yarn's apt package, so remove the source.
RUN grep -rl "dl.yarnpkg.com" /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null \
      | xargs -r rm -f \
 && rm -f /usr/share/keyrings/yarn*.gpg /etc/apt/trusted.gpg.d/yarn*.gpg

RUN apt-get update \
 && apt-get install -y --no-install-recommends stockfish \
 && rm -rf /var/lib/apt/lists/*

# Google Cloud CLI, used once to sign in for Colab T4 runs (make colab-login).
# Installed from Google's apt repository, as in cloud.google.com/sdk/docs/install.
RUN apt-get update \
 && apt-get install -y --no-install-recommends apt-transport-https ca-certificates gnupg curl \
 && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
      > /etc/apt/sources.list.d/google-cloud-sdk.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends google-cloud-cli \
 && rm -rf /var/lib/apt/lists/*
