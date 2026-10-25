# The code agent's runner: runs one snippet of the user's code, then is deleted.
#
# BaseAgent starts it with no network, a read-only file system, 512 MB of memory
# (no swap), 1 CPU, a small process limit and a non-root user; the code goes in
# /work (a small RAM folder). Nothing here can be installed at run time, so the
# packages a snippet may import are the ones below.
#
# Build: make sandbox-images   (tags baseagent-runner:1)

FROM node:24-bookworm-slim AS node

FROM python:3.12-slim-bookworm

# Node (for JavaScript and TypeScript) from the official image.
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
 && npm install --global --no-fund --no-audit typescript@5 tsx@4 @types/node@24 \
 && npm cache clean --force

# Python packages a learning snippet is likely to import, plus ruff for checks.
RUN pip install --no-cache-dir \
      numpy pandas scipy matplotlib requests ruff

RUN useradd --uid 1000 --create-home --shell /bin/bash runner \
 && mkdir -p /work && chown runner:runner /work

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp \
    NODE_OPTIONS=--max-old-space-size=384

USER runner
WORKDIR /work
