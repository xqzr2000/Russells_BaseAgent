# The code agent's OpenCode container: OpenCode only, no shell for the model.
#
# BaseAgent starts one per request with a read-only file system, a non-root
# user, RAM folders for /workspace, $HOME and /tmp, and network access only so
# OpenCode can reach its free models. No API key of any kind is passed in.
#
# The model's tools are limited by the config BaseAgent passes in at start
# (OPENCODE_CONFIG_CONTENT, see src/baseagent/codeagent/opencode_config.py).
# Its only way to run code is the run_code tool below, which asks BaseAgent's
# runner bridge to run it in a separate container with no network.
#
# Build: make sandbox-images   (tags baseagent-opencode:1)

FROM debian:bookworm-slim

ARG OPENCODE_VERSION=1.19.2
# SHA-256 of the release archives, so a changed download fails the build.
ARG OPENCODE_SHA256_X64=4eff8d2cf8614ead97eadf1123d46033dc975eadeac3df04644a64bae0f9e819
ARG OPENCODE_SHA256_ARM64=3f73a474065b4b7c4a64cef2395a6123ae1039c2e41f0acbe29ae6521b50df66

# ripgrep: OpenCode's grep/glob use the system rg instead of downloading one.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl ripgrep \
 && arch="$(dpkg --print-architecture)" \
 && case "$arch" in \
      amd64) asset=x64; sum="$OPENCODE_SHA256_X64" ;; \
      arm64) asset=arm64; sum="$OPENCODE_SHA256_ARM64" ;; \
      *) echo "unsupported architecture: $arch" >&2; exit 1 ;; \
    esac \
 && curl -fsSL -o /tmp/opencode.tgz \
      "https://github.com/anomalyco/opencode/releases/download/v${OPENCODE_VERSION}/opencode-linux-${asset}.tar.gz" \
 && echo "$sum  /tmp/opencode.tgz" | sha256sum -c - \
 && tar -xzf /tmp/opencode.tgz -C /usr/local/bin opencode \
 && chmod 755 /usr/local/bin/opencode \
 && rm /tmp/opencode.tgz \
 && apt-get purge -y curl && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/* \
 && /usr/local/bin/opencode --version

# OpenCode's global config folder (XDG_CONFIG_HOME/opencode), owned by root and
# read-only for the opencode user. Because it isn't writable, OpenCode skips
# its startup `npm install` there, and nothing in a run can change the tools.
# The .gitignore is the one OpenCode would otherwise try to write at startup.
COPY config/ /opt/baseagent/config/opencode/
RUN printf 'node_modules\npackage.json\npackage-lock.json\nbun.lock\n.gitignore' \
      > /opt/baseagent/config/opencode/.gitignore \
 && chmod -R a+rX,go-w /opt/baseagent \
 && useradd --uid 1000 --create-home --home-dir /home/opencode --shell /usr/sbin/nologin opencode

ENV HOME=/home/opencode \
    XDG_CONFIG_HOME=/opt/baseagent/config \
    OPENCODE_DISABLE_PROJECT_CONFIG=1 \
    OPENCODE_DISABLE_AUTOUPDATE=1 \
    OPENCODE_DISABLE_LSP_DOWNLOAD=1 \
    OPENCODE_DISABLE_DEFAULT_PLUGINS=1 \
    OPENCODE_DISABLE_CLAUDE_CODE=1 \
    OPENCODE_DISABLE_EXTERNAL_SKILLS=1 \
    OPENCODE_DISABLE_EMBEDDED_WEB_UI=1 \
    OPENCODE_PURE=1

USER opencode
WORKDIR /workspace
