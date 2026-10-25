FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# The base image ships Yarn's apt repository, whose signing key no longer
# verifies (NO_PUBKEY 62D54FD4003F6525). Any `apt-get update` then fails, which
# breaks the docker-in-docker feature and drops Codespaces into a recovery
# container. Nothing here uses Yarn's apt package, so remove the source.
RUN grep -rl "dl.yarnpkg.com" /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null \
      | xargs -r rm -f \
 && rm -f /usr/share/keyrings/yarn*.gpg /etc/apt/trusted.gpg.d/yarn*.gpg
