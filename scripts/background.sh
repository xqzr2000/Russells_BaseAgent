#!/usr/bin/env bash
# Run the chat room (scripts/dev.sh) in the background.
#
# The dev container runs `background.sh start` every time the Codespace starts or
# resumes (postStartCommand), so the chat room is up when you come back.
#
#   start    start it unless it is already running   (make start)
#   stop     stop it                                  (make stop)
#   status   say whether it is running
#
# Logs go to runs/dev-server.log (make dev-log). `make dev` stops this copy and runs
# the servers in your terminal instead. Set BASEAGENT_AUTOSTART=0 to skip auto-start.
set -euo pipefail
cd "$(dirname "$0")/.."

PID_FILE=runs/dev-server.pid
LOG_FILE=runs/dev-server.log

running() { [[ -f $PID_FILE ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }

case "${1:-status}" in
  start)
    if [[ "${2:-}" == "--auto" && "${BASEAGENT_AUTOSTART:-1}" == "0" ]]; then
      echo "Auto-start is off (BASEAGENT_AUTOSTART=0). Run: make dev"
      exit 0
    fi
    if running; then
      echo "The chat room is already running in the background (make dev-log, make stop)."
      exit 0
    fi
    mkdir -p runs
    # setsid + nohup: the servers outlive the shell that started them (lifecycle
    # commands and terminals end), and the new process group lets `stop` end them all.
    BASEAGENT_BACKGROUND=1 setsid nohup bash scripts/dev.sh >"$LOG_FILE" 2>&1 </dev/null &
    echo $! >"$PID_FILE"
    echo "Started the chat room in the background. Logs: make dev-log. Stop: make stop."
    ;;
  stop)
    if running; then
      group=$(cat "$PID_FILE")
      kill -TERM -- "-$group" 2>/dev/null || true
      for _ in $(seq 50); do
        kill -0 "$group" 2>/dev/null || break
        sleep 0.2
      done
      kill -KILL -- "-$group" 2>/dev/null || true
      [[ "${2:-}" == "--quiet" ]] || echo "Stopped the background chat room."
    elif [[ "${2:-}" != "--quiet" ]]; then
      echo "The chat room isn't running in the background."
    fi
    rm -f "$PID_FILE"
    ;;
  status)
    if running; then echo "running (pid $(cat "$PID_FILE"))"; else echo "not running"; exit 1; fi
    ;;
  *)
    echo "usage: $0 {start|stop|status}" >&2
    exit 2
    ;;
esac
