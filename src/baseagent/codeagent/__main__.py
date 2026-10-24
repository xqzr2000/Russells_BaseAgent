"""Try the code agent from a terminal, without the chat room.

    uv run python -m baseagent.codeagent buggy.py
    uv run python -m baseagent.codeagent buggy.py --mode test --ask "Why doesn't this add up?"
    echo 'print("hi"' | uv run python -m baseagent.codeagent - --language python

Starts the runner bridge on a free port (as the server would), sends the file
through OpenCode in a fresh sandbox with a free model, prints progress to
stderr and the chat answer (Markdown) to stdout. Needs Docker and the images
from `make sandbox-images`.
"""

from __future__ import annotations

import argparse
import asyncio
import socket
import sys
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from baseagent.codeagent.answer import build_answer
from baseagent.codeagent.bridge import RunBridge, make_router
from baseagent.codeagent.opencode_client import CodeAgentError, CodeTask, OpenCodeFixer
from baseagent.codeagent.opencode_config import MODES


def start_bridge(bridge: RunBridge) -> int:
    """Serve the bridge on 0.0.0.0 (so the sandbox can reach it); returns the port."""
    with socket.socket() as s:
        s.bind(("0.0.0.0", 0))
        port = s.getsockname()[1]
    app = FastAPI()
    app.include_router(make_router(bridge))
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    return port


async def run(args: argparse.Namespace) -> int:
    code = sys.stdin.read() if args.file == "-" else Path(args.file).read_text()
    bridge = RunBridge()
    port = start_bridge(bridge)
    fixer = OpenCodeFixer(bridge, bridge_url=f"http://host.docker.internal:{port}/internal/code-runs")
    task = CodeTask(code=code, request=args.ask, language=args.language,
                    filename=None if args.file == "-" else Path(args.file).name, mode=args.mode)
    started = time.monotonic()

    async def progress(message: str) -> None:
        print(f"  [{time.monotonic() - started:5.1f} s] {message}", file=sys.stderr, flush=True)

    try:
        fix = await fixer.fix(task, progress)
    except CodeAgentError as exc:
        print(f"Code agent error: {exc}", file=sys.stderr)
        return 1
    print(f"  [{fix.duration:5.1f} s] done: {len(fix.runs)} run(s), "
          f"{'answered' if fix.answered else 'no structured answer'}"
          f"{', stopped: ' + fix.stopped if fix.stopped else ''}\n", file=sys.stderr)
    print(build_answer(fix))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", help="code file to check, or - for stdin")
    parser.add_argument("--mode", choices=MODES, default="auto")
    parser.add_argument("--ask", default="Check this code and fix what is wrong.",
                        help="what you'd ask in the chat")
    parser.add_argument("--language", help="language when it can't be told from the file name")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
