"""Docker sandbox management and smoke tests."""

from __future__ import annotations

import argparse
import sys

from .process import docker, docker_ready
from .sandbox import Sandbox


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and test the local Docker sandbox backend")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="verify Docker daemon access")
    sub.add_parser("smoke", help="launch a short-lived sandbox (pulls image if absent)")
    sub.add_parser("list", help="show running managed containers")
    sub.add_parser("cleanup", help="remove all managed containers (use with care)")
    args = parser.parse_args()
    if args.command == "doctor":
        ready, detail = docker_ready()
        print(("[ok] Docker daemon " if ready else "[error] ") + detail)
        return 0 if ready else 1
    if args.command == "smoke":
        with Sandbox("python:3.12-slim", deployment_timeout=60) as sandbox:
            result = sandbox.execute(["python", "-c", "print('sandbox ready')"], shell=False)
            print(result["output"].strip())
            return result["returncode"]
    if args.command == "list":
        print(docker("ps", "-a", "--filter", "label=org.agent-sandbox-runtime.managed=true", "--format", "{{.ID}} {{.Status}} {{.Names}}").stdout, end="")
        return 0
    if args.command == "cleanup":
        ids = docker("ps", "-aq", "--filter", "label=org.agent-sandbox-runtime.managed=true").stdout.split()
        if ids:
            docker("rm", "-f", *ids)
        print(f"Removed {len(ids)} managed containers")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
