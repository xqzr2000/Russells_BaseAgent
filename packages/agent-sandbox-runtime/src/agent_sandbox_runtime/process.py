"""Single subprocess boundary for Docker CLI, kept mockable for offline tests."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading


class DockerError(RuntimeError):
    """Docker operation failed with an actionable error."""


def docker(*args: str, timeout: float | None = 120, check: bool = True,
           input: str | None = None, output_limit: int | None = None,
           env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run the Docker CLI.

    ``env`` adds variables to the docker client's own environment, e.g. for
    ``--env NAME`` (value taken from the client, so it never appears in argv).

    With ``output_limit`` (bytes, stdout and stderr together) the output is read
    as it arrives and the client is killed once the limit is passed; the
    result then has ``output_limited = True`` and keeps the first bytes.
    """
    if shutil.which("docker") is None:
        raise DockerError("Docker CLI is not installed. Open the devcontainer or install Docker Desktop.")
    if output_limit is not None:
        return _docker_capped(list(args), timeout, check, input, output_limit)
    try:
        result = subprocess.run(
            ["docker", *args], capture_output=True, text=True, timeout=timeout, input=input,
            errors="replace", env={**os.environ, **env} if env else None,
        )
    except subprocess.TimeoutExpired as exc:
        raise DockerError(f"Docker command timed out after {timeout}s: {args[0] if args else ''}") from exc
    except OSError as exc:
        raise DockerError(f"Could not start Docker: {exc}") from exc
    if check and result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise DockerError(f"docker {args[0] if args else ''} failed ({result.returncode}): {details[-3000:]}")
    return result


def _docker_capped(args: list[str], timeout: float | None, check: bool, input: str | None,
                   limit: int) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.Popen(
            ["docker", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
        )
    except OSError as exc:
        raise DockerError(f"Could not start Docker: {exc}") from exc
    lock = threading.Lock()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    total = [0]
    limited = threading.Event()

    def pump(name: str, stream) -> None:
        while True:
            chunk = stream.read1(65536)
            if not chunk:
                break
            with lock:
                room = limit - total[0]
                if room > 0:
                    buffers[name].extend(chunk[:room])
                total[0] += len(chunk)
                over = total[0] > limit
            if over and not limited.is_set():
                limited.set()
                proc.kill()

    def feed() -> None:
        try:
            proc.stdin.write(input.encode())
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    threads = [threading.Thread(target=pump, args=("stdout", proc.stdout), daemon=True),
               threading.Thread(target=pump, args=("stderr", proc.stderr), daemon=True)]
    if input is not None:
        threads.append(threading.Thread(target=feed, daemon=True))
    for thread in threads:
        thread.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        proc.wait()
        raise DockerError(f"Docker command timed out after {timeout}s: {args[0] if args else ''}") from exc
    finally:
        for thread in threads:
            thread.join(timeout=5)
    result = subprocess.CompletedProcess(
        ["docker", *args], proc.returncode,
        buffers["stdout"].decode(errors="replace"), buffers["stderr"].decode(errors="replace"),
    )
    result.output_limited = limited.is_set()  # type: ignore[attr-defined]
    if check and result.returncode != 0 and not limited.is_set():
        details = (result.stderr or result.stdout).strip()
        raise DockerError(f"docker {args[0] if args else ''} failed ({result.returncode}): {details[-3000:]}")
    return result


def docker_ready() -> tuple[bool, str]:
    try:
        result = docker("info", "--format", "{{.ServerVersion}}", timeout=15)
        return True, result.stdout.strip()
    except DockerError as exc:
        return False, str(exc)
