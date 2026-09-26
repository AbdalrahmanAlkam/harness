"""Bounded process execution with whole-process-group timeout cleanup."""
from __future__ import annotations

from contextlib import contextmanager
import os
import signal
import subprocess
import selectors
import threading
import time
from typing import Iterator

# Every live process is registered here so an operator-initiated stop can reach
# children a tool spawned, not just the Python thread driving them. Attribution is
# by *scope* — an explicit owner string set by whoever is running the work — with
# the thread id kept only as a fallback.
#
# Attribution by thread alone is unsound here. Leaders and the Director run on the
# main thread, so cancelling one of them would otherwise signal every process the
# main thread had started, including a verifier's own subprocess. That silently
# turned a passing proof into a non-zero exit. A scope says which agent owns the
# work, which is what "stop that worker" actually means.
_active: dict[int, tuple[str | None, int]] = {}
_active_lock = threading.Lock()
_local = threading.local()


@contextmanager
def process_scope(owner: str | None) -> Iterator[None]:
    """Attribute every process started inside this block to ``owner``."""
    previous = getattr(_local, "owner", None)
    _local.owner = owner
    try:
        yield
    finally:
        _local.owner = previous


def current_scope() -> str | None:
    return getattr(_local, "owner", None)


def _register(pid: int) -> None:
    with _active_lock:
        _active[pid] = (getattr(_local, "owner", None), threading.get_ident())


def _unregister(pid: int) -> None:
    with _active_lock:
        _active.pop(pid, None)


def _owned(owner: str | None = None, thread: int | None = None) -> list[int]:
    """Process ids matching an owner scope, or a thread when no scope is set."""
    with _active_lock:
        if owner is not None:
            return [pid for pid, (scope, _tid) in _active.items() if scope == owner]
        if thread is not None:
            return [pid for pid, (_scope, tid) in _active.items() if tid == thread]
        return []


def live_pids() -> tuple[int, ...]:
    """Every process group id currently running under this process tree."""
    with _active_lock:
        return tuple(_active)


def terminate_owned(owner: str | None = None, *, thread: int | None = None,
                   sig: int = signal.SIGKILL) -> tuple[int, ...]:
    """Kill the process groups belonging to ``owner``, or to ``thread``.

    A worker that is stopped must not leave a Lean build or a Monte-Carlo sweep
    running after its thread returns, so cancellation calls this rather than merely
    abandoning the tool result. The ``owner`` form is preferred: it reaches exactly
    the work that was stopped, even when a worker shares the main thread.
    """
    killed: list[int] = []
    for pid in _owned(owner, thread):
        try:
            os.killpg(pid, sig)
            killed.append(pid)
        except (ProcessLookupError, PermissionError):
            continue
    return tuple(killed)


def run_process(command, *, cwd, timeout, env=None, shell=False):
    timeout = float(timeout)
    if not 0 < timeout <= 300:
        raise ValueError("Timeout must be between 0 and 300 seconds")
    process = subprocess.Popen(command, cwd=cwd, env=env, shell=shell,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    _register(process.pid)
    outputs = [bytearray(), bytearray()]
    deadline = time.monotonic() + timeout
    try:
        with selectors.DefaultSelector() as selector:
            for index, stream in enumerate((process.stdout, process.stderr)):
                selector.register(stream, selectors.EVENT_READ, index)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                for key, _ in selector.select(min(remaining, 0.1)):
                    chunk = os.read(key.fd, 65_536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    else:
                        buffer = outputs[key.data]
                        buffer.extend(chunk[:max(0, 200_001 - len(buffer))])
            process.wait(timeout=max(0.001, deadline - time.monotonic()))
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise
    finally:
        _unregister(process.pid)
        process.stdout.close()
        process.stderr.close()
    def decode(data):
        return data[:200_000].decode("utf-8", errors="replace") + (
            "\n[Output truncated at 200,000 bytes]" if len(data) > 200_000 else "")
    return subprocess.CompletedProcess(command, process.returncode, *(decode(data) for data in outputs))
