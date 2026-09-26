"""Bounded process execution with whole-process-group timeout cleanup."""
from __future__ import annotations

import os
import signal
import subprocess
import selectors
import threading
import time

# Every live process is registered here so an operator-initiated stop can reach
# children a tool spawned, not just the Python thread driving them. Registration
# is keyed by the *owning thread* because a worker runs its tools on its own
# thread; that is what lets a leader kill exactly the processes belonging to the
# worker it stopped instead of collateral-damaging a concurrent sibling.
_active: dict[int, int] = {}
_active_lock = threading.Lock()
_local = threading.local()


def _register(pid: int) -> None:
    with _active_lock:
        _active[pid] = threading.get_ident()


def _unregister(pid: int) -> None:
    with _active_lock:
        _active.pop(pid, None)


def _owner_ids() -> set[int]:
    with _active_lock:
        return {owner for pid, owner in _active.items() if owner == threading.get_ident()}


def live_pids() -> tuple[int, ...]:
    """Every process group id currently running under this process tree."""
    with _active_lock:
        return tuple(_active)


def terminate_owned(owner: int | None = None, *, sig: int = signal.SIGKILL) -> tuple[int, ...]:
    """Kill the process groups owned by ``owner`` (default: the current thread).

    Returns the pids that were signalled. A worker that is stopped must not
    leave a Lean build or a Monte-Carlo sweep running after its thread returns,
    so cancellation calls this rather than merely abandoning the tool result.
    """
    if owner is None:
        targets = _owner_ids()
    else:
        with _active_lock:
            targets = {candidate for pid, candidate in _active.items() if candidate == owner}
    killed: list[int] = []
    with _active_lock:
        pids = [pid for pid, candidate in _active.items() if candidate in targets]
    for pid in pids:
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
