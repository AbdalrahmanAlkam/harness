"""Bounded process execution with whole-process-group timeout cleanup."""
from __future__ import annotations

import os
import signal
import subprocess
import selectors
import time


def run_process(command, *, cwd, timeout, env=None, shell=False):
    timeout = float(timeout)
    if not 0 < timeout <= 300:
        raise ValueError("Timeout must be between 0 and 300 seconds")
    process = subprocess.Popen(command, cwd=cwd, env=env, shell=shell,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
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
        process.stdout.close()
        process.stderr.close()
    def decode(data):
        return data[:200_000].decode("utf-8", errors="replace") + (
            "\n[Output truncated at 200,000 bytes]" if len(data) > 200_000 else "")
    return subprocess.CompletedProcess(command, process.returncode, *(decode(data) for data in outputs))
