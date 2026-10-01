"""OS-level confinement backends for running a command.

The harness core deliberately ships no sandbox. It ships the *idea* of one --
``docs/plugins.md`` puts anything not everyone needs in a plugin -- and this is
that plugin. Five backends live here, each of which is available on some
machines and not others:

- **landlock**  (Linux, in-kernel, via the ``landlock_restrict_self`` syscall)
- **bubblewrap** (Linux, unprivileged user namespaces)
- **seatbelt**  (macOS, ``sandbox-exec`` with a generated SBPL profile)
- **container** (Linux, docker or podman -- opt-in, never chosen automatically)
- **none**      (always available, and confines nothing)

The one rule everything here is arranged around: **never claim confinement that
was not obtained.** A backend that cannot be proven usable reports why and
reports itself unavailable; it does not run the command and hope. When the
machine offers nothing, the tool still runs the command -- refusing to run at
all would be its own kind of dishonesty -- but it says in the result, in plain
words, that the process was unconfined, and it says that before anyone reads
the output.

A second rule: **the command is never handed to a shell.** ``sandbox_run``
takes an argv list, which the OS execs directly, so nothing in the command is
expanded, globbed, word-split or interpreted. A caller who passes a single
string gets it split with :mod:`shlex` only when it contains no shell syntax at
all; anything with an operator in it is refused, because guessing what a string
was "meant" to say is how a sandbox gets turned into a shell.
"""

from __future__ import annotations

import ctypes
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = [
    "BACKENDS", "Backend", "BackendStatus", "BubblewrapBackend", "ContainerBackend",
    "Invocation", "LandlockBackend", "NoneBackend", "SeatbeltBackend", "detect",
    "sandbox_run", "select",
]

#: How much of a transcript is kept. A confined process can still print without
#: bound, and an unbounded capture is its own denial of service.
_MAX_OUTPUT_BYTES = 100_000

#: Timeout bounds, matching the core's own process runner.
_DEFAULT_TIMEOUT = 120.0
_MAX_TIMEOUT = 300.0

#: Characters that mean a string was written for a shell. A command containing
#: one of these is refused as a string rather than run with a guess at intent.
_SHELL_SYNTAX = set("|&;<>()$`\\\"'*?[]{}#~!\n\r")

#: Preference order for ``backend="auto"``. Landlock first because it is
#: enforced by the kernel with no helper binary, no setuid and no unprivileged
#: user namespace; bubblewrap needs the namespace, which distributions
#: increasingly disable; seatbelt is macOS-only.
_AUTO_ORDER = ("landlock", "bubblewrap", "seatbelt")


# --------------------------------------------------------------------------
# Landlock, through ctypes
# --------------------------------------------------------------------------

# The three Landlock syscalls were added in one commit with consecutive numbers
# (444, 445, 446) on every architecture that has them. Rather than trust a
# per-architecture table, the numbers are *probed*: the create syscall called
# with a null attr and the VERSION flag returns the ABI version it supports, and
# a kernel without Landlock answers -1 with ENOSYS. Detection is therefore a
# fact about this kernel rather than an assumption about this architecture.
_LANDLOCK_SYSCALL_BASE = 444
_LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
_LANDLOCK_MAX_REASONABLE_ABI = 32

# struct landlock_ruleset_attr { __u64 handled_access_fs; }
_RULESET_CTYPE = type("landlock_ruleset_attr", (ctypes.Structure,),
                      {"_fields_": [("handled_access_fs", ctypes.c_uint64)]})
# struct landlock_path_beneath_attr { __u64 allowed_access; __s32 parent_fd; }
_PATH_BENEATH_CTYPE = type("landlock_path_beneath_attr", (ctypes.Structure,),
                            {"_fields_": [("allowed_access", ctypes.c_uint64),
                                          ("parent_fd", ctypes.c_int)]})

_LANDLOCK_RULE_PATH_BENEATH = 1

# Access bits, in the order the ABI defines them. The later ones only exist from
# a given ABI version onwards, and asking an older kernel about a bit it does
# not know is how you get EINVAL instead of a sandbox.
_LL_EXECUTE = 1 << 0
_LL_WRITE_FILE = 1 << 1
_LL_READ_FILE = 1 << 2
_LL_READ_DIR = 1 << 3
_LL_REMOVE_DIR = 1 << 4
_LL_REMOVE_FILE = 1 << 5
_LL_MAKE_CHAR = 1 << 6
_LL_MAKE_DIR = 1 << 7
_LL_MAKE_REG = 1 << 8
_LL_MAKE_SOCK = 1 << 9
_LL_MAKE_FIFO = 1 << 10
_LL_MAKE_BLOCK = 1 << 11
_LL_MAKE_SYM = 1 << 12
_LL_REFER = 1 << 13          # ABI 2
_LL_TRUNCATE = 1 << 14       # ABI 3
_LL_IOCTL_DEV = 1 << 15      # ABI 5

#: What a read-only bind of the whole filesystem allows. Landlock has no "no
#: network" right -- the network is simply not something it governs, which is
#: why every backend's self-description says so out loud.
_RO_ACCESS = _LL_EXECUTE | _LL_READ_FILE | _LL_READ_DIR


def _landlock_bits(abi: int) -> int:
    """Every filesystem access right the running kernel actually implements."""
    bits = (_RO_ACCESS | _LL_WRITE_FILE | _LL_REMOVE_DIR | _LL_REMOVE_FILE
            | _LL_MAKE_CHAR | _LL_MAKE_DIR | _LL_MAKE_REG | _LL_MAKE_SOCK
            | _LL_MAKE_FIFO | _LL_MAKE_BLOCK | _LL_MAKE_SYM)
    if abi >= 2:
        bits |= _LL_REFER
    if abi >= 3:
        bits |= _LL_TRUNCATE
    if abi >= 5:
        bits |= _LL_IOCTL_DEV
    return bits


def _libc() -> Optional[Any]:
    """The process's own libc, or None. Never raises: a missing libc is an
    unavailable backend, not a crash."""
    try:
        return ctypes.CDLL(None, use_errno=True)
    except (OSError, AttributeError):  # pragma: no cover - platform dependent
        return None


def _raw_syscall(libc, number: int, *args: int) -> int:
    """Call a syscall by number. ``-1`` on failure, with errno set.

    Arguments are declared ``c_void_p`` because the Landlock calls take
    pointers, and a pointer and a small integer are the same width on every
    architecture this runs on -- the null attr of the version probe and the
    flag bit mean the same thing to the kernel either way.
    """
    libc.syscall.restype = ctypes.c_long
    libc.syscall.argtypes = [ctypes.c_long] + [ctypes.c_void_p] * len(args)
    ctypes.set_errno(0)
    return int(libc.syscall(number, *args))


def _landlock_probe() -> tuple[Optional[int], str]:
    """Ask the kernel which Landlock ABI it implements.

    Returns ``(abi, "")`` or ``(None, reason)``. This is the honest test: it
    asks, and takes the answer at face value, including "no".
    """
    if not sys.platform.startswith("linux"):
        return None, f"Landlock is a Linux kernel feature; this is {sys.platform}."
    libc = _libc()
    if libc is None or not hasattr(libc, "syscall"):
        return None, "libc could not be loaded, so the syscall cannot be called."
    for base in range(_LANDLOCK_SYSCALL_BASE, _LANDLOCK_SYSCALL_BASE + 3):
        result = _raw_syscall(libc, base, 0, 0, _LANDLOCK_CREATE_RULESET_VERSION)
        if 1 <= result <= _LANDLOCK_MAX_REASONABLE_ABI:
            return result, f"kernel implements Landlock ABI {result}"
    return None, ("the landlock_create_ruleset syscall is not available on this "
                  "kernel (it predates Landlock, or it is disabled)")


def _apply_landlock(workspace: Path) -> None:
    """Restrict the *calling* process, then return so it can exec.

    Landlock is applied to the calling thread and inherited across ``execve``,
    which is why this runs in the child between fork and exec rather than in
    the parent -- a ruleset set in the parent would confine the harness itself.

    The shape is the one ``bwrap --ro-bind / / --bind <ws> <ws>`` expresses:
    the whole filesystem readable and executable, the workspace writable. The
    more specific rule wins for the rights it handles, so a write inside the
    workspace is allowed even though the ``/`` rule does not grant writes.
    """
    abi, _ = _landlock_probe()
    if abi is None:
        raise OSError("Landlock became unavailable between detection and exec")
    libc = _libc()
    assert libc is not None
    create, add, restrict = (_LANDLOCK_SYSCALL_BASE, _LANDLOCK_SYSCALL_BASE + 1,
                             _LANDLOCK_SYSCALL_BASE + 2)
    handled = _landlock_bits(abi)

    # Landlock requires either no_new_privs or CAP_SYS_ADMIN. Asking for
    # no_new_privs is the unprivileged path and also means the confined process
    # cannot regain privilege through a setuid binary.
    libc.prctl(38, 1, 0, 0, 0)  # PR_SET_NO_NEW_PRIVS = 38

    ruleset = _RULESET_CTYPE()
    ruleset.handled_access_fs = handled
    ruleset_fd = _raw_syscall(libc, create, ctypes.byref(ruleset),
                              ctypes.sizeof(ruleset), 0)
    if ruleset_fd < 0:
        raise OSError("landlock_create_ruleset failed")

    try:
        for path, allowed in ((Path("/"), _RO_ACCESS), (workspace, handled)):
            parent = os.open(str(path), os.O_PATH | os.O_CLOEXEC)
            try:
                rule = _PATH_BENEATH_CTYPE()
                rule.allowed_access = allowed
                rule.parent_fd = parent
                if _raw_syscall(libc, add, ruleset_fd, _LANDLOCK_RULE_PATH_BENEATH,
                                ctypes.byref(rule), 0) < 0:
                    raise OSError(f"landlock_add_rule failed for {path}")
            finally:
                os.close(parent)
        if _raw_syscall(libc, restrict, ruleset_fd, 0) < 0:
            raise OSError("landlock_restrict_self failed")
    finally:
        os.close(ruleset_fd)


# --------------------------------------------------------------------------
# Backend protocol
# --------------------------------------------------------------------------


@dataclass
class Invocation:
    """A prepared argv, plus whatever has to be true in the child before exec."""
    argv: List[str]
    preexec: Optional[Callable[[], None]] = None
    #: Files to delete once the child has exited -- a Seatbelt profile is
    #: written to disk because ``sandbox-exec`` takes a path, not a string.
    tempfiles: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class BackendStatus:
    """What detection found, with the reason it found it."""
    name: str
    available: bool
    reason: str
    auto_selectable: bool
    summary: str

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "available": self.available,
                "reason": self.reason, "auto_selectable": self.auto_selectable}


class Backend:
    """One confinement mechanism.

    A backend answers two questions honestly: *can you confine anything here*
    and *what exactly did you confine*. The second is not optional -- a backend
    that cannot describe its own guarantee is not allowed to be selected.
    """
    name = "?"
    summary = ""
    #: Whether ``backend="auto"`` may pick this. The container backend is not:
    #: starting a daemon-backed container is a heavier thing to do on a model's
    #: say-so than restricting the process that is already running.
    auto_selectable = True

    def available(self, *, deep: bool = False) -> tuple[bool, str]:
        raise NotImplementedError

    def wrap(self, argv: Sequence[str], workspace: Path) -> Invocation:
        raise NotImplementedError

    def confinement(self, workspace: Path) -> str:
        """Plain words for what this backend actually did."""
        raise NotImplementedError

    def status(self, *, deep: bool = False) -> BackendStatus:
        """Detection that cannot raise. A backend whose probe blows up is an
        unavailable backend, not a traceback in the middle of a tool call."""
        try:
            available, reason = self.available(deep=deep)
        except Exception as exc:  # noqa: BLE001 - detection must not fail the run
            available, reason = False, f"detection raised {type(exc).__name__}: {exc}"
        return BackendStatus(self.name, bool(available), str(reason),
                             self.auto_selectable, self.summary)


# --------------------------------------------------------------------------
# The backends
# --------------------------------------------------------------------------


class LandlockBackend(Backend):
    """Linux, enforced by the kernel, no helper process and no root."""
    name = "landlock"
    summary = "Linux kernel Landlock, called through ctypes."
    auto_selectable = True

    def available(self, *, deep: bool = False) -> tuple[bool, str]:
        abi, reason = _landlock_probe()
        if abi is None:
            return False, reason
        return True, reason

    def wrap(self, argv: Sequence[str], workspace: Path) -> Invocation:
        # Applied in the child, between fork and exec. ctypes in a preexec_fn
        # is the same caveat as any preexec_fn: it must not depend on state
        # the parent's other threads could be mutating. It touches nothing but
        # a fresh libc handle and path_beneath rules.
        return Invocation(list(argv), preexec=lambda: _apply_landlock(workspace))

    def confinement(self, workspace: Path) -> str:
        return (f"landlock — every file on this system is readable and executable, "
                f"and only {workspace} is writable. The network is NOT restricted: "
                f"Landlock governs the filesystem and says nothing about sockets.")


class BubblewrapBackend(Backend):
    """Linux, via unprivileged user namespaces."""
    name = "bubblewrap"
    summary = "bwrap, in an unprivileged user namespace with no network."
    auto_selectable = True

    def available(self, *, deep: bool = False) -> tuple[bool, str]:
        if not sys.platform.startswith("linux"):
            return False, f"bwrap is a Linux tool; this is {sys.platform}."
        path = shutil.which("bwrap")
        if not path:
            return False, "the bwrap binary is not installed"
        if not hasattr(os, "unshare") and not Path("/proc/self/ns/user").exists():
            return False, "this kernel exposes no user namespaces for bwrap to use"
        if deep:
            # bwrap is installed but unusable if unprivileged user namespaces
            # are switched off, which is a common hardening default. Ask the
            # binary rather than guess: --ro-bind / / only works if the
            # namespace can be created at all.
            try:
                probe = subprocess.run(
                    [path, "--ro-bind", "/", "/", "--", "/bin/true"],
                    capture_output=True, timeout=15)
            except (OSError, subprocess.SubprocessError) as exc:
                return False, f"bwrap is installed but did not run: {exc}"
            if probe.returncode != 0:
                detail = probe.stderr.decode("utf-8", "replace").strip().splitlines()
                return False, ("bwrap could not create an unprivileged user "
                               f"namespace: {detail[-1] if detail else 'no reason given'}")
        return True, f"bwrap is installed at {path} and unprivileged user namespaces are in use"

    def wrap(self, argv: Sequence[str], workspace: Path) -> Invocation:
        bwrap = shutil.which("bwrap")
        if not bwrap:
            raise RuntimeError("bwrap disappeared between detection and exec")
        return Invocation([
            bwrap,
            "--unshare-net",
            "--ro-bind", "/", "/",
            "--bind", str(workspace), str(workspace),
            # Without this a killed wrapper can leave the confined process
            # running with nobody left to reap it.
            "--die-with-parent",
            "--",
            *argv,
        ])

    def confinement(self, workspace: Path) -> str:
        return (f"bubblewrap — the process is in its own user and mount namespace "
                f"with no network, the filesystem is read-only except {workspace}, "
                f"which is read-write.")


class SeatbeltBackend(Backend):
    """macOS, via the profile language ``sandbox-exec`` was built for."""
    name = "seatbelt"
    summary = "sandbox-exec with a generated profile: no network, writes confined."
    auto_selectable = True

    @staticmethod
    def _escape(value: str) -> str:
        """Escape a path for an SBPL string literal."""
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def profile(self, workspace: Path) -> str:
        """The SBPL profile, generated for one workspace.

        Seatbelt evaluates rules in order and the last match wins, so the
        default allow is stated first and the two denials after it -- which is
        also what makes the re-allow for the workspace mean anything.
        """
        return "\n".join([
            "(version 1)",
            "(allow default)",
            # Nothing may open a socket, bind a port or reach a name server.
            "(deny network*)",
            # Nothing may write, and then: everything except the workspace.
            "(deny file-write*)",
            f'(allow file-write* (subpath "{self._escape(str(workspace))}"))',
            # A process that cannot write /dev/null cannot report anything,
            # and that is a usability failure rather than a security one.
            '(allow file-write* (literal "/dev/null"))',
            '(allow file-write* (literal "/dev/stdout"))',
            "",
        ])

    def available(self, *, deep: bool = False) -> tuple[bool, str]:
        if sys.platform != "darwin":
            return False, f"seatbelt is a macOS facility; this is {sys.platform}."
        path = shutil.which("sandbox-exec")
        if not path:
            return False, "the sandbox-exec binary is not present"
        return True, f"sandbox-exec is present at {path}"

    def wrap(self, argv: Sequence[str], workspace: Path) -> Invocation:
        exec_path = shutil.which("sandbox-exec")
        if not exec_path:
            raise RuntimeError("sandbox-exec disappeared between detection and exec")
        handle, profile_path = tempfile.mkstemp(prefix="harness-seatbelt-", suffix=".sb")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(self.profile(workspace))
        return Invocation([exec_path, "-f", profile_path, *argv],
                          tempfiles=[profile_path])

    def confinement(self, workspace: Path) -> str:
        return (f"seatbelt — the process is denied the network outright and may "
                f"write only inside {workspace} (plus /dev/null and /dev/stdout). "
                f"Reads are unrestricted.")


class ContainerBackend(Backend):
    """docker or podman. Real isolation, real cost -- so it is opt-in."""
    name = "container"
    summary = "docker or podman, with no network and the workspace bind-mounted rw."
    #: Never automatic. A container runtime is a different trust domain, a
    #: different failure mode, and on most machines a daemon the harness did
    #: not start. Choosing it is the operator's call, made by naming it.
    auto_selectable = False

    #: Which image the command runs in. Overridable because the useful default
    #: depends entirely on what the command needs installed.
    IMAGE_ENV = "ADAPTIVE_HARNESS_SANDBOX_IMAGE"
    DEFAULT_IMAGE = "python:3.12-slim"

    def runtime(self) -> Optional[str]:
        for candidate in ("docker", "podman"):
            found = shutil.which(candidate)
            if found:
                return found
        return None

    def image(self) -> str:
        return os.environ.get(self.IMAGE_ENV) or self.DEFAULT_IMAGE

    def available(self, *, deep: bool = False) -> tuple[bool, str]:
        note = "opt-in: name it explicitly with backend=\"container\"; auto never selects it"
        if not sys.platform.startswith("linux"):
            return False, f"containers are configured for Linux; this is {sys.platform}."
        runtime = self.runtime()
        if not runtime:
            return False, f"neither docker nor podman is installed ({note})"
        if deep:
            # The binary being on PATH says nothing about the daemon being
            # reachable, and "installed" is not the question worth answering.
            try:
                probe = subprocess.run([runtime, "info"], capture_output=True, timeout=20)
            except (OSError, subprocess.SubprocessError) as exc:
                return False, f"{Path(runtime).name} is installed but did not answer: {exc}"
            if probe.returncode != 0:
                return False, (f"{Path(runtime).name} is installed but its daemon is not "
                               f"reachable ({note})")
        return True, f"{Path(runtime).name} is available using image {self.image()} ({note})"

    def wrap(self, argv: Sequence[str], workspace: Path) -> Invocation:
        runtime = self.runtime()
        if not runtime:
            raise RuntimeError("no container runtime disappeared between detection and exec")
        return Invocation([
            runtime, "run", "--rm",
            "--network", "none",
            "--volume", f"{workspace}:{workspace}",
            "--workdir", str(workspace),
            self.image(),
            *argv,
        ])

    def confinement(self, workspace: Path) -> str:
        return (f"container — the command runs in a fresh {self.image()} container with "
                f"no network at all, with {workspace} bind-mounted read-write and "
                f"nothing else from this machine visible.")


class NoneBackend(Backend):
    """The honest default: run the command, and admit that nothing confined it.

    This backend exists because the alternative is worse. A harness that
    refuses to run anything on a machine with no sandbox installed either gets
    uninstalled or gets a sandbox that is quietly switched off -- and the
    second one is indistinguishable from the first until the day it matters.
    A result that says "this ran unconfined, here is what that means" can be
    checked. A result that implies a sandbox that was never there cannot.
    """
    name = "none"
    summary = "No confinement. Runs the command with this process's own privileges."
    auto_selectable = True

    def available(self, *, deep: bool = False) -> tuple[bool, str]:
        return True, ("always available; it is always the last resort, and it "
                      "confines nothing")

    def wrap(self, argv: Sequence[str], workspace: Path) -> Invocation:
        return Invocation(list(argv))

    def confinement(self, workspace: Path) -> str:
        return ("none — NO CONFINEMENT WAS APPLIED. The command ran with the full "
                "privileges of this process: it could read and write any file this "
                "user can, and it could reach the network. Nothing was restricted. "
                "Do not run anything untrusted through this backend.")


#: Registration order, which is also the order detection reports in.
BACKENDS: tuple[type[Backend], ...] = (
    LandlockBackend, BubblewrapBackend, SeatbeltBackend, ContainerBackend, NoneBackend,
)

_BY_NAME: Dict[str, type[Backend]] = {backend.name: backend for backend in BACKENDS}


def detect(*, probe: bool = False) -> List[BackendStatus]:
    """Every backend, and whether it is actually usable here.

    ``probe=True`` makes the slower, deeper checks -- the ones that ask a
    binary to prove it works rather than asking ``PATH`` whether it exists. It
    is off by default because it costs seconds, and the auto-selection path
    only ever considers backends whose cheap check is the real check.
    """
    return [backend().status(deep=probe) for backend in BACKENDS]


def select(name: str = "auto", *, probe: bool = False) -> tuple[Backend, Optional[str]]:
    """The backend to use, and a note if the choice had to be given up.

    An unrecognised name raises ``ValueError`` rather than falling back. A
    caller who asked for ``landlock`` and got ``none`` has been told something
    false by a system that decided to be helpful.
    """
    if not isinstance(name, str):
        raise ValueError(f"backend must be a string, not {type(name).__name__}")
    key = name.strip().lower()
    if key not in _BY_NAME and key != "auto":
        raise ValueError(
            f"{name!r} is not a backend this plugin implements. Available: "
            f"{', '.join(sorted(_BY_NAME))}, auto. Nothing was run.")
    if key != "auto":
        backend = _BY_NAME[key]()
        status = backend.status(deep=probe)
        if not status.available:
            raise ValueError(
                f"the {key} backend is not usable on this machine: {status.reason}. "
                f"Nothing was run.")
        return backend, None
    for candidate in _AUTO_ORDER:
        status = _BY_NAME[candidate]().status(deep=probe)
        if status.available:
            return _BY_NAME[candidate](), None
    # Nothing confines anything here, so say so rather than pick a winner.
    return NoneBackend(), ("no confinement backend is available on this machine "
                           f"(tried: {', '.join(_AUTO_ORDER)})")


# --------------------------------------------------------------------------
# Command preparation
# --------------------------------------------------------------------------


def _normalise_command(command: Any) -> List[str]:
    """An argv list, always. Nothing here ever reaches a shell.

    A list is used as-is: the OS execs it directly, so ``;``, ``$HOME`` and
    ``*`` are just characters. A string is split with :mod:`shlex` only if it
    contains no shell syntax, because a string containing ``;`` is ambiguous by
    construction and an ambiguous command is not one to guess at.
    """
    if isinstance(command, str):
        if not command.strip():
            raise ValueError("command is empty")
        if set(command) & _SHELL_SYNTAX:
            raise ValueError(
                "the command is a string containing shell syntax "
                f"({''.join(sorted(set(command) & _SHELL_SYNTAX))}). This plugin never "
                "runs a command through a shell, so it will not guess what that string "
                "was meant to say. Pass an argv list instead, e.g. "
                '["sh", "-c", "..."] if you really do want a shell.')
        argv = shlex.split(command)
        if not argv:
            raise ValueError("command is empty")
        return argv
    if isinstance(command, (list, tuple)):
        argv = [str(part) for part in command]
        if not argv:
            raise ValueError("command is empty")
        return argv
    raise ValueError(f"command must be an argv list or a string, not {type(command).__name__}")


def _normalise_timeout(timeout: Any) -> float:
    try:
        value = float(timeout)
    except (TypeError, ValueError):
        raise ValueError(f"timeout must be a number, not {timeout!r}") from None
    if not 0 < value <= _MAX_TIMEOUT:
        raise ValueError(f"timeout must be between 0 and {_MAX_TIMEOUT:g} seconds")
    return value


def _normalise_workspace(value: Any) -> Path:
    """The directory writes are confined to. Defaults to the cwd.

    The filesystem root is refused: a backend asked to confine writes to ``/``
    has been asked to do nothing, and reporting that as confinement would be
    the exact lie this plugin exists to avoid.
    """
    if value in (None, "", "."):
        root = Path.cwd()
    elif isinstance(value, str):
        root = Path(value).expanduser()
    elif isinstance(value, Path):
        root = value
    else:
        raise ValueError(f"workspace must be a path, not {type(value).__name__}")
    try:
        root = root.resolve()
    except OSError as exc:
        raise ValueError(f"workspace could not be resolved: {exc}") from None
    if not root.is_dir():
        raise ValueError(f"workspace is not a directory: {root}")
    if root == Path(root.anchor):
        raise ValueError(
            f"workspace is the filesystem root ({root}); confining writes there "
            "restricts nothing, so it is refused rather than reported as a sandbox")
    return root


def _kill_group(process: subprocess.Popen) -> None:
    """Signal the whole process group, not just the process that was started.

    A confined command that forks children would otherwise leave them running
    after the timeout fired, which is the one case where a timeout matters
    most.
    """
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, AttributeError, OSError):
        try:
            process.kill()
        except OSError:
            pass


def _truncate(raw: bytes) -> str:
    text = raw[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
    if len(raw) > _MAX_OUTPUT_BYTES:
        text += f"\n[output truncated at {_MAX_OUTPUT_BYTES:,} bytes]"
    return text


def _run(invocation: Invocation, *, workspace: Path, timeout: float) -> Dict[str, Any]:
    """Execute an argv, bounded, with group-wide timeout cleanup."""
    try:
        process = subprocess.Popen(
            invocation.argv, cwd=str(workspace),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
            # Its own session means its own process group, so a timeout can
            # reach every descendant rather than only the process we spawned.
            start_new_session=True,
            preexec_fn=invocation.preexec,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"started": False, "error": f"the command could not be started: {exc}",
                "returncode": None, "stdout": "", "stderr": "", "timed_out": False}

    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(process)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - the group was killed
            process.kill()
            stdout, stderr = b"", b""
    return {"started": True, "error": "", "returncode": process.returncode,
            "stdout": _truncate(stdout or b""), "stderr": _truncate(stderr or b""),
            "timed_out": timed_out}


# --------------------------------------------------------------------------
# The tool
# --------------------------------------------------------------------------


def sandbox_run(command, backend="auto", timeout=120, workspace=None):
    """Run a command under the best OS-level confinement this machine offers.

    Returns the harness's tool result shape, with the important part at the
    top: which backend confined the command, or -- in the plainest words
    available -- that nothing did.
    """
    rejected = {"success": False, "output": "", "error": "", "metadata": {}}
    try:
        argv = _normalise_command(command)
        seconds = _normalise_timeout(timeout)
        root = _normalise_workspace(workspace)
    except ValueError as exc:
        return {**rejected, "error": str(exc)}

    try:
        chosen, downgrade = select(backend, probe=True)
    except ValueError as exc:
        # A backend name this plugin does not implement is refused outright.
        return {**rejected, "error": str(exc)}

    statuses = detect()
    note = chosen.confinement(root)
    banner = (f"backend: {note}"
              + (f"\n({downgrade})" if downgrade else ""))

    try:
        invocation = chosen.wrap(argv, root)
    except Exception as exc:  # noqa: BLE001 - a wrap failure must not claim a sandbox
        return {**rejected,
                "error": f"the {chosen.name} backend could not prepare the command: "
                         f"{type(exc).__name__}: {exc}. Nothing was run.",
                "metadata": {"backend": chosen.name, "confinement": "none applied",
                             "confinement_applied": False, "argv": argv,
                             "workspace": str(root)}}

    try:
        outcome = _run(invocation, workspace=root, timeout=seconds)
    finally:
        for leftover in invocation.tempfiles:
            try:
                os.unlink(leftover)
            except OSError:
                pass

    metadata: Dict[str, Any] = {
        "backend": chosen.name,
        "confinement": note,
        "confinement_applied": chosen.name != "none",
        "argv": argv,
        "workspace": str(root),
        "timeout_s": seconds,
        "returncode": outcome["returncode"],
        "timed_out": outcome["timed_out"],
        "backends": [status.as_dict() for status in statuses],
    }

    if not outcome["started"]:
        return {"success": False, "output": banner,
                "error": outcome["error"], "metadata": metadata}

    body = [banner]
    if outcome["stdout"]:
        body.append(f"--- stdout ---\n{outcome['stdout'].rstrip()}")
    if outcome["stderr"]:
        body.append(f"--- stderr ---\n{outcome['stderr'].rstrip()}")
    if outcome["timed_out"]:
        body.append(f"--- killed after {seconds:g}s; the whole process group "
                    f"was signalled ---")
    if not outcome["stdout"] and not outcome["stderr"]:
        body.append("(no output)")

    error = ""
    if outcome["timed_out"]:
        error = f"the command exceeded its {seconds:g}s timeout and its process group was killed"
    elif outcome["returncode"] != 0:
        error = f"the command exited with status {outcome['returncode']}"

    return {"success": not error, "output": "\n".join(body),
            "error": error, "metadata": metadata}
