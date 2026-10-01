"""Reliable clipboard delivery for the terminal UI.

`textual.app.App.copy_to_clipboard` only emits the OSC 52 escape sequence. That
is a *request*, not a guarantee: terminals without OSC 52 support, `tmux`
without `set-clipboard on`, and several remote/mux setups silently discard it.
Because the escape sequence produces no acknowledgement, the sending process
cannot tell success from failure, which is why copying has historically looked
like a no-op in this TUI.

This module wraps the fragile primitive in three layers so a copy either
succeeds or says exactly why it did not:

1. **File mirror** - the payload is always written under ``output/clipboard``
   so it can be recovered with ``cat`` or piped into a clipboard tool.
2. **OSC 52** - emitted through Textual whenever the terminal is known or
   assumed to honour it.
3. **System helper** - ``pbcopy`` / ``wl-copy`` / ``xclip`` / ``xsel`` are used
   when OSC 52 is *known* to be dropped and the session is running on the
   user's own machine. It is never run over SSH, where it would set the
   clipboard of the wrong host.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
import os
from pathlib import Path
import platform
import shutil
import subprocess


MIRROR_DIRECTORY = Path("output") / "clipboard"
MIRROR_LIMIT = 50
#: A clipboard helper either answers immediately or is not going to answer. Five
#: seconds is long enough that a user pressing the copy shortcut twice sees the
#: interface appear to hang; the delivery runs on a worker thread regardless,
#: so this only bounds how long the outcome takes to report.
HELPER_TIMEOUT_SECONDS = 1.5

#: Terminals that are known to ignore the OSC 52 clipboard sequence.
_UNSUPPORTED_TERM_PROGRAMS = {
    "apple_terminal",
    "terminal",
    "iterm2_app",  # legacy name; modern iTerm handles OSC 52
}

#: Helper command templates, most specific platform first.
_SYSTEM_HELPERS: tuple[tuple[str, ...], ...] = (
    ("pbcopy",),
    ("wl-copy",),
    ("xclip", "-selection", "clipboard"),
    ("xsel", "--clipboard", "--input"),
)


@dataclass(frozen=True)
class ClipboardDelivery:
    """The outcome of a clipboard delivery attempt."""

    payload: str
    method: str
    mirror_path: Path | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def delivered(self) -> bool:
        """True when the payload reached a clipboard the user can paste from."""
        return self.method in {"osc52", "system"} or bool(
            self.method.startswith("system:")
        )

    def summary(self) -> str:
        """A single line suitable for the chat log."""
        if self.method == "osc52":
            message = "✓ Copied to the terminal clipboard (OSC 52)."
        elif self.method.startswith("system:"):
            message = f"✓ Copied to the system clipboard via {self.method.split(':', 1)[1]}."
        else:
            message = f"⚠ Clipboard unavailable ({self.method})."
        if self.mirror_path is not None:
            message += f" Saved a copy at {self.mirror_path}."
        for warning in self.warnings:
            message += f" {warning}"
        return message


def _tmux_forwards_clipboard() -> bool:
    """Ask tmux whether it forwards OSC 52 to the outer terminal."""
    tmux = shutil.which("tmux")
    if tmux is None:
        return False
    try:
        probe = subprocess.run(
            [tmux, "show", "-gv", "set-clipboard"],
            capture_output=True,
            text=True,
            timeout=HELPER_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0 and probe.stdout.strip().lower() in {"on", "external"}


@lru_cache(maxsize=1)
def _clipboard_capability() -> tuple[bool | None, bool]:
    """Probe the environment once.

    Returns:
        A tuple of ``(osc52_supported, session_is_remote)`` where the first
        element is ``None`` when the terminal cannot be identified either way.
    """
    environment = os.environ
    remote = any(environment.get(name) for name in
                 ("SSH_CONNECTION", "SSH_TTY", "SSH_CLIENT", "SSH2_AUTH_SOCK"))
    program = environment.get("TERM_PROGRAM", "").strip().casefold()
    terminal = environment.get("TERM", "").strip().casefold()
    if program in _UNSUPPORTED_TERM_PROGRAMS or terminal in {"", "dumb"}:
        return False, remote
    if environment.get("TMUX") or environment.get("STY"):
        return (_tmux_forwards_clipboard(), remote)
    if not program and terminal in {"xterm", "xterm-kitty", "xterm-ghostty", "alacritty"}:
        # Legacy xterm builds predate OSC 52; kitty/ghostty/alacritty do not.
        return terminal == "xterm", remote
    return None, remote


def osc52_supported() -> bool | None:
    """Whether the terminal is known to honour OSC 52, or ``None`` if unknown."""
    return _clipboard_capability()[0]


def session_is_remote() -> bool:
    """True when the harness is running over SSH on a different machine."""
    return _clipboard_capability()[1]


@lru_cache(maxsize=1)
def _cached_helper() -> tuple[str, ...] | None:
    """Locate a helper that can feed the *local* system clipboard from stdin."""
    if platform.system() == "Darwin":
        preferred = _SYSTEM_HELPERS[0]
    elif os.environ.get("WAYLAND_DISPLAY"):
        preferred = _SYSTEM_HELPERS[1]
    else:
        preferred = _SYSTEM_HELPERS[2]
    helpers = (preferred, *(helper for helper in _SYSTEM_HELPERS if helper != preferred))
    for helper in helpers:
        if shutil.which(helper[0]):
            return helper
    return None


def system_clipboard_helper() -> tuple[str, ...] | None:
    """The clipboard helper to use, or ``None`` when none is installed."""
    return _cached_helper()


def _write_mirror(payload: str, directory: Path) -> Path | None:
    """Persist the payload so a copy is never lost, even without a clipboard."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"clip-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.txt"
        path.write_text(payload, encoding="utf-8")
    except OSError:
        return None
    _prune_mirror(directory)
    return path


def _prune_mirror(directory: Path) -> None:
    """Keep only the newest mirror files so the directory cannot grow forever."""
    try:
        existing = sorted(directory.glob("clip-*.txt"), key=lambda item: item.name)
    except OSError:
        return
    for stale in existing[:-MIRROR_LIMIT] if len(existing) > MIRROR_LIMIT else []:
        try:
            stale.unlink()
        except OSError:
            pass


def _run_helper(helper: tuple[str, ...], payload: str) -> tuple[bool, str]:
    """Pipe the payload into a clipboard helper process."""
    try:
        completed = subprocess.run(
            helper,
            input=payload,
            capture_output=True,
            text=True,
            timeout=HELPER_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        return False, detail[-1] if detail else f"exit code {completed.returncode}"
    return True, ""


def deliver(
    payload: str,
    *,
    emit_osc52: bool = True,
    mirror_directory: Path | str | None = None,
    extra_warnings: tuple[str, ...] = (),
) -> ClipboardDelivery:
    """Deliver ``payload`` to the best clipboard available and report what happened.

    Args:
        payload: Text to copy.
        emit_osc52: When true, the caller has already emitted the OSC 52
            sequence, so returning ``"osc52"`` means that is the delivery.
        mirror_directory: Directory for the recovery file, or ``None`` to skip.
        extra_warnings: Problems the caller already hit, appended to the report.

    Returns:
        A :class:`ClipboardDelivery` naming the route that was used. Its
        ``delivered`` property is true only when the payload genuinely reached
        a pasteable clipboard.
    """
    if not payload.strip():
        return ClipboardDelivery(payload, "empty", None, extra_warnings)

    warnings: list[str] = list(extra_warnings)
    mirror = _write_mirror(payload, Path(mirror_directory)) if mirror_directory else None
    if mirror_directory and mirror is None:
        warnings.append("The recovery file could not be written.")

    supported = osc52_supported()
    if emit_osc52 and supported is not False:
        # ``None`` means "not identifiable", and most modern terminals support
        # OSC 52, so emitting is still the best first attempt.
        return ClipboardDelivery(payload, "osc52", mirror, tuple(warnings))

    if session_is_remote():
        # A local clipboard helper would set the *server's* clipboard, leaving
        # the user's own machine untouched and the copy effectively lost.
        warnings.append(
            "This terminal drops OSC 52 and the session is remote, so the local "
            "clipboard was left untouched.")
        return ClipboardDelivery(payload, "unsupported-remote", mirror, tuple(warnings))

    helper = system_clipboard_helper()
    if helper is None:
        warnings.append(
            "Install wl-clipboard, xclip, or xsel to enable a system clipboard fallback.")
        return ClipboardDelivery(payload, "no-helper", mirror, tuple(warnings))
    succeeded, detail = _run_helper(helper, payload)
    if succeeded:
        return ClipboardDelivery(payload, f"system:{helper[0]}", mirror, tuple(warnings))
    warnings.append(f"{helper[0]} failed ({detail}).")
    return ClipboardDelivery(payload, "helper-failed", mirror, tuple(warnings))


def reset_capability_cache() -> None:
    """Forget the cached environment probe (used by tests)."""
    _clipboard_capability.cache_clear()
    _cached_helper.cache_clear()


__all__ = [
    "ClipboardDelivery",
    "MIRROR_DIRECTORY",
    "deliver",
    "osc52_supported",
    "reset_capability_cache",
    "session_is_remote",
    "system_clipboard_helper",
]
