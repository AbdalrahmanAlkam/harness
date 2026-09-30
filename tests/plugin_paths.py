"""Locating a plugin for a test, wherever it happens to live.

The core ships with no plugins -- they are installed, not bundled -- so a test
cannot hard-code a path inside the package. This resolves one from the checkout's
`plugins/` directory, falling back to the installed location, so a plugin test
does not break when the plugin moves or when the harness is installed without
the checkout.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Where a plugin may be, most specific first.
SEARCH_PATHS = (
    REPO / "plugins",
    REPO / "src" / "adaptive_harness" / "plugins" / "bundled",
)


def plugin_path(name: str) -> Path:
    """The directory of the named plugin, or a path that will fail loudly.

    Raising with the searched locations beats returning something that exists
    but is empty: a test that silently loaded nothing would pass for the wrong
    reason.
    """
    for root in SEARCH_PATHS:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    searched = "\n  ".join(str(path) for path in SEARCH_PATHS)
    raise FileNotFoundError(
        f"Could not find the {name!r} plugin. Searched:\n  {searched}")


def all_plugin_names() -> list[str]:
    return sorted(child.name for child in SEARCH_PATHS[0].iterdir()
                  if child.is_dir()) if SEARCH_PATHS[0].is_dir() else []
