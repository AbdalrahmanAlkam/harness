"""Plugin support: extend the harness without editing it.

A plugin can contribute a tool the model may call, a skill, a persistent
setting, a prompt override, and a slash command. The core consults
:class:`~adaptive_harness.plugins.host.PluginHost` for each of those, so a new
capability never requires a change to a file in ``src/adaptive_harness``.

Discovery looks in three places, in increasing precedence: the bundled
``plugins/`` directory, ``~/.config/adaptive-harness/plugins/``, and
``<project>/.harness/plugins/``.

A plugin is untrusted code. Its manifest declares the permissions it wants; the
user grants exactly those, and anything not granted is refused rather than
merely unused. A plugin that fails to load is reported and skipped rather than
being allowed to stop the harness from starting.

See ``docs/plugins.md`` for the manifest schema and a worked example.
"""

from adaptive_harness.plugins.host import (
    PERMISSIONS,
    PluginHost,
    PluginSetting,
    PluginTool,
)

__all__ = [
    "PERMISSIONS",
    "PluginHost",
    "PluginSetting",
    "PluginTool",
]
