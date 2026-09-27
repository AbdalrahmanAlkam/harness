"""One screen for every persistent setting, instead of a command per setting.

The harness grew a command and a function key for each knob, so the set of
things a user can change was scattered across the command palette, the status
line, and a set of modal pickers. Finding "which settings exist" meant knowing
the whole surface.

This screen is the single place that answers it: every setting is a row with its
name, its current value, and a way to change it. Rows for a small set of values
cycle on Enter, which needs no second dialog, while models open the existing
searchable picker so a long catalogue stays navigable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Label, Static

#: Settings whose value is one of a short fixed set. These cycle in place, so
#: changing them costs one keypress instead of a second dialog.
CYCLING = {
    "mode": ("coding", "research", "science", "audit", "auto"),
    "thinking": ("auto", "none", "low", "medium", "high", "xhigh", "max"),
    "safety": ("turbo", "balanced", "cautious", "strict"),
    "step_policy": ("classifier", "fixed", "unbounded"),
    "swarm_mode": ("off", "auto", "on"),
    "isolation_mode": ("off", "auto", "on"),
    "output_filter": ("on", "off"),
}

DESCRIPTIONS = {
    "provider": "Where the task model is sent",
    "model": "The model that does the thinking",
    "secondary_model": "Cheap model that compresses noisy tool output (falls back to the model above)",
    "mode": "Which domain guidance is applied",
    "thinking": "Reasoning budget",
    "safety": "How often the agent checks in",
    "step_policy": "How tool-step limits are chosen",
    "max_steps": "Hard cap on tool steps, or unlimited",
    "swarm_mode": "Multi-agent delegation",
    "isolation_mode": "Git worktree review for edits",
    "output_filter": "Compress low-value tool output with the secondary model",
    "classifier": "Local engine that scores complexity and domain",
    "theme": "Terminal colours",
}


@dataclass(frozen=True)
class Setting:
    """One row: a key, the value to show, and how to change it.

    ``kind`` decides the interaction:

    * ``cycle`` -- a short fixed set, advanced in place on Enter.
    * ``model`` -- opens the app's existing searchable model picker, either for
      the primary model or for the secondary one.
    * ``command`` -- delegates to the command that already owns this setting, so
      the screen never becomes a second implementation of it.
    """

    key: str
    label: str
    value: str
    kind: str = "cycle"
    choices: tuple[str, ...] = ()


class SettingsRow(Static):
    """One selectable line in the settings list.

    A ``Button`` cannot be used here: its component stylesheet sets
    ``border: tall`` and ``line-pad: 1``, and a row sized to two cells has
    *zero* content rows, so the background painted as a highlighted bar while
    the label was never drawn. Worse, selecting a row with ``variant`` adds
    ``-primary``, whose ``border: tall $primary`` outranks a plain
    ``.settings-row`` rule, so the border came back. A focusable ``Static`` has
    no such hidden geometry: one cell, one line of text, always visible.
    """

    can_focus = True

    def on_click(self, event: events.Click) -> None:
        event.stop()
        screen = self.screen
        if isinstance(screen, SettingsScreen):
            screen.select_at(int((self.id or "row-0").removeprefix("row-")))


class SettingsScreen(ModalScreen[str | None]):
    """Browse and change every persistent setting from one place."""

    DEFAULT_CSS = """
    SettingsScreen { align: center middle; background: rgba(0, 0, 0, 0.70); }
    #settings-card { width: 88%; max-width: 100; height: 82%; max-height: 34;
                     background: #1e1e2e; color: #ffffff; border: round #89b4fa; padding: 1 2; }
    #settings-title { height: 2; text-align: center; text-style: bold; color: $accent; }
    #settings-rows { height: 1fr; background: #202b3a; }
    #settings-help { height: 1; color: $text-muted; text-align: center; }
    SettingsRow { width: 100%; height: 1; min-height: 1; padding: 0 1;
                  background: #202b3a; color: #ffffff;
                  text-align: left; content-align: left middle; }
    SettingsRow:hover, SettingsRow:focus, SettingsRow.selected {
                  background: #274c77; color: #ffffff; text-style: bold; }
    #settings-detail { height: 2; color: #cdd6f4; background: #313244; padding: 0 1; }
    """

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("enter", "activate", "Change", show=False),
        Binding("up", "previous", "Previous", show=False),
        Binding("down", "next", "Next", show=False),
    ]

    def __init__(self, settings: list[Setting], *,
                 on_change: Callable[[str, str], None] | None = None,
                 on_open: Callable[[str], None] | None = None):
        super().__init__()
        self.settings = settings
        self.on_change = on_change
        self.on_open = on_open
        self.selected = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-card"):
            yield Label("Settings", id="settings-title")
            yield VerticalScroll(id="settings-rows")
            yield Static("", id="settings-detail")
            yield Static("↑↓ move · Enter change · Esc close", id="settings-help")

    async def on_mount(self) -> None:
        await self._render_rows()
        if self.settings:
            self._focus_row()

    async def _render_rows(self) -> None:
        rows = self.query_one("#settings-rows", VerticalScroll)
        await rows.remove_children()
        await rows.mount_all(
            SettingsRow(self._row_text(index), id=f"row-{index}",
                        classes="selected" if index == self.selected else "")
            for index in range(len(self.settings)))

    async def refresh_rows(self) -> None:
        """Re-render after the app changed a value behind this screen's back."""
        self.selected = min(self.selected, max(0, len(self.settings) - 1))
        await self._render_rows()
        self._update_detail()
        if self.settings:
            self._focus_row()

    def _row_text(self, index: int) -> str:
        setting = self.settings[index]
        marker = "▸" if index == self.selected else " "
        return f"{marker} {setting.label:<16} {setting.value}"

    def _update_detail(self) -> None:
        if not self.settings:
            return
        setting = self.settings[self.selected]
        hint = {"cycle": "Enter cycles", "model": "Enter opens the model picker",
                "command": "Enter opens this setting"}.get(setting.kind, "")
        self.query_one("#settings-detail", Static).update(
            Text(f"{DESCRIPTIONS.get(setting.key, '')}  ·  {hint}"))

    def _mark_selected(self, index: int) -> None:
        for position in range(len(self.settings)):
            row = self.query(f"SettingsRow#row-{position}")
            if not row:
                continue
            row = row.first()
            row.remove_class("selected")
            if position == index:
                row.add_class("selected")
            if position == index:
                row.update(self._row_text(position))

    def _replace(self, index: int, value: str) -> None:
        self.settings[index] = Setting(self.settings[index].key, self.settings[index].label,
                                       value, self.settings[index].kind,
                                       self.settings[index].choices)

    def _focus_row(self) -> None:
        if not self.settings:
            return
        self.query_one(f"SettingsRow#row-{self.selected}").focus()

    def select_at(self, index: int) -> None:
        """Point the selection at a row, from a click or from the keyboard."""
        if not self.settings or not 0 <= index < len(self.settings):
            return
        self._mark_selected(index)
        self.selected = index
        self._update_detail()
        self._focus_row()

    def _move(self, delta: int) -> None:
        if not self.settings:
            return
        self.select_at(max(0, min(self.selected + delta, len(self.settings) - 1)))
        self.query_one(f"SettingsRow#row-{self.selected}").scroll_visible()

    def action_previous(self) -> None:
        self._move(-1)

    def action_next(self) -> None:
        self._move(1)

    def action_activate(self) -> None:
        if self.settings:
            self._activate(self.settings[self.selected])

    def on_settings_row_focused(self, event: events.Focus) -> None:
        widget = event.widget
        if isinstance(widget, SettingsRow) and widget.id and widget.id.startswith("row-"):
            index = int(widget.id.removeprefix("row-"))
            if index != self.selected:
                self._mark_selected(index)
                self.selected = index
                self._update_detail()

    def _activate(self, setting: Setting) -> None:
        if setting.kind == "cycle" and setting.choices:
            current = setting.value if setting.value in setting.choices else setting.choices[0]
            following = setting.choices[(setting.choices.index(current) + 1) % len(setting.choices)]
            self._replace(self.selected, following)
            if self.on_change is not None:
                self.on_change(setting.key, following)
            self.query_one(f"SettingsRow#row-{self.selected}").update(self._row_text(self.selected))
            return
        # Anything richer is owned by the app, which already has a working flow
        # for it. This screen steps aside rather than reimplementing it.
        if self.on_open is not None:
            self.app.pop_screen()
            self.on_open(setting.key)

    def action_close(self) -> None:
        self.dismiss(None)
