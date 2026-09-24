"""Custom Textual widgets for classifier telemetry, clarification modal, and stream views."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from rich.panel import Panel
from rich.console import Group
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from adaptive_harness.llm.client import MODEL_TIERS
from textual.app import ComposeResult
from textual import events
from textual.message import Message
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, RichLog, Static, TextArea


SKILL_LABELS = {
    "code_edit": "🛠️  Code Edit",
    "run_command": "⚡  Run Command",
    "search_explore": "🔍  Search & Explore",
    "testing": "🧪  Run Tests",
    "ask_clarification": "❓  Clarification",
    "general_reasoning": "🧠  Reasoning",
}

THEME_CHOICES = (
    "textual-dark", "nord", "tokyo-night", "dracula",
    "catppuccin-mocha", "gruvbox", "monokai", "textual-light",
)


class ClassifierTelemetryWidget(Static):
    """Real-time classifier telemetry panel displaying live probabilities, entropy, and tier."""

    DEFAULT_CSS = """
    ClassifierTelemetryWidget {
        width: 52;
        height: 100%;
        background: $surface;
        border-left: solid $primary;
        padding: 1;
        overflow-y: auto;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.probabilities: Dict[str, float] = {skill: 0.0 for skill in SKILL_LABELS}
        self.entropy: float = 0.0
        self.margin: float = 0.0
        self.risk_level: str = "idle"
        self.tier: str = "—"
        self.active_model: str = MODEL_TIERS["standard"]
        self.primary_skill: str = "none"
        self.specialized_skills: list[dict[str, str]] = []
        self.skill_confidence = 0.0
        self.skill_tools: list[str] = []
        self.selection: str = "auto"
        self.classifier_engine: str = "sklearn"
        self.classifier_model: str = "TF-IDF + Logistic Regression"
        self.classifier_latency_ms: float = 0.0
        self.domain_mode: str = "—"
        self.domain_selection: str = "auto"
        self.thinking_level: str = "—"
        self.thinking_tokens: int = 0
        self.thinking_selection: str = "auto"
        self.tokens_saved_estimate = 0
        self.provider_cached_tokens = 0
        self.provider_prompt_tokens = 0
        self.memory_resolution = False
        self.overseer_state = "HEALTHY_PROGRESS"
        self.overseer_latency_ms = 0.0
        self.overseer_tier = "gate"
        self.context_used = 0
        self.context_capacity = 0
        self.context_compacted = 0

    def reset_telemetry(self, *, classifier_engine: str | None = None,
                        classifier_model: str | None = None, model: str | None = None,
                        selection: str | None = None) -> None:
        self.probabilities = {skill: 0.0 for skill in SKILL_LABELS}
        self.entropy = 0.0
        self.margin = 0.0
        self.risk_level = "idle"
        self.tier = "—"
        self.primary_skill = "none"
        self.specialized_skills = []
        self.skill_confidence = 0.0
        self.skill_tools = []
        self.classifier_latency_ms = 0.0
        self.domain_mode = "—"
        self.domain_selection = "auto"
        self.thinking_level = "—"
        self.thinking_tokens = 0
        self.thinking_selection = "auto"
        self.tokens_saved_estimate = 0
        self.provider_cached_tokens = 0
        self.provider_prompt_tokens = 0
        self.memory_resolution = False
        self.overseer_state = "HEALTHY_PROGRESS"
        self.overseer_latency_ms = 0.0
        self.overseer_tier = "gate"
        self.context_used = 0
        self.context_capacity = 0
        self.context_compacted = 0
        if classifier_engine is not None:
            self.classifier_engine = classifier_engine
        if classifier_model is not None:
            self.classifier_model = classifier_model
        if model is not None:
            self.active_model = model
        if selection is not None:
            self.selection = selection
        self.refresh()

    def update_telemetry(
        self,
        probabilities: Optional[Dict[str, float]] = None,
        entropy: Optional[float] = None,
        margin: Optional[float] = None,
        risk_level: Optional[str] = None,
        tier: Optional[str] = None,
        model: Optional[str] = None,
        primary_skill: Optional[str] = None,
        specialized_skills: Optional[list[dict[str, str]]] = None,
        skill_confidence: Optional[float] = None,
        skill_tools: Optional[list[str]] = None,
        selection: Optional[str] = None,
        classifier_engine: Optional[str] = None,
        classifier_model: Optional[str] = None,
        classifier_latency_ms: Optional[float] = None,
        domain_mode: Optional[str] = None,
        domain_selection: Optional[str] = None,
        thinking_level: Optional[str] = None,
        thinking_tokens: Optional[int] = None,
        thinking_selection: Optional[str] = None,
        tokens_saved_estimate: Optional[int] = None,
        provider_cached_tokens: Optional[int] = None,
        provider_prompt_tokens: Optional[int] = None,
        memory_resolution: Optional[bool] = None,
        overseer_state: Optional[str] = None,
        overseer_latency_ms: Optional[float] = None,
        overseer_tier: Optional[str] = None,
        context_used: Optional[int] = None,
        context_capacity: Optional[int] = None,
        context_compacted: Optional[int] = None,
    ) -> None:
        if probabilities is not None:
            self.probabilities = probabilities
        if entropy is not None:
            self.entropy = entropy
        if margin is not None:
            self.margin = margin
        if risk_level is not None:
            self.risk_level = risk_level
        if tier is not None:
            self.tier = tier
        if model is not None:
            self.active_model = model
        if primary_skill is not None:
            self.primary_skill = primary_skill
        if specialized_skills is not None:
            self.specialized_skills = specialized_skills
        if skill_confidence is not None:
            self.skill_confidence = skill_confidence
        if skill_tools is not None:
            self.skill_tools = skill_tools
        for key, value in (("selection", selection), ("classifier_engine", classifier_engine),
                           ("classifier_model", classifier_model), ("classifier_latency_ms", classifier_latency_ms),
                           ("domain_mode", domain_mode), ("domain_selection", domain_selection),
                           ("thinking_level", thinking_level), ("thinking_tokens", thinking_tokens),
                           ("thinking_selection", thinking_selection),
                           ("tokens_saved_estimate", tokens_saved_estimate),
                           ("provider_cached_tokens", provider_cached_tokens),
                           ("provider_prompt_tokens", provider_prompt_tokens),
                           ("memory_resolution", memory_resolution)):
            if value is not None:
                setattr(self, key, value)

        for key, value in (("overseer_state", overseer_state),
                           ("overseer_latency_ms", overseer_latency_ms),
                           ("overseer_tier", overseer_tier), ("context_used", context_used),
                           ("context_capacity", context_capacity),
                           ("context_compacted", context_compacted)):
            if value is not None:
                setattr(self, key, value)

        self.refresh()

    def render(self) -> Panel:
        status = Text(overflow="fold")
        status.append("MODEL  ", style="bold cyan")
        status.append(f"{self.selection.upper()} · {self.tier.upper()}\n", style="bold magenta")
        status.append(self.active_model + "\n")
        engine_name = {"semif": "SemIf", "sklearn": "SKLearn", "ollama": "Ollama",
                       "local-slm": "Local SLM", "onnx": "ONNX", "openrouter": "OpenRouter"}.get(
                           self.classifier_engine.lower(), self.classifier_engine)
        model_name = self.classifier_model.rstrip("/").rsplit("/", 1)[-1] if self.classifier_model else ""
        engine_color = "bold cyan" if self.classifier_engine.lower() == "semif" else "bold green"
        status.append("Engine: ", style="bold cyan")
        status.append(f"{engine_name} ({model_name})\n", style=engine_color)
        status.append(f"Latency: {self.classifier_latency_ms:.2f} ms\n", style="cyan")
        overseer_color = "bold green" if self.overseer_state == "HEALTHY_PROGRESS" else "bold yellow"
        status.append("Overseer: ", style="bold cyan")
        status.append(self.overseer_state.replace("_", " ") + "\n", style=overseer_color)
        status.append(f"  {self.overseer_tier} · {self.overseer_latency_ms:.2f} ms\n", style="dim")
        if self.context_capacity:
            fraction = min(1.0, self.context_used / self.context_capacity)
            bars = round(fraction * 10)
            gauge_color = "bold yellow" if fraction >= 0.75 else "cyan"
            status.append("Context: ", style="bold cyan")
            status.append("█" * bars + "░" * (10 - bars), style=gauge_color)
            status.append(f" {self.context_used // 1000}k/{self.context_capacity // 1000}k ({fraction:.0%})\n",
                          style=gauge_color)
            if self.context_compacted:
                status.append(f"⚡ Compacted ~{self.context_compacted:,} tokens\n", style="bold green")
        domain_label = "SECURITY" if self.domain_mode == "audit" else self.domain_mode.upper()
        domain_color = {"coding": "bold cyan", "research": "bold magenta", "science": "bold green",
                        "audit": "bold red"}.get(self.domain_mode, "bold cyan")
        status.append("Domain Mode: ", style="bold cyan")
        status.append(f"[{domain_label}]", style=domain_color)
        status.append(f" {self.domain_selection.upper()}\n", style="dim")
        budget_label = f"{self.thinking_tokens // 1000}k" if self.thinking_tokens >= 1000 else "0"
        status.append("Thinking Level: ", style="bold cyan")
        status.append(f"[{self.thinking_level.upper()} ({budget_label} tokens)]", style="bold yellow")
        status.append(f" {self.thinking_selection.upper()}\n", style="dim")
        status.append(f"Skill  {self.primary_skill}\n")
        if self.specialized_skills:
            for skill in self.specialized_skills:
                status.append("Active Skill: ", style="bold cyan")
                status.append(f"{skill['icon']} {skill['title']}\n", style="bold green")
                status.append(f"  {skill['category']} · {self.skill_confidence:.0%}\n", style="cyan")
            status.append("Tools: " + ", ".join(self.skill_tools) + "\n", style="dim")
        else:
            status.append("Active Skill: automatic general\n", style="dim")
        status.append(f"H(p)  {self.entropy:.3f} bits   Margin  {self.margin*100:.1f}%\n", style="cyan")
        risk_style = "cyan" if self.risk_level == "idle" else "green" if self.risk_level == "low" else "yellow" if self.risk_level == "medium" else "bold red"
        status.append(f"Risk  {self.risk_level.upper()}", style=risk_style)
        status.append(f"\nContext saved: ~{self.tokens_saved_estimate:,} tokens", style="bold green")
        cache_pct = (self.provider_cached_tokens / self.provider_prompt_tokens * 100
                     if self.provider_prompt_tokens else 0)
        status.append(f"\nProvider cache: {self.provider_cached_tokens:,} tokens ({cache_pct:.0f}%)", style="cyan")
        if self.memory_resolution:
            status.append("\nResolution: ⚡ Verified memory (0 API tokens)", style="bold green")

        bars_table = Table(box=None, show_header=False, expand=False, padding=(0, 1))
        bars_table.add_column("Skill", width=21, no_wrap=True)
        bars_table.add_column("Bar", width=10, no_wrap=True)
        bars_table.add_column("Pct", justify="right", width=6, no_wrap=True)

        sorted_skills = sorted(self.probabilities.items(), key=lambda x: x[1], reverse=True)
        for skill, prob in sorted_skills:
            bar_len = max(0, min(10, round(prob * 10)))
            bar_str = "█" * bar_len + "░" * (10 - bar_len)
            c = "green" if prob >= 0.4 else ("yellow" if prob >= 0.15 else "bright_black")
            label = SKILL_LABELS.get(skill, skill.replace("_", " ").title())
            bars_table.add_row(Text(label), Text(bar_str, style=c), Text(f"{prob*100:5.1f}%"))

        return Panel(Group(status, Rule(style="cyan"), Text("SKILL PROBABILITIES", style="bold cyan"), bars_table),
                     title="[bold cyan]ROUTING[/bold cyan]", border_style="cyan")


class HistoryInput(Input):
    """Prompt input with shell-like Up/Down history and draft restoration."""

    def __init__(self, *args, history: Optional[List[str]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = list(history or [])
        self._history_index: int | None = None
        self._draft = ""

    def reset_navigation(self) -> None:
        self._history_index = None
        self._draft = ""

    def on_key(self, event: events.Key) -> None:
        if event.key == "up" and self.history:
            if self._history_index is None:
                self._draft = self.value
                self._history_index = len(self.history) - 1
            else:
                self._history_index = max(0, self._history_index - 1)
            self.value = self.history[self._history_index]
            self.cursor_position = len(self.value)
            event.prevent_default()
            event.stop()
        elif event.key == "down" and self._history_index is not None:
            if self._history_index >= len(self.history) - 1:
                self.value = self._draft
                self.reset_navigation()
            else:
                self._history_index += 1
                self.value = self.history[self._history_index]
            self.cursor_position = len(self.value)
            event.prevent_default()
            event.stop()
        elif self._history_index is not None and (event.character or event.key in {"backspace", "delete"}):
            self.reset_navigation()


class PinnedRichLog(RichLog):
    """A log that stops following output while the reader scrolls upward."""

    ALLOW_SELECT = True

    class PinChanged(Message):
        def __init__(self, pinned: bool):
            super().__init__()
            self.pinned = pinned

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pinned = False

    def _set_pinned(self, pinned: bool) -> None:
        if self.pinned != pinned:
            self.pinned = pinned
            self.auto_scroll = not pinned
            self.post_message(self.PinChanged(pinned))

    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        super()._on_mouse_scroll_up(event)
        self._set_pinned(True)

    def _on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        super()._on_mouse_scroll_down(event)
        self.call_after_refresh(self._unpin_if_bottom)

    def _unpin_if_bottom(self) -> None:
        if self.is_vertical_scroll_end:
            self._set_pinned(False)

    def action_scroll_end(self) -> None:
        super().action_scroll_end()
        self._set_pinned(False)

    def action_scroll_up(self) -> None:
        super().action_scroll_up()
        self._set_pinned(True)

    def action_page_up(self) -> None:
        super().action_page_up()
        self._set_pinned(True)

    def action_scroll_home(self) -> None:
        super().action_scroll_home()
        self._set_pinned(True)

    def action_scroll_down(self) -> None:
        super().action_scroll_down()
        self.call_after_refresh(self._unpin_if_bottom)

    def action_page_down(self) -> None:
        super().action_page_down()
        self.call_after_refresh(self._unpin_if_bottom)


class ThemeOption(Button):
    def __init__(self, theme_name: str, index: int):
        super().__init__(theme_name.replace("-", " ").title(), id=f"theme-{index}", classes="theme-option")
        self.theme_name = theme_name

    def _preview(self) -> None:
        if isinstance(self.screen, ThemePickerModal):
            self.screen.preview(self.theme_name)

    def on_enter(self, event: events.Enter) -> None:
        self._preview()

    def on_focus(self, event: events.Focus) -> None:
        self._preview()


class ThemePickerModal(ModalScreen[str | None]):
    """Preview themes on hover/focus; Enter saves and Escape restores."""

    DEFAULT_CSS = """
    ThemePickerModal { align: center middle; background: rgba(0, 0, 0, 0.65); }
    #theme-card { width: 75%; max-width: 58; height: 85%; max-height: 27;
                  background: $surface; border: round $accent; padding: 1 2; }
    #theme-title { height: 2; text-align: center; text-style: bold; color: $accent; }
    #theme-scroll { height: 1fr; }
    .theme-option { width: 100%; height: 2; min-height: 2; margin-bottom: 0; }
    .theme-option:focus { border: heavy $accent; }
    #theme-help { height: 2; color: $text; text-align: center; }
    """
    BINDINGS = [("escape", "cancel", "Cancel"), ("up", "previous_theme", "Previous"),
                ("down", "next_theme", "Next"), ("enter", "choose_theme", "Apply")]

    def __init__(self, original_theme: str, themes: tuple[str, ...] = THEME_CHOICES):
        super().__init__()
        self.original_theme = original_theme
        self.themes = themes
        self.selected_theme = original_theme

    def compose(self) -> ComposeResult:
        with Vertical(id="theme-card"):
            yield Label("🎨 Choose a Theme", id="theme-title")
            with VerticalScroll(id="theme-scroll"):
                for index, theme in enumerate(self.themes):
                    yield ThemeOption(theme, index)
            yield Static("↑↓ preview  ·  Enter apply  ·  Esc restore", id="theme-help")

    def on_mount(self) -> None:
        index = self.themes.index(self.original_theme) if self.original_theme in self.themes else 0
        self.query_one(f"#theme-{index}", ThemeOption).focus()

    def preview(self, theme: str) -> None:
        if theme in self.themes:
            self.selected_theme = theme
            self.app.theme = theme

    def _focused_index(self) -> int:
        focused = getattr(self.focused, "id", "") or ""
        return int(focused[6:]) if focused.startswith("theme-") else 0

    def action_previous_theme(self) -> None:
        self.query_one(f"#theme-{(self._focused_index() - 1) % len(self.themes)}", ThemeOption).focus()

    def action_next_theme(self) -> None:
        self.query_one(f"#theme-{(self._focused_index() + 1) % len(self.themes)}", ThemeOption).focus()

    def action_choose_theme(self) -> None:
        self.dismiss(self.selected_theme)

    def action_cancel(self) -> None:
        self.app.theme = self.original_theme
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if isinstance(event.button, ThemeOption):
            self.dismiss(event.button.theme_name)


class QuickSelectModal(ModalScreen[str | None]):
    """Searchable, keyboard- and mouse-operated selector for models and sessions."""

    DEFAULT_CSS = """
    QuickSelectModal { align: center middle; background: rgba(0, 0, 0, 0.70); }
    #quick-card { width: 88%; max-width: 96; height: 80%; max-height: 30;
                  background: $surface; border: round $accent; padding: 1 2; }
    #quick-title { height: 2; text-align: center; text-style: bold; color: $accent; }
    #quick-search { margin-bottom: 1; }
    #quick-results { height: 1fr; }
    #quick-detail { height: 3; color: $text; background: $panel; padding: 0 1; }
    #quick-help { height: 1; color: $text-muted; text-align: center; }
    .quick-choice { width: 100%; height: 2; min-height: 2; }
    .quick-choice:focus { border: heavy $accent; }
    """
    BINDINGS = [("escape", "cancel", "Cancel"), ("up", "previous", "Previous"),
                ("down", "next", "Next"), ("enter", "choose", "Select")]

    def __init__(self, title: str, choices: list[tuple[str, str]], *, current: str | None = None):
        super().__init__()
        self.title_text = title
        self.choices = choices
        self.current = current
        self.visible_choices: list[tuple[str, str]] = []
        self.selected_index = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="quick-card"):
            yield Label(self.title_text, id="quick-title")
            yield Input(placeholder="Type to filter…", id="quick-search")
            yield VerticalScroll(id="quick-results")
            yield Static("", id="quick-detail")
            yield Static("Type to filter · ↑↓ navigate · Enter select · Esc cancel", id="quick-help")

    async def on_mount(self) -> None:
        await self._filter("")
        self.query_one("#quick-search", Input).focus()

    async def _filter(self, query: str) -> None:
        words = query.casefold().split()
        self.visible_choices = [choice for choice in self.choices
                                if all(word in (choice[0] + " " + choice[1]).casefold() for word in words)][:60]
        self.selected_index = next((i for i, choice in enumerate(self.visible_choices)
                                    if choice[0] == self.current), 0)
        self._update_detail()
        results = self.query_one("#quick-results", VerticalScroll)
        await results.remove_children()
        if not self.visible_choices:
            await results.mount(Static("No matching choices."))
            return
        await results.mount_all(
            Button(label, id=f"quick-{index}", classes="quick-choice",
                   variant="primary" if index == self.selected_index else "default")
            for index, (_, label) in enumerate(self.visible_choices))

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "quick-search":
            await self._filter(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "quick-search":
            # Input.Submitted bubbles to the app; stop it here or Enter can
            # submit the underlying task prompt after selecting a model.
            event.stop()
            self.action_choose()

    def _select_index(self, index: int) -> None:
        if not self.visible_choices:
            return
        old = self.query(f"#quick-{self.selected_index}")
        if old:
            old.first().variant = "default"
        self.selected_index = max(0, min(index, len(self.visible_choices) - 1))
        selected = self.query_one(f"#quick-{self.selected_index}", Button)
        selected.variant = "primary"
        selected.scroll_visible()
        self._update_detail()

    def _update_detail(self) -> None:
        detail = self.query_one("#quick-detail", Static)
        detail.update(Text(self.visible_choices[self.selected_index][1] if self.visible_choices
                           else "No matching choices."))

    def action_previous(self) -> None:
        self._select_index(self.selected_index - 1)

    def action_next(self) -> None:
        self._select_index(self.selected_index + 1)

    def action_choose(self) -> None:
        if self.visible_choices:
            self.dismiss(self.visible_choices[self.selected_index][0])

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id and event.button.id.startswith("quick-"):
            self.dismiss(self.visible_choices[int(event.button.id[6:])][0])


class OptionDescription(Static):
    """Wrapped option text that also acts as a mouse selection target."""
    ALLOW_SELECT = False

    def __init__(self, value: str, index: int):
        super().__init__(Text(value), id=f"opt-text-{index}", classes="option-text")
        self.index = index

class ClarificationModal(ModalScreen[str]):
    """Pop-up modal dialog prompting the user for clarification in the middle of development."""

    DEFAULT_CSS = """
    ClarificationModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }
    #modal-container {
        width: 90%;
        max-width: 88;
        height: 90%;
        background: $surface;
        border: round $warning;
        padding: 0 2;
    }
    #modal-title {
        height: 2;
        text-align: center;
        text-style: bold;
        color: $warning;
        padding-top: 1;
    }
    #modal-info { height: 5; min-height: 3; width: 100%; overflow-y: auto; border: round $primary; padding: 0 1; }
    #modal-scroll { height: 1fr; min-height: 3; width: 100%; overflow-y: auto; }
    #modal-question { width: 100%; height: auto; max-height: 3; overflow-y: auto; color: $text; text-style: bold; margin-bottom: 1; }
    #modal-context {
        width: 100%;
        height: auto;
        color: $text;
        background: $warning 20%;
        padding: 1;
        margin-bottom: 1;
        max-height: 2;
        overflow-y: auto;
    }
    #options-container { height: auto; }
    .option-row { height: auto; min-height: 3; width: 100%; margin-bottom: 1; background: $panel; }
    .opt-btn { width: 6; min-width: 6; margin-right: 1; }
    .option-text { width: 1fr; min-width: 0; height: auto; padding: 0 1; color: $text; }
    .option-text:hover { background: $primary 20%; }
    .opt-btn:focus { border: heavy $accent; }
    #write-in-input {
        width: 100%;
        height: 3;
        background: $panel;
        color: $text;
    }
    #modal-help { height: 1; color: $accent; }
    #modal-buttons { height: 3; }
    #modal-buttons Button { width: 1fr; }
    """
    BINDINGS = [("escape", "cancel", "Cancel"), ("up", "previous_option", "Previous"),
                ("down", "next_option", "Next"), ("enter", "choose_focused", "Select")]

    def __init__(
        self,
        question: str,
        options: Optional[List[str]] = None,
        context: Optional[str] = None,
    ):
        super().__init__()
        self.question = question
        self.options = (options if options is not None else ["Proceed with recommended plan", "Abort action"])[:9]
        self.context_msg = context or "Classifier detected ambiguity or risk requiring alignment."

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-container"):
            yield Label("⚡ Agent Requires Clarification", id="modal-title")
            with VerticalScroll(id="modal-info"):
                yield Static(Text(self.question), id="modal-question")
                yield Static(Text(self.context_msg), id="modal-context")
            with VerticalScroll(id="modal-scroll"):
                with Vertical(id="options-container"):
                    for i, opt in enumerate(self.options):
                        with Horizontal(classes="option-row"):
                            yield Button(f"{i+1}.", id=f"opt-{i}", classes="opt-btn", variant="primary" if i == 0 else "default")
                            yield OptionDescription(opt, i)

            yield Static("Scroll question/reason · 1-9 choose · ↑↓ options · Tab type · Esc cancel", id="modal-help")
            yield Input(placeholder="Type custom answer and press Enter...", id="write-in-input")

            with Horizontal(id="modal-buttons"):
                yield Button("Submit", id="submit-btn", variant="success")
                yield Button("Cancel", id="cancel-btn", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "cancel-btn":
            self.dismiss("Action cancelled by user")
        elif btn_id == "submit-btn":
            write_in = self.query_one("#write-in-input", Input).value.strip()
            if write_in:
                self.dismiss(write_in)
        elif btn_id and btn_id.startswith("opt-"):
            idx = int(btn_id.split("-")[1])
            self.dismiss(self.options[idx])

    def on_click(self, event: events.Click) -> None:
        if isinstance(event.widget, OptionDescription):
            self.dismiss(self.options[event.widget.index])

    def on_mount(self) -> None:
        # Keep question and reason at the top while the first choice receives focus.
        self.query_one("#modal-info", VerticalScroll).scroll_home(animate=False)
        self.query_one("#modal-scroll", VerticalScroll).scroll_home(animate=False)
        if self.options:
            self.query_one("#opt-0", Button).focus(scroll_visible=False)
        else:
            self.query_one("#write-in-input", Input).focus(scroll_visible=False)

    def on_key(self, event) -> None:
        if not isinstance(self.focused, Input) and event.key in {str(i) for i in range(1, len(self.options) + 1)}:
            self.dismiss(self.options[int(event.key) - 1])
            event.stop()

    def action_cancel(self) -> None:
        self.dismiss("Action cancelled by user")

    def _focused_index(self) -> int:
        focused_id = getattr(self.focused, "id", "") or ""
        return int(focused_id[4:]) if focused_id.startswith("opt-") else 0

    def action_previous_option(self) -> None:
        if not self.options:
            return
        target = self.query_one(f"#opt-{(self._focused_index()-1) % len(self.options)}", Button)
        target.focus()
        target.scroll_visible()

    def action_next_option(self) -> None:
        if not self.options:
            return
        target = self.query_one(f"#opt-{(self._focused_index()+1) % len(self.options)}", Button)
        target.focus()
        target.scroll_visible()

    def action_choose_focused(self) -> None:
        if isinstance(self.focused, Input):
            value = self.focused.value.strip()
            if value:
                self.dismiss(value)
        elif isinstance(self.focused, Button):
            self.focused.press()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        val = event.value.strip()
        if val:
            self.dismiss(val)


class OutputViewerModal(ModalScreen[None]):
    """Scrollable, selectable view of the complete latest agent response."""

    DEFAULT_CSS = """
    OutputViewerModal { align: center middle; background: rgba(0, 0, 0, 0.70); }
    #output-card { width: 92%; height: 88%; background: $surface; border: round $accent; padding: 1 2; }
    #output-title { height: 2; text-align: center; text-style: bold; color: $accent; }
    #output-text { height: 1fr; border: solid $primary; }
    #output-help { height: 1; color: $text-muted; text-align: center; }
    #output-actions { height: 3; }
    #output-actions Button { width: 1fr; }
    """
    BINDINGS = [("escape", "close", "Close"), ("ctrl+shift+c", "copy_all", "Copy all")]

    def __init__(self, content: str, title: str = "Latest Agent Output"):
        super().__init__()
        self.content = content
        self.title_text = title

    def compose(self) -> ComposeResult:
        with Vertical(id="output-card"):
            yield Label(f"{self.title_text} · Select text or copy all", id="output-title")
            yield TextArea(self.content, read_only=True, soft_wrap=True, show_line_numbers=False, id="output-text")
            yield Static("Mouse drag / keyboard select · Ctrl+Shift+C copy selection or all · Esc close", id="output-help")
            with Horizontal(id="output-actions"):
                yield Button("Copy all", id="output-copy", variant="success")
                yield Button("Close", id="output-close")

    def action_close(self) -> None:
        self.dismiss(None)

    def action_copy_all(self) -> None:
        selection = self.get_selected_text()
        if not selection:
            selection = self.query_one("#output-text", TextArea).selected_text
        copied = selection if selection else self.content
        self.app.copy_to_clipboard(copied)
        label = "Selection" if selection else "Full response"
        self.query_one("#output-help", Static).update(f"✓ {label} copied to clipboard · Esc close")

    def on_mount(self) -> None:
        self.query_one("#output-text", TextArea).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "output-copy":
            self.action_copy_all()
        elif event.button.id == "output-close":
            self.action_close()
