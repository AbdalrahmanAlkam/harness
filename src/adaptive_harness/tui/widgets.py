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
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static


class ClassifierTelemetryWidget(Static):
    """Real-time classifier telemetry panel displaying live probabilities, entropy, and tier."""

    DEFAULT_CSS = """
    ClassifierTelemetryWidget {
        width: 38;
        height: 100%;
        background: $surface;
        border-left: solid $primary;
        padding: 1;
        overflow-y: auto;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.probabilities: Dict[str, float] = {}
        self.entropy: float = 0.0
        self.margin: float = 0.0
        self.risk_level: str = "low"
        self.tier: str = "standard"
        self.active_model: str = MODEL_TIERS["standard"]
        self.primary_skill: str = "none"
        self.selection: str = "auto"
        self.classifier_engine: str = "sklearn"
        self.classifier_model: str = "TF-IDF + Logistic Regression"
        self.classifier_latency_ms: float = 0.0
        self.domain_mode: str = "coding"
        self.thinking_level: str = "low"
        self.thinking_tokens: int = 1000

    def update_telemetry(
        self,
        probabilities: Optional[Dict[str, float]] = None,
        entropy: Optional[float] = None,
        margin: Optional[float] = None,
        risk_level: Optional[str] = None,
        tier: Optional[str] = None,
        model: Optional[str] = None,
        primary_skill: Optional[str] = None,
        selection: Optional[str] = None,
        classifier_engine: Optional[str] = None,
        classifier_model: Optional[str] = None,
        classifier_latency_ms: Optional[float] = None,
        domain_mode: Optional[str] = None,
        thinking_level: Optional[str] = None,
        thinking_tokens: Optional[int] = None,
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
        for key, value in (("selection", selection), ("classifier_engine", classifier_engine),
                           ("classifier_model", classifier_model), ("classifier_latency_ms", classifier_latency_ms),
                           ("domain_mode", domain_mode), ("thinking_level", thinking_level),
                           ("thinking_tokens", thinking_tokens)):
            if value is not None:
                setattr(self, key, value)

        self.refresh()

    def render(self) -> Panel:
        status = Text(overflow="fold")
        status.append("MODEL  ", style="bold cyan")
        status.append(f"{self.selection.upper()} · {self.tier.upper()}\n", style="bold magenta")
        status.append(self.active_model + "\n", style="white")
        status.append("CLASSIFIER  ", style="bold cyan")
        status.append(self.classifier_engine.upper() + "\n", style="bold white")
        status.append(self.classifier_model + "\n", style="white")
        status.append(f"Latency  {self.classifier_latency_ms:.2f} ms\n", style="cyan")
        status.append(f"Domain  {self.domain_mode.upper()}\n", style="bold magenta")
        status.append(f"Thinking  {self.thinking_level.upper()} · {self.thinking_tokens:,} tokens\n", style="bold yellow")
        status.append(f"Skill  {self.primary_skill}\n", style="white")
        status.append(f"H(p)  {self.entropy:.3f} bits   Margin  {self.margin*100:.1f}%\n", style="cyan")
        risk_style = "green" if self.risk_level == "low" else "yellow" if self.risk_level == "medium" else "bold red"
        status.append(f"Risk  {self.risk_level.upper()}", style=risk_style)

        bars_table = Table(box=None, show_header=False, expand=True, padding=(0, 0))
        bars_table.add_column("Skill", width=10)
        bars_table.add_column("Bar", width=8)
        bars_table.add_column("Pct", justify="right", width=6)

        sorted_skills = sorted(self.probabilities.items(), key=lambda x: x[1], reverse=True)
        for skill, prob in sorted_skills:
            bar_len = max(0, min(8, round(prob * 8)))
            bar_str = "█" * bar_len + "░" * (8 - bar_len)
            c = "green" if prob >= 0.4 else ("yellow" if prob >= 0.15 else "bright_black")
            clean_name = skill.replace("_", " ")[:10]
            bars_table.add_row(Text(clean_name), Text(bar_str, style=c), Text(f"{prob*100:4.1f}%"))

        return Panel(Group(status, Rule(style="cyan"), Text("SKILL PROBABILITIES", style="bold cyan"), bars_table),
                     title="[bold cyan]ROUTING[/bold cyan]", border_style="cyan")


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
        border: round #fbbf24;
        padding: 0 2;
    }
    #modal-title {
        height: 2;
        text-align: center;
        text-style: bold;
        color: #fbbf24;
        padding-top: 1;
    }
    #modal-scroll { height: 1fr; width: 100%; overflow-y: auto; }
    #modal-question { width: 100%; height: auto; color: $text; text-style: bold; margin-bottom: 1; }
    #modal-context {
        width: 100%;
        height: auto;
        color: #ffffff;
        background: #493b20;
        padding: 1;
        margin-bottom: 1;
    }
    #options-container {
        height: auto;
        margin-bottom: 1;
    }
    .option-row { height: auto; min-height: 3; width: 100%; margin-bottom: 1; background: $panel; }
    .opt-btn { width: 12; min-width: 12; margin-right: 1; }
    .option-text { width: 1fr; min-width: 0; height: auto; padding: 0 1; color: $text; }
    .option-text:hover { background: $primary 20%; }
    .opt-btn:focus { border: heavy #22d3ee; }
    #write-in-input {
        width: 100%;
        height: 3;
        background: $panel;
        color: $text;
    }
    #modal-help { height: 1; color: #22d3ee; }
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
            with VerticalScroll(id="modal-scroll"):
                yield Label(Text(self.question), id="modal-question")
                yield Label(Text(self.context_msg), id="modal-context")

                with Vertical(id="options-container"):
                    for i, opt in enumerate(self.options):
                        with Horizontal(classes="option-row"):
                            yield Button(f"{i+1}. Choose", id=f"opt-{i}", classes="opt-btn", variant="primary" if i == 0 else "default")
                            yield OptionDescription(opt, i)

            yield Static("↓ options  1-9 choose  Tab type  Esc cancel", id="modal-help")
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
        val = event.value.strip()
        if val:
            self.dismiss(val)
