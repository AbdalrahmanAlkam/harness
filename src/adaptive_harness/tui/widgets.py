"""Custom Textual widgets for classifier telemetry, clarification modal, and stream views."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
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
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.probabilities: Dict[str, float] = {}
        self.entropy: float = 0.0
        self.margin: float = 0.0
        self.risk_level: str = "low"
        self.tier: str = "standard"
        self.active_model: str = "openai/gpt-4o"
        self.primary_skill: str = "none"

    def update_telemetry(
        self,
        probabilities: Optional[Dict[str, float]] = None,
        entropy: Optional[float] = None,
        margin: Optional[float] = None,
        risk_level: Optional[str] = None,
        tier: Optional[str] = None,
        model: Optional[str] = None,
        primary_skill: Optional[str] = None,
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

        self.refresh()

    def render(self) -> Panel:
        table = Table(box=None, show_header=False, expand=True, padding=(0, 0))
        table.add_column("Key", style="bold cyan", width=12)
        table.add_column("Value")

        # Model tier badge
        tier_color = "green" if self.tier == "fast" else ("yellow" if self.tier == "standard" else "magenta")
        table.add_row("Model Tier", f"[{tier_color} bold][{self.tier.upper()}][/{tier_color} bold]")
        table.add_row("Active Model", f"[dim]{self.active_model.split('/')[-1]}[/dim]")
        table.add_row("Primary Skill", f"[bold white]{self.primary_skill}[/bold white]")

        # Entropy & Margin
        ent_color = "green" if self.entropy < 1.0 else ("yellow" if self.entropy < 1.5 else "red")
        table.add_row("Entropy H(p)", f"[{ent_color}]{self.entropy:.3f} bits[/{ent_color}]")
        table.add_row("Margin", f"{self.margin*100:4.1f}%")

        # Risk level
        risk_color = "green" if self.risk_level == "low" else ("yellow" if self.risk_level == "medium" else "red bold")
        table.add_row("Risk Level", f"[{risk_color}]{self.risk_level.upper()}[/{risk_color}]")

        # Skill probability bars
        bars_table = Table(box=None, show_header=False, expand=True, padding=(0, 0))
        bars_table.add_column("Skill", width=11)
        bars_table.add_column("Bar", width=12)
        bars_table.add_column("Pct", justify="right", width=6)

        sorted_skills = sorted(self.probabilities.items(), key=lambda x: x[1], reverse=True)
        for skill, prob in sorted_skills:
            bar_len = int(prob * 10)
            bar_str = "█" * bar_len + "░" * (10 - bar_len)
            c = "green" if prob >= 0.4 else ("yellow" if prob >= 0.15 else "dim")
            clean_name = skill.replace("_", " ")[:10]
            bars_table.add_row(clean_name, f"[{c}]{bar_str}[/{c}]", f"{prob*100:4.1f}%")

        content = Vertical()
        out_renderable = Text()
        out_renderable.append("CLASSIFIER TELEMETRY\n", style="bold magenta underline")

        full_table = Table(box=None, show_header=False, expand=True)
        full_table.add_column("Section")
        full_table.add_row(table)
        full_table.add_row(Text("─" * 32, style="dim"))
        full_table.add_row(Text("Intent Probabilities:", style="bold cyan"))
        full_table.add_row(bars_table)

        return Panel(
            full_table,
            title="[bold cyan]Telemetry Brain[/bold cyan]",
            border_style="cyan",
        )


class ClarificationModal(ModalScreen[str]):
    """Pop-up modal dialog prompting the user for clarification in the middle of development."""

    DEFAULT_CSS = """
    ClarificationModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }
    #modal-container {
        width: 70;
        height: auto;
        background: $surface;
        border: thick $warning;
        padding: 1 2;
    }
    #modal-title {
        text-align: center;
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }
    #modal-context {
        color: $text-muted;
        margin-bottom: 1;
    }
    #options-container {
        height: auto;
        margin-bottom: 1;
    }
    .opt-btn {
        margin: 1 0;
        width: 100%;
    }
    #write-in-input {
        margin-top: 1;
        margin-bottom: 1;
    }
    """

    def __init__(
        self,
        question: str,
        options: Optional[List[str]] = None,
        context: Optional[str] = None,
    ):
        super().__init__()
        self.question = question
        self.options = options or ["Proceed with recommended plan", "Abort action"]
        self.context_msg = context or "Classifier detected ambiguity or risk requiring alignment."

    def compose(self) -> ComposeResult:
        with Container(id="modal-container"):
            yield Label("❓ Clarification Needed (Agent Paused)", id="modal-title")
            yield Label(f"[bold white]{self.question}[/bold white]")
            yield Label(f"[dim]{self.context_msg}[/dim]", id="modal-context")

            with Vertical(id="options-container"):
                for i, opt in enumerate(self.options):
                    yield Button(f"{i+1}. {opt}", id=f"opt-{i}", classes="opt-btn", variant="primary" if i == 0 else "default")

            yield Label("Or type a custom instruction:")
            yield Input(placeholder="Type custom answer and press Enter...", id="write-in-input")

            with Horizontal():
                yield Button("Submit", id="submit-btn", variant="success")
                yield Button("Cancel", id="cancel-btn", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "cancel-btn":
            self.dismiss("Action cancelled by user")
        elif btn_id == "submit-btn":
            write_in = self.query_one("#write-in-input", Input).value.strip()
            self.dismiss(write_in or self.options[0])
        elif btn_id and btn_id.startswith("opt-"):
            idx = int(btn_id.split("-")[1])
            self.dismiss(self.options[idx])

    def on_input_submitted(self, event: Input.Submitted) -> None:
        val = event.value.strip()
        self.dismiss(val or self.options[0])
