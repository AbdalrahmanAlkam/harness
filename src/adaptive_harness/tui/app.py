"""Textual Terminal User Interface (TUI) application for Adaptive Agent Harness."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional
from rich.markup import escape
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static

from adaptive_harness.agent.agent import AgentEvent, DeveloperAgent
from adaptive_harness.data.storage import ExperienceRepository
from adaptive_harness.llm.client import LLMClient, MODEL_TIERS
from adaptive_harness.tui.widgets import ClarificationModal, ClassifierTelemetryWidget


class AdaptiveHarnessApp(App):
    """Full-featured Terminal User Interface for Adaptive Agent Harness."""

    CSS = """
    Screen {
        background: $background;
    }
    #main-container {
        height: 1fr;
    }
    #chat-log {
        width: 1fr;
        height: 100%;
        background: $background;
        border: solid $primary;
        padding: 1;
    }
    #input-container {
        height: 3;
        dock: bottom;
        background: $surface;
        padding: 0 1;
    }
    #prompt-input {
        width: 100%;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_screen", "Clear Log"),
        ("f1", "show_help", "Help"),
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        db_path: Path | str = "output/experience.db",
        workspace_root: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model or MODEL_TIERS["standard"]
        self.db_path = db_path
        self.workspace_root = workspace_root

        # LLM Client & Repo
        self.llm_client = LLMClient(
            api_key=self.api_key,
            base_url=self.base_url,
            default_model=self.default_model,
        )
        self.repository = ExperienceRepository(self.db_path)

        # Developer Agent with interactive clarification hook
        self.agent = DeveloperAgent(
            llm_client=self.llm_client,
            repository=self.repository,
            clarification_callback=self._request_interactive_clarification,
            workspace_root=self.workspace_root,
        )
        self._current_clarification_future: Optional[asyncio.Future[str]] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-container"):
            yield RichLog(id="chat-log", wrap=True, highlight=True, markup=True)
            yield ClassifierTelemetryWidget(id="telemetry")

        with Container(id="input-container"):
            yield Input(
                placeholder="Ask agent to code, inspect, test, or run commands (/help for commands)...",
                id="prompt-input",
            )
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Adaptive Agent Harness - Developer TUI"
        self.sub_title = f"Model: {self.agent.llm_client.default_model} | Mode: {'LIVE (OpenRouter)' if not self.agent.llm_client.is_mock else 'OFFLINE MOCK'}"

        log = self.query_one("#chat-log", RichLog)
        log.write("[bold cyan]Welcome to Adaptive Agent Harness 2.0![/bold cyan]")
        log.write(
            "[dim]Autonomous coding agent with pervasive ML routing, active verification, and interactive clarification.[/dim]\n"
        )
        if self.agent.llm_client.is_mock:
            log.write("[yellow]Notice: No OPENROUTER_API_KEY detected. Running in offline simulated mode.[/yellow]")
            log.write("[dim]Type '/key <YOUR_KEY>' to connect your OpenRouter account at any time.[/dim]\n")
        else:
            log.write(f"[green]Connected to OpenRouter using {self.agent.llm_client.default_model}[/green]\n")

        self.query_one("#prompt-input", Input).focus()

    def _request_interactive_clarification(
        self,
        question: str,
        options: Optional[list[str]],
        context: Optional[str],
    ) -> str:
        """Called by agent when ambiguity or risk is detected in the middle of development."""
        import concurrent.futures

        fut: concurrent.futures.Future[str] = concurrent.futures.Future()

        def show_modal():
            def on_dismiss(ans: Optional[str]):
                fut.set_result(ans or (options[0] if options else "Proceed"))

            modal = ClarificationModal(question=question, options=options, context=context)
            self.push_screen(modal, callback=on_dismiss)

        self.call_from_thread(show_modal)
        # Block worker thread safely until modal is dismissed and future resolved
        return fut.result()

    def action_clear_screen(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.clear()

    def action_show_help(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write("[bold yellow]Available Commands:[/bold yellow]")
        log.write("  /key <API_KEY>         - Set OpenRouter API key")
        log.write("  /model <MODEL_ID>      - Switch active LLM model")
        log.write("  /tier <fast|std|deep>  - Switch model tier")
        log.write("  /clear                 - Clear chat history")
        log.write("  /exit                  - Exit application")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        input_widget = self.query_one("#prompt-input", Input)
        input_widget.value = ""

        # Handle slash commands
        if text.startswith("/"):
            self._handle_slash_command(text)
            return

        log = self.query_one("#chat-log", RichLog)
        log.write(f"\n[bold white]Developer >[/bold white] [bold]{text}[/bold]")

        # Launch agent execution in non-blocking worker thread
        self.execute_agent_task(text)

    def _handle_slash_command(self, cmd_text: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        parts = cmd_text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit"):
            self.exit()
        elif cmd == "/clear":
            self.action_clear_screen()
        elif cmd == "/help":
            self.action_show_help()
        elif cmd == "/key":
            if not arg:
                log.write("[red]Usage: /key <OPENROUTER_API_KEY>[/red]")
                return
            self.agent.llm_client = LLMClient(api_key=arg, default_model=self.agent.llm_client.default_model)
            log.write("[green]✓ OpenRouter API key updated![/green]")
            self.sub_title = f"Model: {self.agent.llm_client.default_model} | Mode: LIVE (OpenRouter)"
        elif cmd == "/model":
            if not arg:
                log.write(f"[yellow]Current model: {self.agent.llm_client.default_model}[/yellow]")
                return
            self.agent.llm_client.default_model = arg
            log.write(f"[green]✓ Switched model to {arg}[/green]")
            self.sub_title = f"Model: {arg} | Mode: {'LIVE' if not self.agent.llm_client.is_mock else 'MOCK'}"
        elif cmd == "/tier":
            if arg in MODEL_TIERS:
                target_model = MODEL_TIERS[arg]
                self.agent.llm_client.default_model = target_model
                log.write(f"[green]✓ Switched to {arg.upper()} tier ({target_model})[/green]")
            else:
                log.write(f"[red]Invalid tier. Choose from: {', '.join(MODEL_TIERS.keys())}[/red]")
        else:
            log.write(f"[red]Unknown command: {cmd}. Type /help for options.[/red]")

    @work(thread=True)
    def execute_agent_task(self, task_text: str) -> None:
        """Worker thread executing the agent task and streaming events back to the UI."""
        telemetry = self.query_one("#telemetry", ClassifierTelemetryWidget)
        log = self.query_one("#chat-log", RichLog)

        for event in self.agent.run_stream(task_text):
            et = event.event_type
            p = event.payload

            if et == "skill_classification":
                telemetry.update_telemetry(
                    probabilities=p["probabilities"],
                    primary_skill=p["primary_skill"],
                )
            elif et == "ambiguity_assessment":
                telemetry.update_telemetry(
                    entropy=p["entropy"],
                    margin=p["confidence_margin"],
                    risk_level=p["risk_level"],
                )
            elif et == "model_routing":
                telemetry.update_telemetry(
                    tier=p["tier"],
                    model=p["model"],
                )
            elif et == "thought":
                log.write(f"\n[bold magenta]Agent ({p.get('model', 'thought')}):[/bold magenta] {escape(p['content'])}")
            elif et == "tool_call":
                log.write(
                    f"  [bold yellow]⚡ Tool Call:[/bold yellow] [cyan]{p['name']}[/cyan] [dim]args={escape(str(p['arguments']))}[/dim]"
                )
            elif et == "tool_result":
                status_color = "green" if p["success"] else "red"
                log.write(
                    f"  [{status_color}]Result ({p['time_ms']} ms):[/{status_color}] {escape(p['output'][:300])}"
                )
            elif et == "verification":
                badge_color = "green" if p["status"] == "SUCCESS" else "red bold"
                log.write(f"  [dim]Verification Classifier: [{badge_color}]{p['status']}[/{badge_color}] -> Action: {p['action']}[/dim]")
            elif et == "response":
                log.write(f"\n[bold green]✓ Task Completed in {p['total_time_ms']} ms ({p['steps']} steps)[/bold green]\n")
