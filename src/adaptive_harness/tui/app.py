"""Textual Terminal User Interface (TUI) application for Adaptive Agent Harness."""

from __future__ import annotations

from pathlib import Path
import concurrent.futures
from datetime import datetime
import json
import os
from typing import Optional
from rich.syntax import Syntax
from rich.text import Text
from rich.markdown import Markdown
from textual import work
from textual.app import App, ComposeResult
from textual import events
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static

from adaptive_harness.agent.agent import AgentEvent, DeveloperAgent
from adaptive_harness.agent.skills import SkillCatalog
from adaptive_harness.classifiers.engine import create_backend
from adaptive_harness.classifiers.domain_classifier import parse_domain_mode
from adaptive_harness.classifiers.thinking_classifier import parse_thinking_level, BUDGET_TOKENS
from adaptive_harness.data.storage import ExperienceRepository
from adaptive_harness.data.config import ConfigManager, PromptHistoryStore
from adaptive_harness.data.sessions import SessionStore
from adaptive_harness.llm.client import LLMClient, MODEL_TIERS
from adaptive_harness.llm.catalog import CatalogModel, fetch_models
from adaptive_harness.tui.widgets import (ClarificationModal, ClassifierTelemetryWidget,
    HistoryInput, PinnedRichLog, ThemePickerModal, QuickSelectModal, THEME_CHOICES)


COMMANDS = ("/key", "/model", "/models", "/tier", "/mode", "/thinking", "/safety", "/theme", "/classifier", "/new",
            "/clear", "/history", "/help", "/exit", "/reset", "/workspace", "/sessions",
            "/session", "/skills", "/skill", "/output", "/export")


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
        background: $surface;
        padding: 0 1;
        margin: 1 0 1 0;
    }
    #status-line { height: 1; background: $surface; color: $text; padding: 0 1; }
    #prompt-input {
        width: 100%;
    }
    #command-hints { height: 1; color: $accent; display: none; }
    #command-hints.visible { display: block; }
    #scroll-indicator { height: 1; color: $warning; background: $surface; display: none; padding: 0 1; }
    #scroll-indicator.visible { display: block; }
    #waiting-indicator {
        height: 1;
        color: $warning;
        background: $surface;
        display: none;
        padding: 0 1;
    }
    #waiting-indicator.visible { display: block; }
    #waiting-indicator.pulse { color: $accent; }
    Screen.compact #main-container, Screen.compact #status-line,
    Screen.compact Header, Screen.compact #scroll-indicator,
    Screen.compact #command-hints { display: none; }
    Screen.compact #input-container { margin: 0 0 1 0; }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_screen", "Clear Log"),
        ("ctrl+n", "new_session", "New Session"),
        ("f1", "show_help", "Help"),
        ("f2", "choose_theme", "Theme"),
        ("f3", "toggle_telemetry", "Telemetry"),
        ("f4", "choose_model", "Models"),
        ("f5", "choose_session", "Sessions"),
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        db_path: Path | str = "output/experience.db",
        workspace_root: Optional[str] = None,
        classifier_backend: str = "auto",
        classifier_model: Optional[str] = None,
        classifier_endpoint: Optional[str] = None,
        semif_device: str = "auto",
        semif_4bit: bool = False,
        semif_temperature: float = 1.0,
        mode: str = "auto",
        thinking: str = "auto",
        safety: str | None = None,
        session_id: Optional[str] = None,
        config_dir: Path | str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = ConfigManager(config_dir)
        saved_config = self.config.load()
        self.history_store = PromptHistoryStore(config_dir)
        env_key = os.environ.get("OPENROUTER_API_KEY")
        self.api_key = api_key or env_key or saved_config.get("api_key")
        self.key_source = ("CLI flag" if api_key else "environment" if env_key else
                           "saved configuration" if saved_config.get("api_key") else "offline")
        self.saved_theme = saved_config.get("theme", "textual-dark")
        self.base_url = base_url
        if default_model and default_model.lower() == "auto":
            default_model = None
        self._cli_model_override = default_model
        self.default_model = default_model or MODEL_TIERS["standard"]
        self.db_path = db_path
        self.workspace_root = str(Path(workspace_root or ".").expanduser().resolve())
        if not Path(self.workspace_root).is_dir():
            raise ValueError(f"Workspace directory does not exist: {self.workspace_root}")
        self.classifier_endpoint = classifier_endpoint
        self.semif_device = semif_device
        self.semif_4bit = semif_4bit
        self.semif_temperature = semif_temperature
        initial_mode = parse_domain_mode(mode)
        initial_thinking = parse_thinking_level(thinking)
        self._cli_mode_override = initial_mode
        self._cli_thinking_override = initial_thinking
        if safety is not None and safety not in {"turbo", "cautious"}:
            raise ValueError("Safety profile must be turbo or cautious")
        self._cli_safety_override = safety
        self._default_safety = "turbo"
        self._model_catalog: list[CatalogModel] = []
        self._activity = "Ready"
        self._activity_pulse = False

        # LLM Client & Repo
        self.llm_client = LLMClient(
            api_key=self.api_key,
            base_url=self.base_url,
            default_model=self.default_model,
            force_mock=not bool(self.api_key) and not self._local_endpoint(),
        )
        self.repository = ExperienceRepository(self.db_path)
        self.session_store = SessionStore(self.db_path)
        self.session = self.session_store.load(session_id) if session_id else None
        if session_id and self.session is None:
            self.session_store.close()
            raise ValueError(f"Session not found: {session_id}")
        if self.session is not None:
            self.workspace_root = self.session.workspace
        else:
            self.session = self.session_store.create(self.workspace_root)
        self.skill_catalog = SkillCatalog(self.workspace_root)
        self._busy = False
        self._show_telemetry = True
        self._last_tool_output = ""
        self._last_agent_content = ""
        self._clarification_future: concurrent.futures.Future[str] | None = None
        self._session_restore_warning = ""
        self._quit_when_finished = False
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._tokens_saved_estimate = 0
        self._cached_tokens = 0
        self._has_focus = True
        self._bell_rung = False

        # Developer Agent with interactive clarification hook
        self.agent = DeveloperAgent(
            llm_client=self.llm_client,
            repository=self.repository,
            clarification_callback=self._request_interactive_clarification,
            workspace_root=self.workspace_root,
            explicit_model=default_model,
            preferences_dir=config_dir,
            forced_mode=initial_mode,
            forced_thinking=initial_thinking,
            safety_profile=safety or self._default_safety,
            classifier_backend=create_backend(classifier_backend, classifier_model, classifier_endpoint,
                api_key=self.api_key, device=semif_device, load_in_4bit=semif_4bit,
                temperature=semif_temperature),
        )
        if session_id:
            self._restore_session(preserve_cli_overrides=True)

    def _local_endpoint(self) -> bool:
        from urllib.parse import urlparse
        return urlparse(self.base_url or os.environ.get("OPENROUTER_BASE_URL", "")).hostname in {
            "localhost", "127.0.0.1", "::1"}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-container"):
            yield PinnedRichLog(id="chat-log", min_width=1, wrap=True, highlight=True, markup=True)
            yield ClassifierTelemetryWidget(id="telemetry")

        yield Static(id="status-line")
        yield Static("[Pinned: Scroll to bottom ↓]", id="scroll-indicator")

        with Container(id="input-container"):
            yield HistoryInput(
                placeholder="Ask agent to code, inspect, test, or run commands (/help for commands)...",
                id="prompt-input", history=self.history_store.entries,
            )
            yield Static(id="command-hints")
        yield Static("⚡ Agent waiting on your clarification", id="waiting-indicator")
        yield Footer()

    def on_mount(self) -> None:
        if self.saved_theme in self.available_themes:
            self.theme = self.saved_theme
        self.title = "Adaptive Agent Harness - Developer TUI"
        self.sub_title = f"Model: {self.agent.llm_client.default_model} | {'LIVE' if not self.agent.llm_client.is_mock else 'OFFLINE MOCK'}"

        log = self.query_one("#chat-log", RichLog)
        self.query_one("#telemetry", ClassifierTelemetryWidget).update_telemetry(
            classifier_engine=self.agent.classifier_backend.name,
            classifier_model=self.agent.classifier_backend.model,
            domain_mode=self.agent.forced_mode.value if self.agent.forced_mode else "—",
            domain_selection="forced" if self.agent.forced_mode else "auto",
            thinking_level=self.agent.forced_thinking.value if self.agent.forced_thinking else "—",
            thinking_tokens=BUDGET_TOKENS[self.agent.forced_thinking] if self.agent.forced_thinking else 0,
            thinking_selection="forced" if self.agent.forced_thinking else "auto",
            tokens_saved_estimate=self._tokens_saved_estimate,
            provider_cached_tokens=self._cached_tokens,
            provider_prompt_tokens=self.prompt_tokens)
        log.write("[bold cyan]Welcome to Adaptive Agent Harness 2.0![/bold cyan]")
        log.write(
            "[dim]Autonomous coding agent with pervasive ML routing, active verification, and interactive clarification.[/dim]\n"
        )
        if self.agent.llm_client.is_mock:
            log.write("[yellow]Notice: No OPENROUTER_API_KEY detected. Running in offline simulated mode.[/yellow]")
            log.write("[dim]Type '/key <YOUR_KEY>' to connect your OpenRouter account at any time.[/dim]\n")
        else:
            log.write(Text(f"Connected to {self.agent.llm_client.base_url} using {self.agent.llm_client.default_model}\n", style="green"))
        if self._session_restore_warning:
            log.write(Text(self._session_restore_warning, style="yellow"))
        if self.config.last_error:
            log.write(Text(self.config.last_error, style="yellow"))
        if self.session.messages:
            self._render_session_messages(log)

        self.query_one("#prompt-input", Input).focus()
        self._refresh_status()
        self._apply_layout()
        self.set_interval(0.7, self._pulse_waiting)

    def _pulse_waiting(self) -> None:
        indicator = self.query_one("#waiting-indicator", Static)
        if indicator.has_class("visible"):
            indicator.toggle_class("pulse")
        else:
            indicator.remove_class("pulse")
        if self._busy:
            self._activity_pulse = not self._activity_pulse
            self._refresh_status()

    def on_resize(self, event) -> None:
        self._apply_layout()

    def _apply_layout(self) -> None:
        self.query_one("#main-container").screen.set_class(self.size.height < 12, "compact")
        matches = self.query("#telemetry")
        if matches:
            matches.first().display = self._show_telemetry and self.size.width >= 110

    def _refresh_status(self) -> None:
        model = self.agent.explicit_model or "auto"
        provider = ("🟡 Offline Mock Engine" if self.agent.llm_client.is_mock else
                    "🟢 OpenRouter" if "openrouter.ai" in self.agent.llm_client.base_url else "🟢 Local/Custom")
        telemetry = self.query_one("#telemetry", ClassifierTelemetryWidget)
        mode = ("SECURITY" if telemetry.domain_mode == "audit" else
                telemetry.domain_mode.upper() if telemetry.domain_mode != "—" else "AUTO")
        thinking = telemetry.thinking_level.upper() if telemetry.thinking_level != "—" else "AUTO"
        activity = ("● " if self._activity_pulse and self._busy else "◦ ") + self._activity
        self.sub_title = f"{provider} ({model})  ·  {activity}  ·  {mode} / {thinking}  ·  P:{self.prompt_tokens:,} C:{self.completion_tokens:,}"
        self.query_one("#status-line", Static).update(Text(
            f"{activity}  ·  {provider}  ·  {mode} / {thinking}  ·  {self.agent.safety_profile.upper()}  ·  P:{self.prompt_tokens:,} C:{self.completion_tokens:,}  ·  Session {self.session.id}  ·  {self.workspace_root}", style="bold cyan"))

    def _save_session(self) -> None:
        self.session.workspace = self.workspace_root
        self.session.messages = self.agent.messages
        self.session.model = self.agent.explicit_model
        self.session.skills = list(self.agent.active_skills)
        self.session.settings = {"classifier_backend": self.agent.classifier_backend.name,
                                 "classifier_model": self.agent.classifier_backend.model,
                                 "classifier_endpoint": self.classifier_endpoint or "",
                                 "semif_device": self.semif_device, "semif_4bit": str(self.semif_4bit),
                                 "semif_temperature": str(self.semif_temperature),
                                 "mode": self.agent.forced_mode.value if self.agent.forced_mode else "auto",
                                 "thinking": self.agent.forced_thinking.value if self.agent.forced_thinking else "auto",
                                 "safety": self.agent.safety_profile,
                                 "prompt_tokens": str(self.prompt_tokens),
                                 "completion_tokens": str(self.completion_tokens),
                                 "estimated_tokens_saved": str(self._tokens_saved_estimate),
                                 "cached_tokens": str(self._cached_tokens)}
        self.session_store.save(self.session)

    def _restore_session(self, *, preserve_cli_overrides: bool = False) -> None:
        self._session_restore_warning = ""
        self.agent.messages = self.session.messages or [{"role": "system", "content": self.agent.system_prompt}]
        self.agent.explicit_model = (self._cli_model_override if preserve_cli_overrides and self._cli_model_override
                                     else self.session.model)
        if self.agent.explicit_model:
            self.agent.llm_client.default_model = self.agent.explicit_model
        else:
            self.agent.llm_client.default_model = MODEL_TIERS["standard"]
        self.agent.set_workspace(self.session.workspace)
        settings = self.session.settings or {}
        try:
            self.agent.forced_mode = (self._cli_mode_override if preserve_cli_overrides and self._cli_mode_override
                                      else parse_domain_mode(settings.get("mode", "auto")))
            self.agent.forced_thinking = (self._cli_thinking_override if preserve_cli_overrides and self._cli_thinking_override
                                          else parse_thinking_level(settings.get("thinking", "auto")))
        except ValueError as exc:
            self.agent.forced_mode = None
            self.agent.forced_thinking = None
            self._session_restore_warning = f"Saved mode or thinking level is invalid ({exc}); using auto."
        self.agent.safety_profile = (self._cli_safety_override if preserve_cli_overrides and self._cli_safety_override
                                     else settings.get("safety", self._default_safety))
        if self.agent.safety_profile not in {"turbo", "cautious"}:
            self.agent.safety_profile = "turbo"
        for field, setting in (("prompt_tokens", "prompt_tokens"),
                               ("completion_tokens", "completion_tokens"),
                               ("_tokens_saved_estimate", "estimated_tokens_saved"),
                               ("_cached_tokens", "cached_tokens")):
            try:
                setattr(self, field, max(0, int(settings.get(setting, "0"))))
            except (ValueError, TypeError):
                setattr(self, field, 0)
        self.classifier_endpoint = settings.get("classifier_endpoint") or None
        self.semif_device = settings.get("semif_device", self.semif_device)
        self.semif_4bit = settings.get("semif_4bit", str(self.semif_4bit)).lower() == "true"
        try:
            self.semif_temperature = float(settings.get("semif_temperature", self.semif_temperature))
        except (TypeError, ValueError):
            self.semif_temperature = 1.0
        if settings.get("classifier_backend"):
            try:
                self.agent.classifier_backend = create_backend(settings["classifier_backend"],
                    settings.get("classifier_model"), self.classifier_endpoint, api_key=self.api_key,
                    device=self.semif_device, load_in_4bit=self.semif_4bit,
                    temperature=self.semif_temperature)
            except (ValueError, RuntimeError, OSError) as exc:
                self.agent.classifier_backend = create_backend("sklearn")
                self._session_restore_warning = f"Saved classifier could not load ({exc}); using sklearn."
        self.skill_catalog = SkillCatalog(self.session.workspace)
        self.agent.active_skills.clear()
        for name in self.session.skills or []:
            try:
                self.agent.active_skills[name] = self.skill_catalog.read(name)
            except (OSError, ValueError):
                pass

    def on_unmount(self) -> None:
        if self._clarification_future and not self._clarification_future.done():
            self._clarification_future.set_result("Action cancelled by user")
        if not self._busy:
            self._save_session()
        self.session_store.close()

    def _request_interactive_clarification(
        self,
        question: str,
        options: Optional[list[str]],
        context: Optional[str],
    ) -> str:
        """Called by agent when ambiguity or risk is detected in the middle of development."""
        fut: concurrent.futures.Future[str] = concurrent.futures.Future()
        self._clarification_future = fut

        def show_modal():
            self.query_one("#waiting-indicator", Static).add_class("visible")
            def on_dismiss(ans: Optional[str]):
                self.query_one("#waiting-indicator", Static).remove_class("visible")
                if not fut.done():
                    fut.set_result(ans or "Action cancelled by user")
                self._clarification_future = None

            modal = ClarificationModal(question=question, options=options, context=context)
            self.push_screen(modal, callback=on_dismiss)

        self.call_from_thread(show_modal)
        # Block worker thread safely until modal is dismissed and future resolved
        return fut.result()

    def action_clear_screen(self) -> None:
        log = self.query_one("#chat-log", PinnedRichLog)
        log.clear()
        log._set_pinned(False)

    def action_quit(self) -> None:
        if self._busy and not self._quit_when_finished:
            self._quit_when_finished = True
            self.query_one("#chat-log", RichLog).write(Text(
                "Will quit when the current task ends. Press Ctrl+C again to exit now; Esc cancels a question.",
                style="yellow"))
            return
        self.exit()

    def action_toggle_telemetry(self) -> None:
        self._show_telemetry = not self._show_telemetry
        self._apply_layout()

    def action_choose_model(self) -> None:
        if self._busy:
            return
        self._activity = "Fetching models"
        self._refresh_status()
        self._fetch_model_catalog()

    @work(thread=True)
    def _fetch_model_catalog(self) -> None:
        error = ""
        try:
            models = fetch_models(self.api_key)
        except Exception as exc:
            models = self._model_catalog
            error = f"Model catalog unavailable ({type(exc).__name__}); showing saved choices."
        if self.is_running:
            self.call_from_thread(self._show_model_picker, models, error)

    def _show_model_picker(self, models: list[CatalogModel], error: str = "") -> None:
        self._activity = "Ready"
        self._refresh_status()
        if error:
            self.query_one("#chat-log", RichLog).write(Text(error, style="yellow"))
        if models:
            self._model_catalog = models
        choices = [("auto", "AUTO  ·  route by task complexity")]
        seen = {"auto"}
        for model_id in [*MODEL_TIERS.values(), self.agent.explicit_model or ""]:
            if model_id and model_id not in seen:
                choices.append((model_id, f"★ {model_id}"))
                seen.add(model_id)
        for model in models:
            if model.id not in seen:
                context = f" · {model.context_length // 1000}k context" if model.context_length else ""
                choices.append((model.id, f"{model.name}  ·  {model.id}{context}"))
                seen.add(model.id)
        def picked(model_id: str | None) -> None:
            if model_id is not None:
                self._set_model(model_id)
        self.push_screen(QuickSelectModal("Choose an OpenRouter model", choices,
                                          current=self.agent.explicit_model or "auto"), callback=picked)

    def _set_model(self, model_id: str) -> None:
        self.agent.explicit_model = None if model_id.lower() == "auto" else model_id
        if self.agent.explicit_model:
            self.agent.llm_client.default_model = model_id
        else:
            self.agent.llm_client.default_model = MODEL_TIERS["standard"]
        self.query_one("#chat-log", RichLog).write(Text(f"✓ Model: {model_id}", style="green"))
        self.query_one("#telemetry", ClassifierTelemetryWidget).update_telemetry(
            model=self.agent.explicit_model or self.agent.llm_client.default_model,
            selection="forced" if self.agent.explicit_model else "auto")
        self._save_session()
        self._refresh_status()

    def _set_safety(self, profile: str) -> None:
        if profile not in {"turbo", "cautious"}:
            self.query_one("#chat-log", RichLog).write(Text("Choose turbo or cautious.", style="yellow"))
            return
        self.agent.safety_profile = profile
        self._save_session()
        self._refresh_status()
        self.query_one("#chat-log", RichLog).write(Text(f"✓ Interaction profile: {profile}", style="green"))

    def action_choose_session(self) -> None:
        if self._busy:
            return
        choices = [(saved.id, f"{saved.title}  ·  {saved.id}  ·  {saved.workspace}")
                   for saved in self.session_store.list(limit=100)]
        self.push_screen(QuickSelectModal("Choose a session", choices, current=self.session.id),
                         callback=lambda session_id: self._load_session(session_id) if session_id else None)

    def _load_session(self, session_id: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        saved = self.session_store.load(session_id)
        if saved is None:
            log.write(Text(f"Session not found: {session_id}", style="red"))
            return
        if not Path(saved.workspace).is_dir():
            log.write(Text(f"Session workspace is unavailable: {saved.workspace}", style="red"))
            return
        self._save_session()
        self.session = saved
        self.workspace_root = saved.workspace
        self._restore_session()
        self._last_tool_output = ""
        self._last_agent_content = ""
        telemetry = self.query_one("#telemetry", ClassifierTelemetryWidget)
        telemetry.reset_telemetry(classifier_engine=self.agent.classifier_backend.name,
                                  classifier_model=self.agent.classifier_backend.model)
        telemetry.update_telemetry(
            domain_mode=self.agent.forced_mode.value if self.agent.forced_mode else "—",
            domain_selection="forced" if self.agent.forced_mode else "auto",
            thinking_level=self.agent.forced_thinking.value if self.agent.forced_thinking else "—",
            thinking_tokens=BUDGET_TOKENS[self.agent.forced_thinking] if self.agent.forced_thinking else 0,
            thinking_selection="forced" if self.agent.forced_thinking else "auto",
            tokens_saved_estimate=self._tokens_saved_estimate,
            provider_cached_tokens=self._cached_tokens,
            provider_prompt_tokens=self.prompt_tokens)
        log.clear()
        log.write(Text(f"Loaded session {saved.id}: {saved.title} ({len(saved.messages)} messages)", style="green"))
        self._render_session_messages(log)
        if self._session_restore_warning:
            log.write(Text(self._session_restore_warning, style="yellow"))
        self._refresh_status()

    def _render_session_messages(self, log: RichLog) -> None:
        """Restore recent conversation context without exposing the system prompt."""
        history = [message for message in self.agent.messages if message.get("role") != "system"]
        for message in history[-40:]:
            role = message.get("role")
            content = str(message.get("content") or "")
            if role == "user":
                log.write(Text("\nDeveloper > " + content, style="bold"))
            elif role == "assistant":
                if content:
                    log.write(Text("\nAgent:", style="bold magenta"))
                    log.write(Markdown(content))
                for call in message.get("tool_calls") or []:
                    name = (call.get("function") or {}).get("name", "tool")
                    log.write(Text(f"⚡ {name}", style="yellow"))
            elif role == "tool":
                label = message.get("name", "tool")
                preview = content[:800] + ("\n… Use /export for the full transcript." if len(content) > 800 else "")
                log.write(Text(f"{label}: {preview}", style="green" if not content.startswith("ERROR:") else "red"))

    def action_new_session(self) -> None:
        if self._busy:
            self.query_one("#chat-log", RichLog).write(Text("Wait for the current task to finish.", style="yellow"))
            return
        self._reset_session(new=True)

    def action_choose_theme(self) -> None:
        original = self.theme
        def selected(theme: str | None) -> None:
            if theme is None:
                self.theme = original
                return
            try:
                self.config.save_theme(theme)
            except OSError as exc:
                self.theme = original
                self.query_one("#chat-log", RichLog).write(Text(f"Could not save theme: {exc}", style="red"))
                return
            self.theme = theme
            self.saved_theme = theme
            self.query_one("#chat-log", RichLog).write(Text(f"✓ Theme saved: {theme}", style="green"))
        self.push_screen(ThemePickerModal(original, tuple(t for t in THEME_CHOICES if t in self.available_themes)), callback=selected)

    def _reset_session(self, *, new: bool, title: str = "New session") -> None:
        if new:
            self._save_session()
            self.session = self.session_store.create(self.workspace_root, title)
        else:
            self.session.title = title
        self.agent.messages = [{"role": "system", "content": self.agent.system_prompt}]
        self.agent.active_skills.clear()
        self.agent.forced_mode = self._cli_mode_override
        self.agent.forced_thinking = self._cli_thinking_override
        self.agent.safety_profile = self._cli_safety_override or self._default_safety
        self._last_tool_output = ""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._tokens_saved_estimate = 0
        self._cached_tokens = 0
        self._activity = "Ready"
        telemetry = self.query_one("#telemetry", ClassifierTelemetryWidget)
        telemetry.reset_telemetry(classifier_engine=self.agent.classifier_backend.name,
                                  classifier_model=self.agent.classifier_backend.model,
                                  model=self.agent.explicit_model or self.agent.llm_client.default_model,
                                  selection="forced" if self.agent.explicit_model else "auto")
        telemetry.update_telemetry(
            domain_mode=self.agent.forced_mode.value if self.agent.forced_mode else "—",
            domain_selection="forced" if self.agent.forced_mode else "auto",
            thinking_level=self.agent.forced_thinking.value if self.agent.forced_thinking else "—",
            thinking_tokens=BUDGET_TOKENS[self.agent.forced_thinking] if self.agent.forced_thinking else 0,
            thinking_selection="forced" if self.agent.forced_thinking else "auto")
        log = self.query_one("#chat-log", PinnedRichLog)
        log.clear()
        log._set_pinned(False)
        log.write(Text(f"━━━ {'New session' if new else 'Session reset'} · {self.session.id} ━━━", style="bold cyan"))
        self._save_session()
        self._refresh_status()

    def on_input_changed(self, event: Input.Changed) -> None:
        value = event.value.strip().lower()
        hints = self.query_one("#command-hints", Static)
        matches = [cmd for cmd in COMMANDS if cmd.startswith(value)][:10] if value.startswith("/") and " " not in value and self.size.height >= 12 else []
        hints.update(Text("  ".join(matches), style="bold cyan"))
        hints.set_class(bool(matches), "visible")
        self.query_one("#input-container", Container).styles.height = 4 if matches else 3

    def on_pinned_rich_log_pin_changed(self, event: PinnedRichLog.PinChanged) -> None:
        self.query_one("#scroll-indicator", Static).set_class(event.pinned, "visible")

    def on_app_blur(self, event: events.AppBlur) -> None:
        self._has_focus = False

    def on_app_focus(self, event: events.AppFocus) -> None:
        self._has_focus = True

    def action_show_help(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write("[bold yellow]Available Commands:[/bold yellow]")
        log.write("  /key <API_KEY> | status | clear  - Manage private OpenRouter key")
        log.write("  /model [MODEL_ID]      - Browse models (F4) or set one directly")
        log.write("  /models                - Search live OpenRouter catalog")
        log.write("  /tier <fast|standard|reasoning> - Force model tier")
        log.write("  /mode <coding|research|science|security|auto> - Set operational mode")
        log.write("  /thinking <none|low|medium|deep|auto> - Set reasoning budget")
        log.write("  /safety <turbo|cautious> - Control ordinary ambiguity prompts (default: turbo)")
        log.write("  /classifier <semif|sklearn|backend> [model/path] - Switch decision engine")
        log.write("  /workspace [path]      - Show or change working directory")
        log.write("  /sessions              - Browse saved sessions (F5); /sessions list prints IDs")
        log.write("  /session new [title] | load <id> | save")
        log.write("  /skills                - List installed skills")
        log.write("  /skill list            - Browse 22 built-in and custom skills")
        log.write("  /skill <name|off>      - Force a skill or restore automatic routing")
        log.write("  /output                - Show the last tool result (up to 20k characters)")
        log.write("  /new | /reset          - Start a new session or reset current one")
        log.write("  /theme [name]          - Preview and save terminal theme (F2)")
        log.write("  /history               - Show recent prompts; Up/Down recalls prompts")
        log.write("  /export [markdown|json] - Save the session to output/sessions/")
        log.write("  F3                    - Show or hide telemetry")
        log.write("  /clear                 - Clear chat history")
        log.write("  /exit                  - Exit application")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        input_widget = self.query_one("#prompt-input", HistoryInput)
        if self._busy and not text.startswith("/"):
            self.query_one("#chat-log", RichLog).write(Text("Agent is still working. Wait for this task to finish.", style="yellow"))
            return
        input_widget.value = ""
        input_widget.reset_navigation()

        # Handle slash commands
        if text.startswith("/"):
            self._handle_slash_command(text)
            return

        try:
            if self.history_store.record(text):
                input_widget.history = list(self.history_store.entries)
        except OSError as exc:
            self.query_one("#chat-log", RichLog).write(Text(f"Prompt history could not be saved: {exc}", style="yellow"))

        log = self.query_one("#chat-log", RichLog)
        log.write(Text("\nDeveloper > " + text, style="bold"))
        if self.session.title == "New session":
            self.session.title = text[:72]

        # Launch agent execution in non-blocking worker thread
        self._busy = True
        self._activity = "Classifying task"
        self._last_agent_content = ""
        self.query_one("#telemetry", ClassifierTelemetryWidget).update_telemetry(memory_resolution=False)
        self._refresh_status()
        self._bell_rung = False
        self.execute_agent_task(text)

    def _handle_slash_command(self, cmd_text: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        parts = cmd_text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if self._busy and cmd in {"/key", "/model", "/models", "/tier", "/mode", "/thinking", "/safety", "/classifier", "/sessions",
                                  "/workspace", "/session", "/skill", "/new", "/reset", "/export"}:
            log.write(Text("Wait for the current task before changing settings or exiting.", style="yellow"))
            return

        if cmd in ("/exit", "/quit"):
            self.action_quit()
        elif cmd == "/clear":
            self.action_clear_screen()
        elif cmd == "/help":
            self.action_show_help()
        elif cmd == "/new":
            self._reset_session(new=True, title=arg or "New session")
        elif cmd == "/reset":
            self._reset_session(new=False)
        elif cmd == "/history":
            recent = self.history_store.entries[-15:]
            log.write(Text("Recent prompts:" if recent else "No saved prompts yet.", style="bold cyan"))
            for index, prompt in enumerate(recent, 1):
                log.write(Text(f"{index:2}. {prompt}"))
        elif cmd == "/theme":
            if not arg:
                self.action_choose_theme()
            elif arg in self.available_themes:
                try:
                    self.config.save_theme(arg)
                except OSError as exc:
                    log.write(Text(f"Could not save theme: {exc}", style="red"))
                    return
                self.theme = arg
                self.saved_theme = arg
                log.write(Text(f"✓ Theme saved: {arg}", style="green"))
            else:
                log.write(Text(f"Unknown theme: {arg}. Use /theme to browse.", style="yellow"))
        elif cmd == "/export":
            self._export_session(arg or "markdown")
        elif cmd == "/key":
            if not arg:
                log.write(Text("Use /key <OPENROUTER_API_KEY>, /key status, or /key clear", style="yellow"))
                return
            if arg.lower() == "status":
                masked = (self.api_key[:8] + "••••••••" + self.api_key[-4:]
                          if self.api_key and len(self.api_key) > 12 else
                          "••••••••" if self.api_key else "none")
                log.write(Text(f"API key source: {self.key_source} · {masked}", style="cyan"))
                return
            if arg.lower() == "clear":
                try:
                    self.config.clear_key()
                except OSError as exc:
                    log.write(Text(f"Could not clear saved key: {exc}", style="red"))
                    return
                self.api_key = None
                self.key_source = "offline"
                self.agent.llm_client = LLMClient(api_key=None, base_url=self.base_url,
                    default_model=self.agent.llm_client.default_model, force_mock=True)
                self.llm_client = self.agent.llm_client
                if self.agent.classifier_backend.name == "openrouter":
                    self.agent.classifier_backend.api_key = None
                log.write(Text("✓ Saved API key cleared; offline mock mode is active.", style="green"))
                self._refresh_status()
                return
            try:
                self.config.save_key(arg)
            except (OSError, ValueError) as exc:
                log.write(Text(f"Could not save API key: {exc}", style="red"))
                return
            self.api_key = arg
            self.key_source = "saved configuration"
            self.agent.llm_client = LLMClient(api_key=arg, base_url=self.base_url,
                                              default_model=self.agent.llm_client.default_model)
            self.llm_client = self.agent.llm_client
            if self.agent.classifier_backend.name == "openrouter":
                self.agent.classifier_backend.api_key = arg
            log.write(Text("✓ API key securely saved for future sessions", style="green"))
            self._refresh_status()
        elif cmd == "/model":
            if not arg:
                self.action_choose_model()
                return
            self._set_model(arg)
        elif cmd == "/models":
            self.action_choose_model()
        elif cmd == "/tier":
            if arg in MODEL_TIERS:
                target_model = MODEL_TIERS[arg]
                self.agent.llm_client.default_model = target_model
                self.agent.explicit_model = target_model
                log.write(f"[green]✓ Switched to {arg.upper()} tier ({target_model})[/green]")
                self.query_one("#telemetry", ClassifierTelemetryWidget).update_telemetry(
                    model=target_model, tier=arg, selection="forced")
                self._save_session()
                self._refresh_status()
            else:
                log.write(f"[red]Invalid tier. Choose from: {', '.join(MODEL_TIERS.keys())}[/red]")
        elif cmd == "/mode":
            if not arg:
                log.write(Text(f"Mode: {self.agent.forced_mode.value if self.agent.forced_mode else 'auto'}", style="cyan"))
                return
            try:
                self.agent.forced_mode = parse_domain_mode(arg)
            except ValueError as exc:
                log.write(Text(str(exc), style="red"))
                return
            self.query_one("#telemetry", ClassifierTelemetryWidget).update_telemetry(
                domain_mode=self.agent.forced_mode.value if self.agent.forced_mode else "—",
                domain_selection="forced" if self.agent.forced_mode else "auto")
            self._save_session()
            self._refresh_status()
            log.write(Text(f"✓ Mode: {self.agent.forced_mode.value if self.agent.forced_mode else 'auto'}", style="green"))
        elif cmd == "/thinking":
            if not arg:
                log.write(Text(f"Thinking: {self.agent.forced_thinking.value if self.agent.forced_thinking else 'auto'}", style="cyan"))
                return
            try:
                self.agent.forced_thinking = parse_thinking_level(arg)
            except ValueError as exc:
                log.write(Text(str(exc), style="red"))
                return
            self.query_one("#telemetry", ClassifierTelemetryWidget).update_telemetry(
                thinking_level=self.agent.forced_thinking.value if self.agent.forced_thinking else "—",
                thinking_tokens=BUDGET_TOKENS[self.agent.forced_thinking] if self.agent.forced_thinking else 0,
                thinking_selection="forced" if self.agent.forced_thinking else "auto")
            self._save_session()
            self._refresh_status()
            log.write(Text(f"✓ Thinking: {self.agent.forced_thinking.value if self.agent.forced_thinking else 'auto'}", style="green"))
        elif cmd == "/safety":
            if not arg:
                self.push_screen(QuickSelectModal("Choose interaction profile", [
                    ("turbo", "Turbo · proceed through uncertainty; stop for destructive actions"),
                    ("cautious", "Cautious · ask on high semantic uncertainty"),
                ], current=self.agent.safety_profile),
                    callback=lambda selected: self._set_safety(selected) if selected else None)
            else:
                self._set_safety(arg)
        elif cmd == "/classifier":
            args = arg.split(maxsplit=1)
            if not args:
                log.write(Text(f"Classifier: {self.agent.classifier_backend.name} {self.agent.classifier_backend.model}"))
                return
            try:
                backend = create_backend(args[0], args[1] if len(args) > 1 else None, self.classifier_endpoint,
                    api_key=self.api_key, device=self.semif_device, load_in_4bit=self.semif_4bit,
                    temperature=self.semif_temperature)
            except (ValueError, RuntimeError, OSError) as exc:
                log.write(Text(str(exc), style="red"))
                return
            self.agent.classifier_backend = backend
            log.write(Text(f"✓ Classifier: {backend.name} {backend.model}", style="green"))
            self.query_one("#telemetry", ClassifierTelemetryWidget).update_telemetry(
                classifier_engine=backend.name, classifier_model=backend.model)
            self._save_session()
            self._refresh_status()
        elif cmd == "/workspace":
            if not arg:
                log.write(Text(f"Workspace: {self.workspace_root}", style="cyan"))
                return
            if self._busy:
                log.write(Text("Wait for the current task before changing workspace.", style="yellow"))
                return
            try:
                self.agent.set_workspace(arg)
            except ValueError as exc:
                log.write(Text(str(exc), style="red"))
                return
            self.workspace_root = str(self.agent.workspace_root)
            self.skill_catalog = SkillCatalog(self.workspace_root)
            self.agent.active_skills.clear()
            self._save_session()
            self._refresh_status()
            log.write(Text(f"Workspace: {self.workspace_root}", style="green"))
        elif cmd == "/sessions":
            if arg == "list":
                for saved in self.session_store.list():
                    log.write(Text(f"{saved.id}  {saved.title}  ·  {saved.workspace}", style="cyan" if saved.id == self.session.id else ""))
            else:
                self.action_choose_session()
        elif cmd == "/session":
            if self._busy:
                log.write(Text("Wait for the current task before switching sessions.", style="yellow"))
                return
            action, _, value = arg.partition(" ")
            if action == "new":
                self._reset_session(new=True, title=value.strip() or "New session")
            elif action == "load" and value.strip():
                self._load_session(value.strip())
            elif action == "save":
                self._save_session()
                log.write(Text(f"Saved session {self.session.id}", style="green"))
            else:
                log.write(Text("Use /session new [title], /session load <id>, or /session save", style="yellow"))
            self._refresh_status()
        elif cmd == "/skills" or (cmd == "/skill" and arg == "list"):
            skills = self.skill_catalog.all()
            for name, skill in skills.items():
                marker = "●" if name in self.agent.active_skills else "○"
                log.write(Text(f"{marker} {skill.icon} {name} · {skill.category} — {skill.trigger}", style="cyan"))
        elif cmd == "/skill":
            if not arg:
                log.write(Text("Use /skills to list and /skill <name> to toggle.", style="yellow"))
                return
            if arg == "off":
                self.agent.active_skills.clear()
            elif arg in self.agent.active_skills:
                del self.agent.active_skills[arg]
            else:
                try:
                    self.agent.active_skills[arg] = self.skill_catalog.read(arg)
                except (ValueError, OSError) as exc:
                    log.write(Text(str(exc), style="red"))
                    return
            self._save_session()
            self._refresh_status()
            log.write(Text(f"Active skills: {', '.join(self.agent.active_skills) or 'none'}", style="green"))
        elif cmd == "/output":
            if self._last_tool_output:
                log.write(Text(self._last_tool_output))
            else:
                log.write(Text("No tool result yet.", style="yellow"))
        else:
            log.write(Text(f"Unknown command: {cmd}. Type /help for options.", style="red"))

    def _export_session(self, export_format: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        if export_format not in {"markdown", "json"}:
            log.write(Text("Use /export markdown or /export json", style="yellow"))
            return
        self._save_session()
        export_dir = Path(self.workspace_root) / "output" / "sessions"
        suffix = "md" if export_format == "markdown" else "json"
        path = export_dir / f"{self.session.id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.{suffix}"
        transcript = {
            "session_id": self.session.id, "title": self.session.title,
            "workspace": self.workspace_root, "model": self.agent.explicit_model or "auto",
            "usage": {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens},
            "messages": self.agent.messages,
        }
        if export_format == "json":
            content = json.dumps(transcript, indent=2, ensure_ascii=False) + "\n"
        else:
            lines = [f"# {self.session.title}", "", f"Session: `{self.session.id}`",
                     f"Workspace: `{self.workspace_root}`", "",
                     f"Tokens: prompt {self.prompt_tokens:,}, completion {self.completion_tokens:,}", ""]
            for message in self.agent.messages:
                lines.extend([f"## {message.get('role', 'message').title()}", "", str(message.get("content") or ""), ""])
                for call in message.get("tool_calls", []):
                    function = call.get("function", {})
                    lines.extend([f"Tool call: `{function.get('name', 'unknown')}`", "", "````json",
                                  str(function.get("arguments", "{}")), "````", ""])
            content = "\n".join(lines)
        try:
            export_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            log.write(Text(f"Could not export session: {exc}", style="red"))
            return
        log.write(Text(f"✓ Exported session to {path}", style="green"))

    @work(thread=True)
    def execute_agent_task(self, task_text: str) -> None:
        """Worker thread executing the agent task and streaming events back to the UI."""
        try:
            for event in self.agent.run_stream(task_text):
                self.call_from_thread(self._render_event, event)
        except Exception as exc:
            if self.is_running:
                try:
                    self.call_from_thread(self._render_error, f"Agent error: {type(exc).__name__}: {exc}")
                except RuntimeError:
                    pass
        finally:
            try:
                if self.is_running:
                    self.call_from_thread(self._finish_task)
            except RuntimeError:
                pass

    def _finish_task(self) -> None:
        self._busy = False
        self._activity = "Ready"
        self._save_session()
        self._refresh_status()
        if self._quit_when_finished:
            self.exit()
        else:
            self.query_one("#prompt-input", Input).focus()

    def _render_error(self, message: str) -> None:
        self._activity = "Error"
        self._refresh_status()
        self.query_one("#chat-log", RichLog).write(Text(message, style="bold red"))

    def _render_event(self, event: AgentEvent) -> None:
        telemetry = self.query_one("#telemetry", ClassifierTelemetryWidget)
        log = self.query_one("#chat-log", RichLog)
        et = event.event_type
        p = event.payload

        if et == "agent_stage":
                stage = p["stage"]
                self._activity = {
                    "thinking": f"Thinking / generating · step {p['step']}",
                    "generating": f"Generating · step {p['step']}",
                    "tool_running": f"Running {p.get('tool', '')}",
                    "verifying": f"Verifying {p.get('tool', '')}",
                }.get(stage, stage)
                self._refresh_status()
        elif et == "skill_classification":
                telemetry.update_telemetry(
                    probabilities=p["probabilities"],
                    primary_skill=p["primary_skill"],
                    classifier_engine=p["backend"], classifier_model=p["classifier_model"],
                    classifier_latency_ms=p["latency_ms"],
                )
        elif et == "specialized_skill":
                telemetry.update_telemetry(specialized_skills=p["skills"],
                    skill_confidence=p["confidence"], skill_tools=p["tools"])
                if p["skills"]:
                    names = " → ".join(item["title"] for item in p["skills"])
                    log.write(Text(f"Skill: {names} ({p['selection']}, {p['confidence']:.0%})", style="cyan"))
        elif et == "skill_verification":
                if p["missing"]:
                    log.write(Text(f"Skill checks pending ({p['skill']}): {', '.join(p['missing'])}", style="yellow"))
                else:
                    log.write(Text(f"Skill checks verified: {p['skill']}", style="green"))
        elif et == "ambiguity_assessment":
                telemetry.update_telemetry(
                    entropy=p["entropy"],
                    margin=p["confidence_margin"],
                    risk_level=p["risk_level"],
                )
        elif et == "domain_mode":
                telemetry.update_telemetry(domain_mode=p["mode"], domain_selection=p.get("selection", "auto"))
                self._refresh_status()
        elif et == "thinking_budget":
                telemetry.update_telemetry(thinking_level=p["level"], thinking_tokens=p["tokens"],
                                           thinking_selection=p.get("selection", "auto"),
                                           classifier_latency_ms=p["classifier_total_ms"])
                self._refresh_status()
        elif et == "model_routing":
                telemetry.update_telemetry(
                    tier=p["tier"],
                    model=p["model"],
                    selection=p["selection"],
                )
        elif et == "classifier_error":
                log.write(Text(f"Classifier {p['backend']} failed: {p['error']}; using sklearn", style="yellow"))
        elif et == "classifier_loading":
                self._activity = "Loading local classifier"
                self._refresh_status()
                log.write(Text(f"Loading local SemIf model {p['model']} for its first decision…", style="cyan"))
        elif et == "classifier_fallback":
                telemetry.update_telemetry(classifier_engine=p["backend"], classifier_model=p["model"])
        elif et == "clarification_needed":
                self._activity = "Waiting for your choice"
                self._refresh_status()
                log.write(Text(f"Clarification needed: {p.get('reason') or p['question']}", style="yellow"))
        elif et == "clarification_answered":
                self._activity = "Resuming task"
                self._refresh_status()
        elif et == "llm_error":
                log.write(Text(p["message"], style="bold red"))
        elif et == "storage_error":
                log.write(Text(p["error"], style="yellow"))
        elif et == "thought":
                self._last_agent_content = p["content"]
                log.write(Text(f"\nAgent · {p.get('model', 'model')}:", style="bold magenta"))
                log.write(Markdown(p["content"]))
        elif et == "tool_call":
                arguments = json.dumps(p["arguments"], ensure_ascii=False)
                preview = arguments[:400] + ("… Use /output after completion." if len(arguments) > 400 else "")
                log.write(Text(f"⚡ {p['name']}  {preview}", style="yellow"))
        elif et == "tool_result":
                self._tokens_saved_estimate += p.get("saved_tokens_estimate", 0)
                telemetry.update_telemetry(tokens_saved_estimate=self._tokens_saved_estimate)
                status_color = "green" if p["success"] else "red"
                log.write(Text(f"Result ({p['time_ms']} ms) · {p['name']}", style=status_color))
                output = p["output"] or ""
                if p.get("error"):
                    output = f"ERROR: {p['error']}\n{output}" if output else f"ERROR: {p['error']}"
                output = output or "(empty)"
                self._last_tool_output = output
                if "Diff:\n" in output or output.startswith("@@"):
                    diff = output.split("Diff:\n", 1)[-1]
                    log.write(Syntax(diff, "diff", theme="monokai", line_numbers=False))
                elif len(output) > 800:
                    log.write(Text(output[:800] + f"\n… {len(output)-800} more characters. Use /output to expand."))
                else:
                    log.write(Text(output))
                if p.get("time_ms", 0) >= 5000 and not self._has_focus and not self._bell_rung:
                    self.bell()
                    self._bell_rung = True
        elif et == "verification":
                log.write(Text(f"Verification: {p['status']} → {p['action']}", style="green" if p["status"] == "SUCCESS" else "red"))
        elif et == "response":
                if p.get("content") and p["content"] != self._last_agent_content:
                    log.write(Text("\nAgent:", style="bold magenta"))
                    log.write(Markdown(p["content"]))
                elif not p.get("content") and not p["success"]:
                    log.write(Text("No final answer was produced; inspect the last tool result or retry.", style="yellow"))
                usage = p.get("usage") or {}
                self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                self._cached_tokens += int(p.get("cached_tokens", 0) or 0)
                telemetry.update_telemetry(provider_cached_tokens=self._cached_tokens,
                                           provider_prompt_tokens=self.prompt_tokens)
                status = "✓ Task completed" if p["success"] else "Task stopped before completion"
                log.write(Text(f"\n{status} in {p['total_time_ms']} ms ({p['steps']} steps)\n",
                               style="bold green" if p["success"] else "bold yellow"))
                if p.get("total_time_ms", 0) >= 5000 and not self._has_focus and not self._bell_rung:
                    self.bell()
                    self._bell_rung = True
                self._refresh_status()
        elif et == "memory_hit":
                self._activity = "Verified memory · 0 API tokens"
                telemetry.update_telemetry(probabilities={key: 0.0 for key in telemetry.probabilities},
                    entropy=0.0, margin=0.0, risk_level="idle", tier="memory", model="Local verified cache",
                    domain_mode="—", thinking_level="—", thinking_tokens=0, memory_resolution=True)
                log.write(Text("⚡ Verified memory answer reused; no model request.", style="bold green"))
                self._refresh_status()
        elif et == "clarification_memory_hit":
                log.write(Text("Using a saved preference for this question.", style="cyan"))
