"""Textual Terminal User Interface (TUI) application for Adaptive Agent Harness."""

from __future__ import annotations

from pathlib import Path
import asyncio
import concurrent.futures
from datetime import datetime
import json
import os
import re
import threading
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
from adaptive_harness.agent.swarm import DeveloperAgentWorker, SwarmCoordinator
from adaptive_harness.agent.skills import SkillCatalog
from adaptive_harness.classifiers.engine import create_backend
from adaptive_harness.classifiers.domain_classifier import parse_domain_mode
from adaptive_harness.classifiers.thinking_classifier import parse_thinking_level, BUDGET_TOKENS
from adaptive_harness.data.storage import ExperienceRepository
from adaptive_harness.data.config import ConfigManager, PromptHistoryStore
from adaptive_harness.data.credentials import CredentialsManager
from adaptive_harness.data.sessions import SessionStore
from adaptive_harness.llm.client import LLMClient, MODEL_TIERS
from adaptive_harness.classifiers.thinking_classifier import ThinkingLevel
from adaptive_harness.llm.providers import PROVIDERS, PROVIDER_TIERS, provider_for_url
from adaptive_harness.llm.catalog import CatalogModel, fetch_models
from adaptive_harness.tui.widgets import (ClarificationModal, ClassifierTelemetryWidget,
    HistoryInput, PinnedRichLog, ThemePickerModal, QuickSelectModal, OutputViewerModal,
    CommandPalette, DiffReviewModal, THEME_CHOICES)
from adaptive_harness.tui.formatting import format_model_markdown
from adaptive_harness.workspace.worktree import WorktreeManager, WorktreeTask, WorktreeError


COMMANDS = ("/key", "/provider", "/model", "/mode", "/models", "/tier", "/theme", "/thinking", "/safety", "/classifier", "/new",
            "/clear", "/history", "/help", "/exit", "/reset", "/workspace", "/sessions", "/usage",
            "/session", "/skills", "/skill", "/output", "/tool-output", "/copy", "/export",
            "/diff", "/isolation", "/swarm")

COMMAND_DESCRIPTIONS = {
    "/provider": "Switch task model provider",
    "/key": "Manage private provider API keys",
    "/model": "Choose a model or restore auto routing",
    "/models": "Search available models",
    "/tier": "Force fast, standard, or reasoning",
    "/mode": "Choose coding, research, science, or security",
    "/thinking": "Set reasoning budget",
    "/safety": "Choose interaction profile",
    "/theme": "Preview and save a terminal theme",
    "/classifier": "Switch local classification engine",
    "/new": "Start a new session",
    "/reset": "Clear current session state",
    "/clear": "Clear the chat log",
    "/history": "Show earlier prompts",
    "/help": "Show commands and shortcuts",
    "/exit": "Close the TUI",
    "/workspace": "Choose a working directory",
    "/sessions": "Browse saved sessions",
    "/session": "Load or save a session",
    "/usage": "Show token usage and cost",
    "/skills": "Browse installed skills",
    "/skill": "Force or disable a skill",
    "/output": "Select or copy the latest agent response",
    "/tool-output": "Inspect the latest tool result",
    "/copy": "Copy selected chat text",
    "/export": "Export this session",
    "/diff": "Review isolated changes",
    "/isolation": "Set worktree isolation: auto, on, off",
    "/swarm": "Set multi-agent mode: auto, on, off",
}


def matching_commands(value: str) -> list[tuple[str, str]]:
    """Prefer prefix matches, then ordered-subsequence matches for discovery."""
    if not value.startswith("/") or " " in value:
        return []
    needle = value[1:].casefold()
    def fuzzy(command: str) -> bool:
        letters = iter(command[1:].casefold())
        return all(any(letter == candidate for candidate in letters) for letter in needle)
    matched = [command for command in COMMANDS if command[1:].startswith(needle)]
    matched.extend(command for command in COMMANDS if command not in matched and fuzzy(command))
    return [(command, COMMAND_DESCRIPTIONS[command]) for command in matched]


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
    Screen.compact Header, Screen.compact #scroll-indicator { margin: 0; }
    Screen.compact #input-container { margin: 0 0 1 0; }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_screen", "Clear Log"),
        ("ctrl+shift+c", "copy_output", "Copy Output"),
        ("ctrl+n", "new_session", "New Session"),
        ("f1", "show_help", "Help"),
        ("f2", "choose_theme", "Theme"),
        ("f3", "review_diff", "Diff Review"),
        ("f4", "choose_model", "Models"),
        ("f5", "choose_session", "Sessions"),
        ("f6", "toggle_telemetry", "Telemetry"),
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        backup_providers: tuple[str, ...] | None = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        db_path: Path | str = "output/experience.db",
        workspace_root: Optional[str] = None,
        classifier_backend: str = "auto",
        classifier_model: Optional[str] = None,
        classifier_endpoint: Optional[str] = None,
        overseer_model: Optional[str] = None,
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
        self.credentials = CredentialsManager(config_dir)
        try:
            self.provider_keys = self.credentials.load()
        except (OSError, ValueError, json.JSONDecodeError):
            self.provider_keys = {}
        legacy_config_key = saved_config.get("api_key") and "openrouter" not in self.provider_keys
        if legacy_config_key:
            # Existing users retain their one-time saved key after the move to
            # dedicated per-provider credentials.
            self.provider_keys["openrouter"] = saved_config["api_key"]
        if provider is not None and provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")
        self.provider_name = provider or (provider_for_url(base_url) if base_url else
                                          saved_config.get("provider", "openrouter"))
        if self.provider_name not in PROVIDERS:
            self.provider_name = "openrouter"
        self.backup_providers = tuple(backup_providers or ())
        if any(name not in PROVIDERS for name in self.backup_providers):
            raise ValueError("Unknown backup provider")
        self.history_store = PromptHistoryStore(config_dir)
        env_key = os.environ.get(PROVIDERS[self.provider_name].env_key or "")
        self.api_key = api_key or env_key or self.provider_keys.get(self.provider_name)
        if api_key:
            self.provider_keys[self.provider_name] = api_key
        self.key_source = ("CLI flag" if api_key else "environment" if env_key else
                           "saved configuration" if legacy_config_key else
                           "saved credentials" if self.provider_keys.get(self.provider_name) else "offline")
        self.saved_theme = saved_config.get("theme", "textual-dark")
        self.base_url = base_url
        if default_model and default_model.lower() == "auto":
            default_model = None
        self._cli_model_override = default_model
        self._cli_provider_override = provider
        self.default_model = default_model or MODEL_TIERS["standard"]
        self.db_path = db_path
        self.workspace_root = str(Path(workspace_root or ".").expanduser().resolve())
        if not Path(self.workspace_root).is_dir():
            raise ValueError(f"Workspace directory does not exist: {self.workspace_root}")
        self.classifier_endpoint = classifier_endpoint
        self.overseer_model = overseer_model
        self.semif_device = semif_device
        self.semif_4bit = semif_4bit
        self.semif_temperature = semif_temperature
        initial_mode = parse_domain_mode(mode)
        initial_thinking = parse_thinking_level(thinking)
        self._cli_mode_override = initial_mode
        self._cli_thinking_override = initial_thinking
        if safety is not None and safety not in {"turbo", "balanced", "cautious", "strict"}:
            raise ValueError("Safety profile must be turbo, balanced, cautious, or strict")
        self._cli_safety_override = safety
        self._default_safety = "turbo"
        self._model_catalog: list[CatalogModel] = []
        self._activity = "Ready"
        self._activity_pulse = False
        self._palette_dismissed_value: str | None = None
        self.isolation_mode = "auto"
        self.swarm_mode = "auto"
        self._review_manager: WorktreeManager | None = None
        self._review_task: WorktreeTask | None = None
        self._review_patch = ""

        # LLM Client & Repo
        self.llm_client = LLMClient(
            api_key=self.api_key,
            base_url=self.base_url,
            default_model=self.default_model,
            provider=self.provider_name,
            provider_keys=self.provider_keys,
            backup_providers=tuple(name for name in self.backup_providers if name != self.provider_name),
            force_mock=not bool(self.api_key) and self.provider_name != "local" and not self._local_endpoint(),
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
        self._reasoning_tokens = 0
        self._cache_write_tokens = 0
        self._reported_cost_usd = 0.0
        self._cost_reported = False
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
            overseer_backend=create_backend("onnx", overseer_model) if overseer_model else None,
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
            yield CommandPalette(id="command-palette")

        yield Static(id="status-line")
        yield Static("[Pinned: Scroll to bottom ↓]", id="scroll-indicator")

        with Container(id="input-container"):
            yield HistoryInput(
                placeholder="Ask agent to code, inspect, test, or run commands (/help for commands)...",
                id="prompt-input", history=self.history_store.entries,
            )
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
        provider = ("🟡 Offline Mock Engine" if self.agent.llm_client.is_mock else
                    f"🟢 {self.agent.llm_client.provider.title()}")
        telemetry = self.query_one("#telemetry", ClassifierTelemetryWidget)
        model = self.agent.explicit_model or telemetry.active_model or self.agent.llm_client.default_model
        mode = ("SECURITY" if telemetry.domain_mode == "audit" else
                telemetry.domain_mode.upper() if telemetry.domain_mode != "—" else "AUTO")
        thinking = telemetry.thinking_level.upper() if telemetry.thinking_level != "—" else "AUTO"
        activity = ("● " if self._activity_pulse and self._busy else "◦ ") + self._activity
        token_total = self.prompt_tokens + self.completion_tokens
        cost_display = f"${self._reported_cost_usd:.6f}" if self._cost_reported else "cost n/a"
        context_label = (f" · Ctx:{telemetry.context_used // 1000}k/{telemetry.context_capacity // 1000}k"
                         if telemetry.context_capacity else "")
        self.sub_title = f"{provider} ({model})  ·  {activity}  ·  {mode} / {thinking}{context_label}  ·  Tokens:{token_total:,}  ·  {cost_display}"
        self.query_one("#status-line", Static).update(Text(
            f"{activity}  ·  {provider}  ·  {mode}/{thinking}  ·  Tokens {token_total:,}  ·  {cost_display}  ·  Session {self.session.id}  ·  {self.workspace_root}", style="bold cyan"))

    def _save_session(self) -> None:
        self.session.workspace = self.workspace_root
        self.session.messages = self.agent.messages
        self.session.model = self.agent.explicit_model
        self.session.skills = list(self.agent.active_skills)
        self.session.settings = {"classifier_backend": self.agent.classifier_backend.name,
                                 "provider": self.agent.llm_client.provider,
                                 "classifier_model": self.agent.classifier_backend.model,
                                 "classifier_endpoint": self.classifier_endpoint or "",
                                 "semif_device": self.semif_device, "semif_4bit": str(self.semif_4bit),
                                 "semif_temperature": str(self.semif_temperature),
                                 "mode": self.agent.forced_mode.value if self.agent.forced_mode else "auto",
                                 "thinking": self.agent.forced_thinking.value if self.agent.forced_thinking else "auto",
                                 "safety": self.agent.safety_profile,
                                 "prompt_tokens": str(self.prompt_tokens),
                                 "completion_tokens": str(self.completion_tokens),
                                 "reasoning_tokens": str(self._reasoning_tokens),
                                 "cache_write_tokens": str(self._cache_write_tokens),
                                 "reported_cost_usd": f"{self._reported_cost_usd:.12f}",
                                 "cost_reported": str(self._cost_reported).lower(),
                                 "estimated_tokens_saved": str(self._tokens_saved_estimate),
                                 "cached_tokens": str(self._cached_tokens)}
        self.session_store.save(self.session)

    def _restore_session(self, *, preserve_cli_overrides: bool = False) -> None:
        self._session_restore_warning = ""
        settings = self.session.settings or {}
        saved_provider = settings.get("provider", self.provider_name)
        if preserve_cli_overrides and self._cli_provider_override:
            saved_provider = self._cli_provider_override
        if saved_provider in PROVIDERS and saved_provider != self.agent.llm_client.provider:
            self.provider_name = saved_provider
            env_name = PROVIDERS[saved_provider].env_key
            self.api_key = os.environ.get(env_name or "") or self.provider_keys.get(saved_provider)
            self.agent.llm_client = LLMClient(api_key=self.api_key, provider=saved_provider,
                provider_keys=self.provider_keys,
                backup_providers=tuple(name for name in self.backup_providers if name != saved_provider),
                force_mock=not bool(self.api_key) and saved_provider != "local")
            self.llm_client = self.agent.llm_client
            self.base_url = self.llm_client.base_url
        self.agent.messages = self.session.messages or [{"role": "system", "content": self.agent.system_prompt}]
        self._last_agent_content = next((str(message.get("content")) for message in reversed(self.agent.messages)
            if message.get("role") == "assistant" and message.get("content") and not message.get("tool_calls")), "")
        self.agent.explicit_model = (self._cli_model_override if preserve_cli_overrides and self._cli_model_override
                                     else self.session.model)
        if self.agent.explicit_model:
            self.agent.llm_client.default_model = self.agent.explicit_model
        else:
            self.agent.llm_client.default_model = PROVIDERS[self.provider_name].default_model
        self.agent.set_workspace(self.session.workspace)
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
        if self.agent.safety_profile not in {"turbo", "balanced", "cautious", "strict"}:
            self.agent.safety_profile = "turbo"
        for field, setting in (("prompt_tokens", "prompt_tokens"),
                               ("completion_tokens", "completion_tokens"),
                               ("_reasoning_tokens", "reasoning_tokens"),
                               ("_cache_write_tokens", "cache_write_tokens"),
                               ("_tokens_saved_estimate", "estimated_tokens_saved"),
                               ("_cached_tokens", "cached_tokens")):
            try:
                setattr(self, field, max(0, int(settings.get(setting, "0"))))
            except (ValueError, TypeError):
                setattr(self, field, 0)
        try:
            self._reported_cost_usd = max(0.0, float(settings.get("reported_cost_usd", "0")))
        except (TypeError, ValueError):
            self._reported_cost_usd = 0.0
        self._cost_reported = settings.get("cost_reported", "false").lower() == "true"
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

    def action_copy_output(self) -> None:
        selected = self.screen.get_selected_text()
        content = selected.strip() if selected and selected.strip() else self._last_agent_content
        if not content:
            self.query_one("#chat-log", RichLog).write(Text("No agent output to copy yet.", style="yellow"))
            return
        try:
            self.copy_to_clipboard(content)
            self.query_one("#chat-log", RichLog).write(Text("✓ Selected text copied." if selected else
                "✓ Latest agent response copied. Select chat text with the mouse to copy a passage.", style="green"))
        except Exception as exc:
            self.query_one("#chat-log", RichLog).write(Text(f"Clipboard unavailable: {exc}", style="yellow"))

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

    def action_review_diff(self) -> None:
        if self._busy:
            return
        if not self._review_task or not self._review_patch:
            self.query_one("#chat-log", RichLog).write(Text("No isolated changes are awaiting review.", style="dim"))
            return
        if not isinstance(self.screen, DiffReviewModal):
            self.push_screen(DiffReviewModal(self._review_patch, self._review_task.id),
                             callback=self._handle_review_decision)

    def _handle_review_decision(self, decision: str | None) -> None:
        manager, task = self._review_manager, self._review_task
        if not manager or not task or decision not in {"merge", "discard"}:
            return
        log = self.query_one("#chat-log", RichLog)
        try:
            result = manager.apply(task) if decision == "merge" else (manager.abort(task) or "Isolated changes discarded")
        except WorktreeError as exc:
            log.write(Text(f"Review could not complete: {exc}. The worktree remains available; press F3 to retry.", style="bold red"))
            return
        self._review_manager = None
        self._review_task = None
        self._review_patch = ""
        self.query_one("#telemetry", ClassifierTelemetryWidget).update_telemetry(workspace_isolation="Direct workspace")
        log.write(Text(result, style="bold green" if decision == "merge" else "yellow"))

    def on_unmount(self) -> None:
        if self._review_manager and self._review_task:
            try:
                self._review_manager.abort(self._review_task)
            except WorktreeError:
                pass

    def action_choose_model(self) -> None:
        if self._busy:
            return
        self._activity = "Fetching models"
        self._refresh_status()
        loop = asyncio.get_running_loop()
        threading.Thread(target=self._fetch_model_catalog, args=(loop,), daemon=True,
                         name="adaptive-harness-model-catalog").start()

    def _fetch_model_catalog(self, loop: asyncio.AbstractEventLoop) -> None:
        error = ""
        try:
            models = fetch_models(self.api_key) if self.provider_name == "openrouter" else []
        except Exception as exc:
            models = self._model_catalog
            error = f"Model catalog unavailable ({type(exc).__name__}); showing saved choices."
        if not loop.is_closed():
            def show_picker() -> None:
                if self.is_running:
                    with self._context():
                        self._show_model_picker(models, error)
            loop.call_soon_threadsafe(show_picker)

    def _show_model_picker(self, models: list[CatalogModel], error: str = "") -> None:
        self._activity = "Ready"
        self._refresh_status()
        if error:
            self.query_one("#chat-log", RichLog).write(Text(error, style="yellow"))
        if models:
            self._model_catalog = models
        choices = [("auto", "AUTO  ·  route by task complexity")]
        seen = {"auto"}
        provider_models = ([*MODEL_TIERS.values()] if self.provider_name == "openrouter" else
                           [self.llm_client.get_model_for_tier(tier) for tier in ("fast", "standard", "reasoning")])
        for model_id in [*provider_models, self.agent.explicit_model or ""]:
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
        self.push_screen(QuickSelectModal(f"Choose a {self.provider_name.title()} model", choices,
                                          current=self.agent.explicit_model or "auto"), callback=picked)

    def _set_model(self, model_id: str) -> None:
        self.agent.explicit_model = None if model_id.lower() == "auto" else model_id
        if self.agent.explicit_model:
            self.agent.llm_client.default_model = model_id
        else:
            self.agent.llm_client.default_model = PROVIDERS[self.provider_name].default_model
        selected_catalog = next((item for item in self._model_catalog if item.id == model_id), None)
        self.agent.context_window_override = selected_catalog.context_length if selected_catalog else None
        self.query_one("#chat-log", RichLog).write(Text(f"✓ Model: {model_id}", style="green"))
        self.query_one("#telemetry", ClassifierTelemetryWidget).update_telemetry(
            model=self.agent.explicit_model or self.agent.llm_client.default_model,
            selection="forced" if self.agent.explicit_model else "auto")
        self._save_session()
        self._refresh_status()

    def _connect_provider(self, provider: str) -> None:
        self.provider_name = provider
        env_name = PROVIDERS[provider].env_key
        self.api_key = (os.environ.get(env_name or "") or self.provider_keys.get(provider))
        self.key_source = ("environment" if os.environ.get(env_name or "") else
                           "saved credentials" if self.provider_keys.get(provider) else "offline")
        selected_model = (self.agent.explicit_model if provider == self.agent.llm_client.provider else None)
        self.agent.explicit_model = selected_model
        self.agent.llm_client = LLMClient(api_key=self.api_key, provider=provider,
            provider_keys=self.provider_keys,
            default_model=selected_model or PROVIDERS[provider].default_model,
            backup_providers=tuple(name for name in self.backup_providers if name != provider),
            force_mock=not bool(self.api_key) and provider != "local")
        self.llm_client = self.agent.llm_client
        self.base_url = self.llm_client.base_url
        self.config.update(provider=provider)
        self._refresh_status()

    def _set_safety(self, profile: str) -> None:
        if profile not in {"turbo", "balanced", "cautious", "strict"}:
            self.query_one("#chat-log", RichLog).write(Text("Choose turbo, balanced, cautious, or strict.", style="yellow"))
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
                    log.write(Markdown(format_model_markdown(content)))
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
        self._last_agent_content = ""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._reasoning_tokens = 0
        self._cache_write_tokens = 0
        self._reported_cost_usd = 0.0
        self._cost_reported = False
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
        if event.input.id != "prompt-input":
            return
        if event.value != self._palette_dismissed_value:
            self._palette_dismissed_value = None
        choices = (matching_commands(event.value) if self._palette_dismissed_value is None
                   and self.size.height >= 12 else [])
        self.query_one("#command-palette", CommandPalette).set_choices(choices)

    @property
    def command_palette_visible(self) -> bool:
        return self.query_one("#command-palette", CommandPalette).has_class("visible")

    def move_command_selection(self, delta: int) -> None:
        palette = self.query_one("#command-palette", CommandPalette)
        if palette.choices:
            palette.set_choices(palette.choices, (palette.selected_index + delta) % len(palette.choices))

    def complete_selected_command(self) -> None:
        palette = self.query_one("#command-palette", CommandPalette)
        if not palette.choices:
            return
        command = palette.choices[palette.selected_index][0]
        prompt = self.query_one("#prompt-input", HistoryInput)
        prompt.value = command + " "
        prompt.cursor_position = len(prompt.value)
        palette.set_choices([])

    def dismiss_command_palette(self) -> None:
        prompt = self.query_one("#prompt-input", HistoryInput)
        self._palette_dismissed_value = prompt.value
        self.query_one("#command-palette", CommandPalette).set_choices([])

    def on_pinned_rich_log_pin_changed(self, event: PinnedRichLog.PinChanged) -> None:
        self.query_one("#scroll-indicator", Static).set_class(event.pinned, "visible")

    def on_app_blur(self, event: events.AppBlur) -> None:
        self._has_focus = False

    def on_app_focus(self, event: events.AppFocus) -> None:
        self._has_focus = True

    def action_show_help(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write("[bold yellow]Available Commands:[/bold yellow]")
        log.write("  /provider <name>       - Switch OpenRouter, Anthropic, OpenAI, DeepSeek, Google, Groq, or local")
        log.write("  /key <provider> <key>   - Save a private provider key; /key status or /key clear <provider>")
        log.write("  /model [MODEL_ID]      - Browse models (F4) or set one directly")
        log.write("  /models                - Search live OpenRouter catalog")
        log.write("  /tier <fast|standard|reasoning> - Force model tier")
        log.write("  /copy                   - Copy selected chat text or latest agent reply")
        log.write("  Ctrl+Shift+C            - Copy selected chat text or latest agent reply")
        log.write("  /safety <turbo|balanced|cautious|strict> - Set tool confirmation level")
        log.write("  /mode <coding|research|science|security|auto> - Set operational mode")
        log.write("  /thinking <none|low|medium|deep|auto> - Set reasoning budget")
        log.write("  /classifier <semif|sklearn|backend> [model/path] - Switch decision engine")
        log.write("  /workspace [path]      - Show or change working directory")
        log.write("  /sessions              - Browse saved sessions (F5); /sessions list prints IDs")
        log.write("  /session new [title] | load <id> | save")
        log.write("  /skills                - List installed skills")
        log.write("  /skill list            - Browse 22 built-in and custom skills")
        log.write("  /skill <name|off>      - Force a skill or restore automatic routing")
        log.write("  /output                - Open a selectable full-text view of the latest agent response")
        log.write("  /tool-output           - Open a selectable view of the latest tool result")
        log.write("  /usage                 - Show all session token types and reported cost")
        log.write("  /new | /reset          - Start a new session or reset current one")
        log.write("  /theme [name]          - Preview and save terminal theme (F2)")
        log.write("  /history               - Show recent prompts; Up/Down recalls prompts")
        log.write("  /export [markdown|json] - Save the session to output/sessions/")
        log.write("  F3                    - Review isolated changes")
        log.write("  F6                    - Show or hide telemetry")
        log.write("  /clear                 - Clear chat history")
        log.write("  /exit                  - Exit application")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "prompt-input":
            return
        if self.command_palette_visible:
            self.complete_selected_command()
            event.stop()
            return
        text = event.value.strip()
        if not text:
            return
        if self._review_task and text not in {"/diff", "/help", "/exit"}:
            self.query_one("#chat-log", RichLog).write(Text(
                "Review the isolated changes with F3 or /diff before starting another task.", style="yellow"))
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

        if self._busy and cmd in {"/key", "/provider", "/model", "/models", "/tier", "/mode", "/thinking", "/safety", "/classifier", "/sessions",
                                  "/workspace", "/session", "/skill", "/new", "/reset", "/export", "/isolation", "/swarm"}:
            log.write(Text("Wait for the current task before changing settings or exiting.", style="yellow"))
            return

        if cmd in ("/exit", "/quit"):
            self.action_quit()
        elif cmd == "/diff":
            self.action_review_diff()
        elif cmd in {"/isolation", "/swarm"}:
            if arg not in {"auto", "on", "off"}:
                log.write(Text(f"Usage: {cmd} auto|on|off", style="yellow"))
            else:
                setattr(self, "isolation_mode" if cmd == "/isolation" else "swarm_mode", arg)
                log.write(Text(f"{cmd[1:].title()} mode: {arg}", style="green"))
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
                log.write(Text("Use /key <provider> <key>, /key status, or /key clear <provider>.", style="yellow"))
                return
            if arg.lower() == "status":
                for name in PROVIDERS:
                    if name == "local":
                        continue
                    key = (self.api_key if name == self.provider_name else self.provider_keys.get(name))
                    masked = (key[:8] + "••••••••" + key[-4:] if key and len(key) > 12 else
                              "••••••••" if key else "none")
                    log.write(Text(f"{name}: {masked}" +
                        (f" ({self.key_source})" if name == self.provider_name else ""), style="cyan"))
                return
            if arg.lower().startswith("clear"):
                target = arg.split(maxsplit=1)[1].lower() if len(arg.split(maxsplit=1)) > 1 else self.provider_name
                try:
                    self.credentials.clear(target)
                    if target == "openrouter":
                        self.config.clear_key()
                except (OSError, ValueError) as exc:
                    log.write(Text(f"Could not clear saved key: {exc}", style="red"))
                    return
                self.provider_keys.pop(target, None)
                if target == self.provider_name:
                    self._connect_provider(target)
                log.write(Text(f"✓ Saved {target} key cleared.", style="green"))
                self._refresh_status()
                return
            key_parts = arg.split(maxsplit=1)
            target, key = (key_parts[0].lower(), key_parts[1]) if len(key_parts) == 2 else (self.provider_name, arg)
            try:
                self.credentials.set(target, key)
            except (OSError, ValueError) as exc:
                log.write(Text(f"Could not save API key: {exc}", style="red"))
                return
            self.provider_keys[target] = key
            if target == self.provider_name:
                self._connect_provider(target)
            if target == "openrouter" and self.agent.classifier_backend.name == "openrouter":
                self.agent.classifier_backend.api_key = key
            log.write(Text(f"✓ {target} API key securely saved for future sessions", style="green"))
            self._refresh_status()
        elif cmd == "/provider":
            target = arg.lower()
            if target not in PROVIDERS:
                log.write(Text("Choose: " + ", ".join(PROVIDERS), style="yellow"))
                return
            self._connect_provider(target)
            self._save_session()
            log.write(Text(f"✓ Provider: {target} · {self.llm_client.default_model}", style="green"))
        elif cmd == "/model":
            if not arg:
                self.action_choose_model()
                return
            self._set_model(arg)
        elif cmd == "/models":
            self.action_choose_model()
        elif cmd == "/tier":
            if arg in MODEL_TIERS:
                target_model = PROVIDER_TIERS.get(self.provider_name, MODEL_TIERS)[arg]
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
                    ("turbo", "Turbo · few interruptions; still block high-risk actions"),
                    ("balanced", "Balanced · resolve unclear targets and block destructive actions"),
                    ("cautious", "Cautious · ask when SemIf finds genuinely ambiguous intent"),
                    ("strict", "Strict · approve each shell, edit, write, test, and web tool call"),
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
            if self._last_agent_content:
                self.push_screen(OutputViewerModal(format_model_markdown(self._last_agent_content)))
            else:
                log.write(Text("No agent response yet.", style="yellow"))
        elif cmd == "/tool-output":
            if self._last_tool_output:
                self.push_screen(OutputViewerModal(self._last_tool_output, title="Latest Tool Output"))
            else:
                log.write(Text("No tool result yet.", style="yellow"))
        elif cmd == "/usage":
            self._show_usage()
        elif cmd == "/copy":
            self.action_copy_output()
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
            "usage": self._usage_snapshot(),
            "messages": self.agent.messages,
        }
        if export_format == "json":
            content = json.dumps(transcript, indent=2, ensure_ascii=False) + "\n"
        else:
            lines = [f"# {self.session.title}", "", f"Session: `{self.session.id}`",
                     f"Workspace: `{self.workspace_root}`", "",
                     f"Usage: {self._usage_summary()}", ""]
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

    def _usage_snapshot(self) -> dict:
        return {
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "input_tokens": self.prompt_tokens,
            "output_tokens": self.completion_tokens,
            "reasoning_tokens": self._reasoning_tokens,
            "cache_read_tokens": self._cached_tokens,
            "cache_write_tokens": self._cache_write_tokens,
            "cost_usd": self._reported_cost_usd if self._cost_reported else None,
            "cost_source": "provider-reported" if self._cost_reported else "unavailable",
        }

    def _usage_summary(self) -> str:
        usage = self._usage_snapshot()
        cost = f"${usage['cost_usd']:.6f} (reported)" if usage["cost_usd"] is not None else "not reported by provider"
        return (f"{usage['total_tokens']:,} total · input {usage['input_tokens']:,} · "
                f"output {usage['output_tokens']:,} · reasoning {usage['reasoning_tokens']:,} · "
                f"cache read {usage['cache_read_tokens']:,} · cache write {usage['cache_write_tokens']:,} · cost {cost}")

    def _show_usage(self) -> None:
        self.query_one("#chat-log", RichLog).write(Text("Session usage · " + self._usage_summary(), style="bold cyan"))

    def _should_isolate(self, task: str) -> bool:
        if self.swarm_mode == "on" or self.isolation_mode == "on":
            return True
        if self.isolation_mode == "off":
            return False
        level = self.agent.forced_thinking or self.agent.thinking_classifier.classify(task).level
        editing = bool(re.search(r"\b(edit|implement|fix|refactor|add|write|change|modify|build|upgrade|migrate|create)\b",
                                 task, flags=re.I))
        return editing and level in {ThinkingLevel.MEDIUM, ThinkingLevel.DEEP, ThinkingLevel.EXTREME}

    def _should_swarm(self, task: str) -> bool:
        if self.swarm_mode == "off":
            return False
        if self.swarm_mode == "on":
            return True
        level = self.agent.forced_thinking or self.agent.thinking_classifier.classify(task).level
        return level in {ThinkingLevel.DEEP, ThinkingLevel.EXTREME} and self._should_isolate(task)

    def _swarm_client(self) -> LLMClient:
        source = self.agent.llm_client
        return LLMClient(api_key=source.api_key, base_url=source.base_url,
                         default_model=self.agent.explicit_model or source.default_model,
                         force_mock=source.force_mock, provider=source.provider,
                         provider_keys=source.provider_keys,
                         backup_providers=source.backup_providers)

    def _worktree_started(self, task: WorktreeTask) -> None:
        self.query_one("#telemetry", ClassifierTelemetryWidget).update_telemetry(
            workspace_isolation=f"Isolated task-{task.id}")
        self.query_one("#chat-log", RichLog).write(Text(
            f"⚑ Working in isolated Git worktree task-{task.id}; main checkout stays untouched until review.",
            style="bold cyan"))

    def _swarm_status_changed(self, status: dict[str, str]) -> None:
        self.query_one("#telemetry", ClassifierTelemetryWidget).update_telemetry(swarm_status=status)

    def _swarm_completed(self, report, task_text: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        self.agent.messages.append({"role": "user", "content": task_text})
        summary = "\n".join(f"{result.role.value.title()} ({result.phase.value}): {result.summary}"
                            for result in report.results)
        self.agent.messages.append({"role": "assistant", "content": summary})
        for result in report.results:
            log.write(Text(f"{result.role.value.title()} · {result.phase.value}: "
                           f"{'✓' if result.success else '✗'} {result.summary[:400]}",
                           style="green" if result.success else "red"))
        log.write(Text("Swarm verification passed" if report.success else
                       "Swarm stopped without verified completion", style="bold green" if report.success else "bold yellow"))

    def _worktree_ready(self, manager: WorktreeManager, task: WorktreeTask, patch: str) -> None:
        self._review_manager, self._review_task, self._review_patch = manager, task, patch
        self.query_one("#chat-log", RichLog).write(Text(
            f"Isolated task-{task.id} produced changes. Review with F3 or /diff before applying.", style="bold cyan"))
        self.push_screen(DiffReviewModal(patch, task.id), callback=self._handle_review_decision)

    def _worktree_finished_without_review(self) -> None:
        self.query_one("#telemetry", ClassifierTelemetryWidget).update_telemetry(
            workspace_isolation="Direct workspace", swarm_status={})

    @work(thread=True)
    def execute_agent_task(self, task_text: str) -> None:
        """Worker thread executing the agent task and streaming events back to the UI."""
        manager: WorktreeManager | None = None
        isolated: WorktreeTask | None = None
        ready_for_review = False
        try:
            if self._should_isolate(task_text):
                try:
                    manager = WorktreeManager(self.workspace_root)
                    isolated = manager.create()
                except WorktreeError as exc:
                    if "not a git repository" not in str(exc).lower() or self.isolation_mode == "on" or self.swarm_mode == "on":
                        raise
                    manager = None
                if isolated:
                    self.agent.set_workspace(isolated.workspace)
                    self.call_from_thread(self._worktree_started, isolated)
            if self._should_swarm(task_text) and isolated:
                coordinator = SwarmCoordinator(DeveloperAgentWorker(llm_client_factory=self._swarm_client),
                    on_status=lambda status: self.call_from_thread(self._swarm_status_changed, dict(status)))
                report = coordinator.run(task_text, isolated.workspace, isolated=True)
                self.call_from_thread(self._swarm_completed, report, task_text)
                success = report.success
            else:
                success = False
                for event in self.agent.run_stream(task_text):
                    if event.event_type == "response":
                        success = bool(event.payload.get("success"))
                    self.call_from_thread(self._render_event, event)
            if manager and isolated:
                patch = manager.patch(isolated)
                if success and patch:
                    ready_for_review = True
                    self.call_from_thread(self._worktree_ready, manager, isolated, patch)
                else:
                    manager.abort(isolated)
                    self.call_from_thread(self._worktree_finished_without_review)
        except Exception as exc:
            if manager and isolated and not ready_for_review:
                try:
                    manager.abort(isolated)
                except WorktreeError:
                    pass
            if self.is_running:
                try:
                    self.call_from_thread(self._render_error, f"Agent error: {type(exc).__name__}: {exc}")
                except RuntimeError:
                    pass
        finally:
            if isolated:
                self.agent.set_workspace(self.workspace_root)
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
        if self._review_task:
            self._quit_when_finished = False
            return
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
                    "thinking": f"Thinking · step {p['step']} · {p.get('thinking_tokens', 0):,} budget tokens",
                    "generating": f"Generating response · step {p['step']}",
                    "tool_running": f"Running {p.get('tool', '')}",
                    "verifying": f"Verifying {p.get('tool', '')}",
                }.get(stage, stage)
                self._refresh_status()
        elif et == "context_status":
                telemetry.update_telemetry(context_used=p["used_tokens"], context_capacity=p["capacity"],
                    context_compacted=p["compacted_tokens"] if p["compacted"] else 0)
                self._refresh_status()
        elif et == "overseer":
                telemetry.update_telemetry(overseer_state=p["state"],
                    overseer_latency_ms=p.get("latency_ms", 0), overseer_tier=p.get("tier", "gate"))
                if p.get("directive"):
                    log.write(Text(f"⚠ Overseer: {p['state'].replace('_', ' ').title()} · strategy correction injected",
                                   style="bold yellow"))
        elif et == "provider_failover":
                log.write(Text(f"Provider failover: {p['from']} → {p['to']} ({p['model']})", style="bold yellow"))
                telemetry.update_telemetry(model=p["model"], selection="failover")
                self.provider_name = p["to"]
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
                catalog_model = next((item for item in self._model_catalog if item.id == p["model"]), None)
                self.agent.context_window_override = (catalog_model.context_length
                    if catalog_model and catalog_model.context_length else None)
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
        elif et == "llm_notice":
                log.write(Text(p["message"], style="yellow"))
        elif et == "storage_error":
                log.write(Text(p["error"], style="yellow"))
        elif et == "thought":
                self._last_agent_content = p["content"]
                log.write(Text(f"\nAgent · {p.get('model', 'model')}:", style="bold magenta"))
                log.write(Markdown(format_model_markdown(p["content"])))
        elif et == "tool_call":
                arguments = json.dumps(p["arguments"], ensure_ascii=False)
                preview = arguments[:400] + ("… Use /export for full tool arguments." if len(arguments) > 400 else "")
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
                elif p.get("chart"):
                    log.write(Text(output, style="bold cyan"))
                else:
                    visible_output = format_model_markdown(output, plain=True) if (
                        "$$" in output or r"\[" in output or r"\(" in output or r"\frac" in output
                    ) else output
                    if len(visible_output) > 800:
                        log.write(Text(visible_output[:800] + f"\n… {len(visible_output)-800} more characters. Use /tool-output to expand."))
                    else:
                        log.write(Text(visible_output))
                if p.get("time_ms", 0) >= 5000 and not self._has_focus and not self._bell_rung:
                    self.bell()
                    self._bell_rung = True
        elif et == "verification":
                log.write(Text(f"Verification: {p['status']} → {p['action']}", style="green" if p["status"] == "SUCCESS" else "red"))
        elif et == "response":
                if p.get("content") and p["content"] != self._last_agent_content:
                    self._last_agent_content = p["content"]
                    log.write(Text("\nAgent:", style="bold magenta"))
                    log.write(Markdown(format_model_markdown(p["content"])))
                elif not p.get("content") and not p["success"]:
                    log.write(Text("No final answer was produced; inspect the last tool result or retry.", style="yellow"))
                usage = p.get("usage") or {}
                self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                self._reasoning_tokens += int(usage.get("reasoning_tokens", 0) or 0)
                self._cache_write_tokens += int(usage.get("cache_write_tokens", 0) or 0)
                self._cached_tokens += int(p.get("cached_tokens", 0) or 0)
                self._reported_cost_usd += float(p.get("cost_usd", 0.0) or 0.0)
                self._cost_reported = self._cost_reported or bool(p.get("cost_reported", False))
                telemetry.update_telemetry(provider_cached_tokens=self._cached_tokens,
                                           provider_prompt_tokens=self.prompt_tokens)
                status = "✓ Task completed" if p["success"] else {
                    "response_length_limit": "Response stopped at the model output limit",
                    "step_limit": "Task reached the tool-step limit",
                    "verification_failed": "Task needs another verification pass",
                    "skill_verification_failed": "Task did not meet skill verification checks",
                    "provider_error": "Provider request failed",
                }.get(p.get("stop_reason"), "Task stopped before completion")
                cost_display = f"${self._reported_cost_usd:.6f}" if self._cost_reported else "not reported by provider"
                log.write(Text(f"Session tokens: {self.prompt_tokens + self.completion_tokens:,} total · "
                    f"input {self.prompt_tokens:,} · output {self.completion_tokens:,} · reasoning {self._reasoning_tokens:,} · "
                    f"cache read {self._cached_tokens:,} · cache write {self._cache_write_tokens:,} · cost {cost_display}",
                    style="dim cyan"))
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
