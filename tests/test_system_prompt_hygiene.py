"""The composed system prompt must be sent once, and echoed once.

Two separate guarantees are pinned here, because they fail independently and
the symptom is easy to misread as one bug:

* On the wire, the model must receive exactly one ``system`` message per
  request. The agent rebuilds ``messages[0]`` before every request; if that ever
  became an append, every turn would resend the whole prompt and pay for it in
  tokens on every call.
* In the log, the prompt is rendered for auditing, and a byte-identical reprint
  on every turn buried the conversation while reading as though the prompt were
  being injected over and over. It is shown when it first appears, when it
  changes, and when a new session starts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_harness.agent.agent import AgentEvent, DeveloperAgent
from adaptive_harness.classifiers.domain_classifier import DomainMode
from adaptive_harness.llm.client import LLMClient
from adaptive_harness.tui.app import AdaptiveHarnessApp
from adaptive_harness.tui.widgets import PinnedRichLog


class RecordingClient(LLMClient):
    """Captures the exact message list sent on every request."""

    def __init__(self) -> None:
        super().__init__(force_mock=True)
        self.sends: list[list[dict]] = []

    def complete(self, **kwargs):
        self.sends.append([dict(message) for message in (kwargs.get("messages") or [])])
        return super().complete(**kwargs)


def _agent(tmp_path: Path) -> tuple[DeveloperAgent, RecordingClient]:
    client = RecordingClient()
    agent = DeveloperAgent(llm_client=client, workspace_root=str(tmp_path),
                           preferences_dir=tmp_path / "prefs")
    return agent, client


# -- what the model receives ------------------------------------------------


def test_the_model_receives_exactly_one_system_message_per_request(tmp_path: Path):
    agent, client = _agent(tmp_path)
    for _ in range(3):
        list(agent.run_stream("say hello"))

    assert len(client.sends) == 3
    for messages in client.sends:
        systems = [m for m in messages if m.get("role") == "system"]
        assert len(systems) == 1, f"expected 1 system message, got {len(systems)}"
        # It stays at the head, where an API expects it.
        assert messages[0]["role"] == "system"
    # The transcript grows by the conversation, never by a repeated prompt.
    assert [len(m) for m in client.sends] == sorted(len(m) for m in client.sends)


def test_the_rebuilt_prompt_replaces_rather_than_accumulates(tmp_path: Path):
    """Two runs must not leave two system messages behind in the agent state."""
    agent, client = _agent(tmp_path)
    list(agent.run_stream("first"))
    first = client.sends[-1][0]["content"]
    list(agent.run_stream("second"))

    assert sum(1 for m in agent.messages if m.get("role") == "system") == 1
    # An unchanged environment rebuilds an identical prompt, so the second
    # request carries the same text rather than a concatenation.
    assert client.sends[-1][0]["content"] == first


def test_switching_mode_changes_the_prompt_that_is_sent(tmp_path: Path):
    """The prompt is rebuilt per request on purpose; prove it is not cached.

    A coding-cued task is used so the unforced run genuinely resolves to coding,
    which makes the forced switch observable rather than a no-op.
    """
    agent, client = _agent(tmp_path)
    list(agent.run_stream("fix the failing test in app.py and refactor the code"))
    before = client.sends[-1][0]["content"]
    assert "coding" in before

    agent.forced_mode = DomainMode.RESEARCH
    list(agent.run_stream("fix the failing test in app.py and refactor the code"))
    after = client.sends[-1][0]["content"]
    assert after != before
    assert "research" in after


# -- what the log shows -----------------------------------------------------


def _system_prompt_events(content: str) -> list[AgentEvent]:
    return [AgentEvent("system_prompt", {"content": content, "ingested": {}})]


def _log_text(log: PinnedRichLog) -> str:
    return "\n".join(strip.text for strip in log.lines)


@pytest.mark.anyio
async def test_an_identical_system_prompt_is_echoed_only_once(tmp_path: Path):
    app = AdaptiveHarnessApp(db_path=tmp_path / "echo.db", workspace_root=str(tmp_path),
                             config_dir=tmp_path / "cfg")
    async with app.run_test(size=(100, 30)) as pilot:
        log = app.query_one("#chat-log", PinnedRichLog)
        log.clear()

        for _ in range(4):
            app._render_event(AgentEvent("system_prompt",
                                    {"content": "PROMPT-BODY", "ingested": {}}))
            await pilot.pause()

        text = _log_text(log)
        assert text.count("PROMPT-BODY") == 1
        # The summary still reports every turn, so nothing looks silently dropped.
        assert text.count("Ingested system prompt") == 4
        assert "unchanged since the previous prompt" in text


@pytest.mark.anyio
async def test_a_changed_system_prompt_is_echoed_in_full(tmp_path: Path):
    """Suppression must not hide a prompt that genuinely changed."""
    app = AdaptiveHarnessApp(db_path=tmp_path / "change.db", workspace_root=str(tmp_path),
                             config_dir=tmp_path / "cfg")
    async with app.run_test(size=(100, 30)) as pilot:
        log = app.query_one("#chat-log", PinnedRichLog)
        log.clear()

        app._render_event(AgentEvent("system_prompt", {"content": "BODY-ONE", "ingested": {}}))
        app._render_event(AgentEvent("system_prompt", {"content": "BODY-ONE", "ingested": {}}))
        app._render_event(AgentEvent("system_prompt", {"content": "BODY-TWO", "ingested": {}}))
        await pilot.pause()

        text = _log_text(log)
        assert text.count("BODY-ONE") == 1
        assert text.count("BODY-TWO") == 1


@pytest.mark.anyio
async def test_a_new_session_audits_the_prompt_again(tmp_path: Path):
    app = AdaptiveHarnessApp(db_path=tmp_path / "new.db", workspace_root=str(tmp_path),
                             config_dir=tmp_path / "cfg")
    async with app.run_test(size=(100, 30)) as pilot:
        log = app.query_one("#chat-log", PinnedRichLog)
        app._render_event(AgentEvent("system_prompt", {"content": "BODY", "ingested": {}}))
        await pilot.pause()
        assert _log_text(log).count("BODY") == 1

        app._reset_session(new=True, title="next")
        log = app.query_one("#chat-log", PinnedRichLog)
        app._render_event(AgentEvent("system_prompt", {"content": "BODY", "ingested": {}}))
        await pilot.pause()
        assert _log_text(log).count("BODY") == 1, "a new session must show it again"
