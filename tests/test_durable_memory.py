"""Phase 1.5 — durable memory in three tiers.

The tiers differ in who wrote them, and the agent-authored tier is the one that
needs care: a model that can silently change what it will be told next turn can,
over a few turns, set its own instructions. So a proposal never takes effect
silently — it is scored, shown as a one-line diff, and accepted by a human.

The tests are mostly about refusals, because the failure mode is a secret
written to disk and replayed into every later session.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from adaptive_harness.agent.memory import (
    MAX_MEMORY_CHARS,
    Memory,
    MemoryStore,
    is_rememberable,
    score_generality,
)


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path)


# --- a proposal is not a memory --------------------------------------------


def test_a_proposed_memory_does_not_take_effect_until_accepted(tmp_path: Path):
    store = _store(tmp_path)
    store.propose("We always use tabs in Makefiles.", task="fix the makefile")
    assert len(store) == 0, "a proposal applied itself"
    assert store.recall("fix the makefile") == []


def test_accepting_makes_it_recallable(tmp_path: Path):
    store = _store(tmp_path)
    proposal = store.propose("We always use tabs in Makefiles.", task="make")
    assert store.accept(proposal.memory.id) is not None
    assert len(store) == 1
    assert store.recall("make") == ["We always use tabs in Makefiles."]


def test_declining_drops_it_forever(tmp_path: Path):
    store = _store(tmp_path)
    proposal = store.propose("Prefer ruff over flake8.")
    store.decline(proposal.memory.id)
    assert store.proposals == {}
    assert len(store) == 0


def test_accepting_an_unknown_id_is_not_a_crash(tmp_path: Path):
    assert _store(tmp_path).accept("nope") is None


def test_a_memory_can_be_revoked(tmp_path: Path):
    store = _store(tmp_path)
    proposal = store.propose("Prefer ruff over flake8.")
    store.accept(proposal.memory.id)
    assert store.forget(proposal.memory.id)
    assert len(store) == 0
    assert not store.forget(proposal.memory.id)


def test_clear_empties_everything(tmp_path: Path):
    store = _store(tmp_path)
    proposal = store.propose("Prefer ruff over flake8.")
    store.accept(proposal.memory.id)
    store.propose("something else")
    assert store.clear() == 1
    assert len(store) == 0 and store.proposals == {}


# --- a secret is never written to disk -------------------------------------


@pytest.mark.parametrize("text", [
    "The API key is sk-or-v1-abc",            # prose, with a space
    "my api_key is hunter2",                  # config, with an underscore
    "use API-KEY abcdef",
    "a password for the database",
    "the bearer token expires",
    "sk-or-v1-1234567890abcdef",              # bare credential
    "sk-ant-api03-abcdefghijklmnop",
    "gsk_abcdefghij1234567890",
    "AIzaSyA1234567890abcdefghijklmnop",
    "ghp_abcdefghijklmnopqrstuvwxyz012345",
    "xoxb-123456789012-abcdefghijkl",
])
def test_a_credential_is_refused(text: str):
    ok, reason = is_rememberable(text)
    assert not ok, f"a credential-shaped memory was accepted: {text!r}"
    assert reason


@pytest.mark.parametrize("text", [
    "We use tabs in Makefiles",
    "Prefer ruff over flake8",
    "The test suite lives in tests/",
    "Never commit the output directory",
])
def test_an_ordinary_project_fact_is_accepted(text: str):
    ok, _ = is_rememberable(text)
    assert ok, f"a legitimate memory was refused: {text!r}"


def test_a_credential_cannot_even_be_proposed(tmp_path: Path):
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.propose("the api key is sk-or-v1-abc123")
    assert store.proposals == {}


def test_an_absolute_path_is_refused_as_machine_specific(tmp_path: Path):
    ok, reason = is_rememberable("the build lives in /home/someone/project")
    assert not ok
    assert "machine-specific" in reason


def test_an_overlong_note_is_refused(tmp_path: Path):
    ok, reason = is_rememberable("x" * (MAX_MEMORY_CHARS + 10))
    assert not ok and "longer than" in reason


def test_an_empty_memory_is_refused():
    assert not is_rememberable("   ")[0]


# --- scope: a task observation is not a project fact -----------------------


def test_a_rule_sounding_memory_scores_as_durable():
    generality, reason = score_generality(
        "We always use tabs in Makefiles, because spaces break the target.")
    assert generality > 0.5
    assert "rule" in reason


def test_a_time_specific_observation_scores_as_temporary():
    generality, reason = score_generality(
        "For now, retry is disabled in this branch.")
    assert generality < 0.5
    assert "time-specific" in reason


def test_a_low_score_becomes_a_task_memory(tmp_path: Path):
    store = _store(tmp_path)
    proposal = store.propose("For this run, the build is broken in a specific way.")
    assert proposal.memory.scope == "task"
    assert "task-specific" in proposal.preview


def test_a_task_memory_is_not_replayed_for_unrelated_work(tmp_path: Path):
    store = _store(tmp_path)
    proposal = store.propose("The parser migration failed on the vendored module.")
    store.accept(proposal.memory.id)
    assert store.recall("unrelated database cleanup") == []
    assert store.recall("the parser migration") != []


def test_project_memories_are_always_recalled(tmp_path: Path):
    store = _store(tmp_path)
    proposal = store.propose("We always use tabs in Makefiles.")
    store.accept(proposal.memory.id)
    assert store.recall("anything at all") != []


# --- the operator has to be able to see it ---------------------------------


def test_the_preview_carries_the_decision_not_just_the_data(tmp_path: Path):
    store = _store(tmp_path)
    proposal = store.propose("We always use tabs in Makefiles.")
    assert proposal.preview.startswith("+")
    assert "project" in proposal.preview
    assert "durable" in proposal.preview


def test_describe_lists_memories_and_pending_proposals(tmp_path: Path):
    store = _store(tmp_path)
    accepted = store.propose("We always use tabs in Makefiles.")
    store.accept(accepted.memory.id)
    store.propose("Something to decide.")
    described = store.describe()
    assert "in effect" in described
    assert "tabs" in described
    assert "awaiting your answer" in described


def test_the_context_block_is_empty_when_there_is_nothing_to_say(tmp_path: Path):
    assert _store(tmp_path).as_context("anything") == ""


def test_the_context_block_names_itself(tmp_path: Path):
    store = _store(tmp_path)
    proposal = store.propose("We always use tabs in Makefiles.")
    store.accept(proposal.memory.id)
    context = store.as_context("make")
    assert "remembered for this project" in context
    assert "tabs" in context


# --- durability and isolation -----------------------------------------------


def test_accepted_memories_survive_a_restart(tmp_path: Path):
    first = _store(tmp_path)
    proposal = first.propose("We always use tabs in Makefiles.")
    first.accept(proposal.memory.id)

    second = _store(tmp_path)
    assert len(second) == 1
    assert second.recall("make") == ["We always use tabs in Makefiles."]


def test_the_file_is_private(tmp_path: Path):
    """A memory file holds whatever the model learned about the project."""
    store = _store(tmp_path)
    proposal = store.propose("We always use tabs in Makefiles.")
    store.accept(proposal.memory.id)
    assert store.path.stat().st_mode & 0o077 == 0, "the memory file is group- or world-readable"


def test_a_pending_proposal_is_not_written_to_disk(tmp_path: Path):
    """Only accepted memories are persisted; a proposal is a question."""
    store = _store(tmp_path)
    store.propose("We always use tabs in Makefiles.")
    assert not store.path.exists() or "tabs" not in store.path.read_text(encoding="utf-8")


def test_a_corrupt_memory_file_is_reported_not_silently_empty(tmp_path: Path):
    (tmp_path / "memory.json").write_text("{not json", encoding="utf-8")
    store = _store(tmp_path)
    assert store.load_error, "a corrupt memory file must say so"
    assert "WARNING" in store.describe()


def test_concurrent_proposals_do_not_collide(tmp_path: Path):
    store = _store(tmp_path)
    errors: list[Exception] = []

    def propose(index: int) -> None:
        try:
            store.propose(f"Fact number {index} about the project conventions.")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=propose, args=(index,)) for index in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    ids = {proposal.memory.id for proposal in store.proposals.values()}
    assert len(ids) == 20, "two proposals shared an id"
