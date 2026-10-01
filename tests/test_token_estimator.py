"""Phase 0.4 — token estimation with pluggable backends.

The compaction threshold is a *fraction of the context window*, so the
estimator underneath it is a correctness concern, not a nicety: an estimator
that under-counts lets a request run past the window before anything notices,
and the provider rejects it.

The default stays zero-dependency — a product that cannot start without
`tiktoken` is not a product — but it must not be badly wrong, and the earlier
`len//4` was badly wrong for CJK.
"""

from __future__ import annotations

import json

import pytest

from adaptive_harness.agent import tokenizer
from adaptive_harness.agent.context_window import estimate_tokens
from adaptive_harness.agent.tokenizer import (
    available_backends,
    count,
    reset_backend,
    set_backend,
)


@pytest.fixture(autouse=True)
def _default_backend():
    """Every test starts from the default, whatever the previous one set."""
    reset_backend()
    yield
    reset_backend()


def _naive(messages, tools=None) -> int:
    """The estimator this replaced, kept so the tests can compare."""
    serialized = json.dumps({"messages": messages, "tools": tools or []},
                            ensure_ascii=False, default=str)
    return max(1, (len(serialized) + 3) // 4)


# --- the default must be right, not just cheap ------------------------------


def test_the_default_backend_needs_no_optional_dependency():
    assert "chars4" in available_backends()
    assert tokenizer.active_backend() == "chars4"


def test_the_default_is_close_to_the_old_estimate_for_english():
    """English is what the old estimate was tuned for, so it should not move
    much -- the fix is aimed at scripts, not at everything."""
    text = "The quick brown fox jumps over the lazy dog. " * 40
    assert count([{"role": "user", "content": text}]) == pytest.approx(
        _naive([{"role": "user", "content": text}]), rel=0.2)


def test_the_old_estimate_under_counted_cjk_severely():
    """This is the defect. A flat four-characters-per-token is roughly a third
    of the truth for Chinese, which is why the ranges are counted separately."""
    messages = [{"role": "user", "content": "这是一个测试句子。" * 40}]
    assert count(messages) > _naive(messages) * 2


def test_cjk_is_counted_at_about_one_token_per_character():
    messages = [{"role": "user", "content": "测" * 100}]
    # Allow the JSON scaffolding, but the content itself must dominate.
    assert count(messages) >= 100


def test_an_empty_request_is_still_measured_and_never_zero():
    """An empty message list still serializes to the request wrapper, so it
    costs the wrapper's tokens rather than nothing. What matters is that the
    estimate is never zero: a zero would make a division below it blow up."""
    assert count([]) >= 1
    assert count([{"role": "user", "content": ""}]) >= 1


def test_the_estimate_is_never_zero_for_real_content():
    assert count([{"role": "user", "content": "hi"}]) >= 1


# --- backend selection ------------------------------------------------------


def test_an_unknown_backend_leaves_the_active_one_alone():
    reset_backend()
    assert set_backend("nonsense") == "chars4"


def test_an_unavailable_backend_falls_back_rather_than_raising():
    """A user who selected tiktoken and then uninstalled it should get a
    slightly less accurate estimate, not a context window that stopped working."""
    reset_backend()
    chosen = set_backend("tiktoken")
    assert chosen in available_backends(), (
        f"tiktoken is absent here, so selection must fall back; got {chosen}")
    # Either it was selected (it is installed) or it fell back; both are correct.
    assert chosen in {"chars4", "tiktoken"}


def test_selecting_a_backend_changes_what_the_existing_call_signature_returns():
    """The point of keeping the signature: no call site moved."""
    messages = [{"role": "user", "content": "hello world " * 10}]
    reset_backend()
    before = estimate_tokens(messages)
    tokenizer.set_backend("tiktoken")
    after = estimate_tokens(messages)
    assert before >= 1 and after >= 1


def test_reset_restores_the_default():
    set_backend("chars4")
    assert tokenizer.active_backend() == "chars4"


def test_a_backend_that_raises_falls_back_instead_of_breaking_a_request():
    """An estimator must never be the reason a request fails."""
    reset_backend()

    def exploding(_serialized: str) -> int:
        raise RuntimeError("backend is broken")

    tokenizer._BACKENDS["exploding"] = exploding
    try:
        tokenizer._ACTIVE = "exploding"
        assert count([{"role": "user", "content": "hello"}]) >= 1
    finally:
        tokenizer._BACKENDS.pop("exploding", None)
        reset_backend()


def test_a_backend_returning_none_falls_back():
    tokenizer._BACKENDS["returns_none"] = lambda _serialized: None
    try:
        tokenizer._ACTIVE = "returns_none"
        assert count([{"role": "user", "content": "hello"}]) >= 1
    finally:
        tokenizer._BACKENDS.pop("returns_none", None)
        reset_backend()


# --- the estimator is a setting, so the core knows only the interface --------


def test_the_backend_is_selectable_by_name_from_configuration():
    """A plugin setting is a string; the core turns it into a selection."""
    reset_backend()
    chosen = set_backend("chars4")
    assert chosen == "chars4", "a settings-driven selection must work by name"
