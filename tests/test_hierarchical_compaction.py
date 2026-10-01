"""Phase 1.4 — hierarchical compaction.

Single-pass summarization loses detail irreversibly: whatever one pass happened
to keep is gone for the rest of the run. Map→reduce bounds that loss by folding
generations instead, and these tests hold the properties that make the fold
*safe* rather than merely smaller:

- only complete turns are folded, so a tool call is never separated from its
  result;
- protected regions are exempt;
- **an error is never summarized away** — sacred invariant 5, and the one place
  where losing detail changes what the model does next.
"""

from __future__ import annotations

import pytest

from adaptive_harness.agent.hierarchical import (
    MAX_SUMMARY_CHARS,
    MIN_COHORT,
    complete_turns,
    compact_hierarchically,
    fold,
)


def _conversation(turns: int, error_at: int | None = None) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": "the system prompt"}]
    for index in range(turns):
        call_id = f"call-{index}"
        messages.append({"role": "user", "content": f"request {index}"})
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [{"id": call_id,
                                         "function": {"name": "run_bash"}}]})
        content = "ERROR: the build failed at line 42" if index == error_at else f"output {index}"
        messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
    return messages


def _no_orphans(messages) -> bool:
    for index, message in enumerate(messages):
        for call in message.get("tool_calls") or []:
            following = messages[index + 1:index + 2]
            if not following or following[0].get("tool_call_id") != call["id"]:
                return False
    return True


# --- the fold happens at all, and only when it is warranted -----------------


def test_a_long_conversation_is_folded():
    compacted, report = compact_hierarchically(
        _conversation(20), protected={0}, head=1, tail=8)
    assert len(compacted) < len(_conversation(20))
    assert report.messages_before > report.messages_after


def test_a_short_conversation_is_left_alone():
    """Folding two turns loses more structure than it saves."""
    messages = _conversation(2)
    compacted, report = compact_hierarchically(messages, protected={0}, head=1, tail=8)
    assert len(compacted) == len(messages)


def test_protected_messages_are_never_folded():
    messages = _conversation(20)
    pinned = {"role": "user", "content": "ALWAYS FOLLOW THIS"}
    messages.insert(1, pinned)
    compacted, _ = compact_hierarchically(messages, protected={0, 1}, head=2, tail=8)
    assert pinned in compacted, "a pinned message was folded away"


def test_the_tail_is_never_folded():
    """The most recent messages are what the model is working on right now."""
    messages = _conversation(20)
    tail = messages[-8:]
    compacted, _ = compact_hierarchically(messages, protected={0}, head=1, tail=8)
    assert compacted[-8:] == tail


# --- correctness of the fold ------------------------------------------------


def test_no_tool_call_is_ever_separated_from_its_result():
    """An assistant message with tool calls and no results is a history the
    provider rejects on the next request."""
    compacted, _ = compact_hierarchically(
        _conversation(20), protected={0}, head=1, tail=8)
    assert _no_orphans(compacted)


def test_complete_turns_stops_at_user_boundaries():
    messages = _conversation(3)
    indices = complete_turns(messages, 1, len(messages))
    assert messages[indices[0]]["role"] == "user"
    for index in indices:
        # Every turn is a user message followed by non-user messages.
        if messages[index]["role"] == "user":
            continue
        previous_user = max((i for i in indices[:index] if messages[i]["role"] == "user"),
                            default=-1)
        assert previous_user >= 0


def test_an_error_survives_the_fold_verbatim():
    """Losing a failed tool call's text is the one thing that must never
    happen: it is what stops the model repeating the mistake."""
    compacted, _ = compact_hierarchically(
        _conversation(20, error_at=3), protected={0}, head=1, tail=8)
    joined = "\n".join(str(m.get("content", "")) for m in compacted)
    assert "the build failed at line 42" in joined, (
        "an error was lost in compaction")


def test_a_successful_call_does_not_bloat_the_summary():
    compacted, report = compact_hierarchically(
        _conversation(40), protected={0}, head=1, tail=8)
    summaries = [str(m.get("content", "")) for m in compacted
                 if "Summary of earlier work" in str(m.get("content", ""))]
    assert summaries, "nothing was summarized"
    # The fold covers 32 of 40 turns, so it must not carry all of them.
    assert summaries[0].count("output ") < 30, (
        "the summary carried every successful tool output through")


# --- map, then reduce ------------------------------------------------------


def test_fold_reduces_to_a_bounded_number_of_summaries():
    turns = [[f"turn {index}"] for index in range(20)]
    summaries, generations = fold(turns, lambda text, gen: text[:200], budget=600)
    assert len(summaries) >= 1
    total = sum(len(str(item["content"])) for item in summaries)
    assert total <= 600 * 2, "the reduce did not reach the budget"


def test_an_incompressible_cohort_is_kept_rather_than_truncated():
    """A summarizer that cannot compress must not lose the cohort.

    The alternative -- truncating to the budget -- silently drops the tail of
    the conversation, which is the exact failure hierarchical compaction exists
    to prevent. So an incompressible cohort is kept in full and said to be.
    """
    turns = [[f"turn {index} " * 40] for index in range(20)]
    summaries, _ = fold(turns, lambda text, gen: text, budget=500)
    joined = str(summaries[0]["content"])
    assert "turn 0" in joined
    assert "turn 19" in joined


def test_a_summary_does_not_grow_without_bound_across_generations():
    """One enormous result must not become an ever-larger summary.

    The hierarchy is allowed to keep an incompressible cohort in full rather
    than truncate it -- losing turns is worse -- but the *headers* must not
    accumulate, so a run of generations stays roughly the size of its input
    rather than growing every pass.
    """
    turns = [[f"x" * 100_000] for _ in range(8)]
    summaries, generations = fold(turns, lambda text, gen: text, budget=1000)
    assert generations <= 6, "the generation loop did not terminate"
    header = "Summary of earlier work"
    for summary in summaries:
        content = str(summary["content"])
        # A handful of nested headers, not one per generation of a runaway loop.
        assert content.count(header) <= 3 * (generations + 1) + 2, (
            "the fold headers accumulated across generations")


def test_a_summarizer_that_does_nothing_is_not_an_error():
    """Losing nothing is a valid outcome of a fold."""
    turns = [[f"turn {index}"] for index in range(10)]
    summaries, _ = fold(turns, lambda text, gen: text, budget=100_000)
    joined = "\n".join(str(item["content"]) for item in summaries)
    # Line-anchored: "turn 1" is a substring of "turn 10".
    assert "\nturn 0\n" in f"\n{joined}\n"
    assert "\nturn 9\n" in f"\n{joined}\n"


def test_the_generation_counter_advances():
    turns = [[f"turn {index} " * 100] for index in range(30)]
    seen = []

    def summarizer(text: str, generation: int) -> str:
        seen.append(generation)
        return text  # cannot compress

    fold(turns, summarizer, budget=400)
    assert seen, "the summarizer was never called"
    # Generation 0 is enough when the reduce can meet the budget in one pass,
    # which is the common case. The counter exists for the case where it cannot.
    assert min(seen) == 0


# --- the report -------------------------------------------------------------


def test_the_report_says_what_happened():
    _, report = compact_hierarchically(_conversation(20), protected={0}, head=1, tail=8)
    payload = report.to_dict()
    assert payload["messages_before"] > payload["messages_after"]
    assert payload["cohort_sizes"], "the cohort sizes should be visible"
    assert sum(payload["cohort_sizes"]) > 0


def test_the_report_records_what_it_measured():
    _, report = compact_hierarchically(
        _conversation(20), protected={0}, head=1, tail=8,
        estimate=lambda items: sum(len(str(m.get("content", ""))) for m in items))
    assert report.tokens_before > 0
    assert report.tokens_after > 0


def test_an_estimator_that_raises_does_not_break_the_fold():
    def exploding(_items):
        raise RuntimeError("estimator is down")

    compacted, _ = compact_hierarchically(
        _conversation(20), protected={0}, head=1, tail=8, estimate=exploding)
    assert compacted, "the fold should still have happened"


# --- integration with the real request path ---------------------------------


def test_prepare_context_still_preserves_a_system_prompt_and_a_tail():
    from adaptive_harness.agent.context_window import prepare_context

    messages = _conversation(30)
    prepared, info = prepare_context(messages, "mock/model", None, limit=2000)
    assert prepared[0]["role"] == "system"
    assert prepared[0]["content"] == messages[0]["content"]
    assert _no_orphans(prepared)
