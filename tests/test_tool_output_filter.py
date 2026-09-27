"""Tests for the classifier-gated tool-output filter.

The safety properties matter more than the saving, so they are tested first: an
error is never compressed, a broken classifier never costs the agent evidence,
and an evicted token is reported as gone rather than silently returning the
wrong text.
"""

from __future__ import annotations

import pytest

from adaptive_harness.agent.tool_filter import ToolOutputArchive, ToolOutputFilter
from adaptive_harness.classifiers.engine import OUTPUT_VALUE_LABELS, SklearnBackend
from adaptive_harness.tools.recall import ReadFullOutputTool

NOISE = "\n".join(f"test_module_{i} PASSED" for i in range(200))
SIGNAL = "Traceback (most recent call last):\n  NameError: name 'foo' is not defined\n" * 40


class StubClassification:
    def __init__(self, label, **probabilities):
        self.label = label
        self.probabilities = probabilities


class StubClassifier:
    def __init__(self, label="noise", **probabilities):
        self.label, self.probabilities = label, probabilities
        self.calls = 0

    def classify(self, text, labels):
        self.calls += 1
        assert list(labels) == OUTPUT_VALUE_LABELS
        return StubClassification(self.label, **self.probabilities)


class ExplodingClassifier:
    def classify(self, text, labels):
        raise RuntimeError("backend is down")


def _filter(classifier=None, summarize=None, **kwargs):
    return ToolOutputFilter(
        classifier=classifier or StubClassifier("noise", noise=0.97, signal=0.03),
        summarize=summarize or (lambda tool, text: "SUMMARY: 200 tests passed."),
        **kwargs)


# -- archive ---------------------------------------------------------------


def test_archive_round_trips_and_is_bounded():
    archive = ToolOutputArchive(limit=2)
    first = archive.store("run_pytest", "alpha")
    second = archive.store("run_bash", "beta")
    assert archive.fetch(first) == "tool=run_pytest\nalpha"
    assert archive.fetch(second) == "tool=run_bash\nbeta"
    third = archive.store("read_file", "gamma")
    # The oldest entry is evicted, and it is reported as gone.
    assert archive.fetch(first) is None
    assert archive.fetch(third) == "tool=read_file\ngamma"
    assert len(archive) == 2


def test_archive_tokens_are_unique_and_monotonic():
    archive = ToolOutputArchive()
    tokens = [archive.store("t", str(i)) for i in range(5)]
    assert len(set(tokens)) == 5
    assert tokens == ["OUTPUT-1", "OUTPUT-2", "OUTPUT-3", "OUTPUT-4", "OUTPUT-5"]


def test_archive_rejects_a_useless_limit():
    with pytest.raises(ValueError):
        ToolOutputArchive(limit=0)


# -- the gate --------------------------------------------------------------


def test_noise_is_compressed_and_left_a_token():
    filter_ = _filter()
    outcome = filter_.process("run_pytest", NOISE)
    assert outcome.filtered
    assert outcome.token == "OUTPUT-1"
    assert "SUMMARY" in outcome.content
    assert "OUTPUT-1" in outcome.content
    assert outcome.saved_chars > 0
    # The notice is terse: it must not restate the summary.
    notice = outcome.content.split("\n\n", 1)[1]
    assert notice.count("\n") == 0
    assert len(notice) < 220


def test_signal_is_passed_through_untouched():
    filter_ = _filter(StubClassifier("signal", noise=0.02, signal=0.98))
    outcome = filter_.process("run_bash", SIGNAL)
    assert not outcome.filtered
    assert outcome.content == SIGNAL
    assert outcome.token is None


def test_errors_are_never_filtered_even_when_judged_noise():
    """A failure is the evidence the agent needs; compressing it loses the run."""
    filter_ = _filter()
    outcome = filter_.process("run_pytest", NOISE, success=False)
    assert not outcome.filtered
    assert outcome.content == NOISE
    # The classifier is not even consulted, so a misfire cannot cost an error.
    assert filter_.classifier.calls == 0


def test_short_output_is_never_filtered():
    filter_ = _filter()
    short = "ok\n" * 10
    outcome = filter_.process("run_bash", short)
    assert not outcome.filtered
    assert filter_.classifier.calls == 0


def test_a_broken_classifier_costs_no_evidence():
    """Fail open: an unavailable backend must never delete tool output."""
    outcome = _filter(ExplodingClassifier()).process("run_pytest", NOISE)
    assert not outcome.filtered
    assert outcome.content == NOISE
    assert "classifier unavailable" in outcome.reason


def test_a_failing_summariser_falls_back_to_the_raw_output():
    def boom(tool, text):
        raise TimeoutError("secondary model did not answer")

    outcome = _filter(summarize=boom).process("run_pytest", NOISE)
    assert not outcome.filtered
    assert outcome.content == NOISE
    assert "summariser failed" in outcome.reason


def test_a_summary_that_saves_nothing_is_rejected():
    outcome = _filter(summarize=lambda tool, text: text).process("run_pytest", NOISE)
    assert not outcome.filtered
    assert "summary not shorter" in outcome.reason


def test_a_weak_noise_verdict_does_not_filter():
    """Below the threshold the gate must not act, even on the 'noise' label."""
    filter_ = _filter(StubClassifier("noise", noise=0.40, signal=0.60),
                      noise_threshold=0.5)
    assert not filter_.process("run_pytest", NOISE).filtered
    # ...and the same verdict does filter once the threshold allows it.
    strict = _filter(StubClassifier("noise", noise=0.40, signal=0.60),
                     noise_threshold=0.3)
    assert strict.process("run_pytest", NOISE).filtered


def test_a_summary_longer_than_the_cap_is_truncated():
    # 3000 chars is shorter than NOISE (4489) but far over the 200-char cap.
    filter_ = _filter(summarize=lambda tool, text: "x " * 1500, max_summary_chars=200)
    outcome = filter_.process("run_pytest", NOISE)
    assert outcome.filtered
    summary = outcome.content.split("\n\n", 1)[0]
    assert summary == "x " * 100


def test_a_summary_longer_than_the_original_is_rejected_not_truncated():
    """Truncation must not disguise an echo as a summary."""
    filter_ = _filter(summarize=lambda tool, text: text + " more", max_summary_chars=200)
    outcome = filter_.process("run_pytest", NOISE)
    assert not outcome.filtered
    assert "summary not shorter" in outcome.reason


def test_a_notice_that_eats_the_whole_saving_is_rejected():
    """A tiny result is not worth replacing with a summary plus a notice."""
    filter_ = _filter(min_chars=50, max_summary_chars=10_000)
    outcome = filter_.process("run_pytest", "p" * 60)
    assert not outcome.filtered


def test_the_original_text_is_recoverable_from_the_token():
    filter_ = _filter()
    outcome = filter_.process("run_pytest", NOISE)
    recovered = filter_.archive.fetch(outcome.token)
    assert recovered == f"tool=run_pytest\n{NOISE}"


# -- the tool --------------------------------------------------------------


def test_read_full_output_returns_the_archived_text():
    filter_ = _filter()
    outcome = filter_.process("run_pytest", NOISE)
    tool = ReadFullOutputTool(filter_.archive)
    result = tool.execute(token=outcome.token)
    assert result.success
    assert NOISE in result.output
    assert result.metadata["truncated"] is False


def test_read_full_output_reports_an_evicted_token_honestly():
    archive = ToolOutputArchive(limit=1)
    stale = archive.store("run_pytest", "old")
    archive.store("run_bash", "new")
    result = ReadFullOutputTool(archive).execute(token=stale)
    assert not result.success
    assert "expired" in result.error
    assert "re-run" in result.error


def test_read_full_output_validates_its_input():
    tool = ReadFullOutputTool(ToolOutputArchive())
    assert not tool.execute(token="").success
    assert not tool.execute(token="   ").success
    assert "required" in tool.execute(token="").error


def test_read_full_output_truncates_a_huge_archive_entry():
    archive = ToolOutputArchive()
    token = archive.store("run_pytest", "y" * 40_000)
    result = ReadFullOutputTool(archive, limit=1000).execute(token=token)
    assert result.success
    assert result.metadata["truncated"] is True
    assert result.metadata["total_chars"] == 40_000 + len("tool=run_pytest\n")
    # The cap is announced, not hidden: the transcript promised the full output.
    assert "this read is capped" in result.output
    assert "characters omitted" in result.output
    assert len(result.output) < 1300


def test_an_uncapped_read_does_not_claim_to_be_capped():
    archive = ToolOutputArchive()
    token = archive.store("run_bash", "z" * 500)
    result = ReadFullOutputTool(archive, limit=20_000).execute(token=token)
    assert result.metadata["truncated"] is False
    assert "this read is capped" not in result.output


# -- the real backend ------------------------------------------------------


def test_the_real_backend_can_judge_the_new_labels():
    """SklearnBackend used to raise for this label set, which meant no filtering."""
    backend = SklearnBackend()
    verdict = backend.classify(SIGNAL, OUTPUT_VALUE_LABELS)
    assert set(verdict.probabilities) == set(OUTPUT_VALUE_LABELS)
    assert abs(sum(verdict.probabilities.values()) - 1.0) < 1e-9


def test_the_real_backend_separates_chatter_from_evidence():
    backend = SklearnBackend()
    assert backend.classify(SIGNAL, OUTPUT_VALUE_LABELS).label == "signal"
    assert backend.classify(NOISE, OUTPUT_VALUE_LABELS).label == "noise"
