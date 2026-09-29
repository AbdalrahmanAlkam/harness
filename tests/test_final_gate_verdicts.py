"""Phase 4 — the Quality Controller's verdicts.

The gate exists because an LLM assertion alone is not evidence. These tests hold
that line, and they are mostly about what must *not* pass:

- a summary that claims work no tool call supports is never accepted;
- a summary that contradicts the recorded evidence is never accepted;
- the gate never rewrites the model's text — it reports, and the model authors;
- a classifier can add findings but never remove a deterministic one.

Requirement extraction is deliberately syntactic and local, so these also pin
that it does not need a model to run.
"""

from __future__ import annotations

import pytest

from adaptive_harness.agent.final_gate import (
    ACCEPT,
    MISSING,
    PARTIALLY_SATISFIED,
    REJECT,
    REVISE,
    UNSUPPORTED_CLAIM,
    Adjudication,
    Evidence,
    EvidenceLedger,
    Requirement,
    RequirementSet,
    adjudicate,
    evaluate,
    extract_requirements,
    find_contradictions,
    revision_instruction,
)


def _ledger(*records) -> EvidenceLedger:
    ledger = EvidenceLedger()
    for record in records:
        ledger.record(record)
    return ledger


# --- 4.1 requirement extraction --------------------------------------------


def test_a_request_with_two_obligations_yields_two_requirements():
    """Treating "add retry and update the docs" as one thing is how a run
    reports itself done having done half of it."""
    requirements = extract_requirements("add retry logic and update the docs")
    assert [item.text for item in requirements.requirements] == [
        "add retry logic", "update the docs"]


def test_requirements_are_numbered_and_persisted():
    """A resumed session must not lose the ground truth."""
    requirements = extract_requirements("fix the parser. then add a test")
    payload = requirements.to_dict()
    restored = RequirementSet.from_dict(payload)
    assert [item.id for item in restored.requirements] == ["R1", "R2"]
    assert restored.to_dict() == payload


def test_an_empty_request_yields_no_requirements():
    assert len(extract_requirements("")) == 0
    assert len(extract_requirements("   ")) == 0


def test_a_single_obligation_is_not_split_into_fragments():
    requirements = extract_requirements("refactor the parser to be iterative")
    assert len(requirements) == 1


def test_each_requirement_carries_words_it_can_be_evidenced_by():
    requirements = extract_requirements("add `retry()` to the http client")
    keywords = requirements.requirements[0].keywords
    assert keywords, "a requirement with no keywords can never be evidenced"
    assert any("retry" in word for word in keywords), (
        "a quoted term is almost always the thing being asked for")


def test_extraction_is_local_and_free():
    """Sacred invariant 1: no foundation-model tokens to decide what enters."""
    import inspect

    source = inspect.getsource(extract_requirements)
    for banned in ("llm_client", "complete(", "openai", "anthropic"):
        assert banned not in source


# --- 4.2 the evidence ledger is observed, not narrated ----------------------


def test_the_ledger_records_what_actually_happened():
    ledger = _ledger(
        Evidence(tool="write_file", target="src/retry.py", success=True, digest="written"),
        Evidence(tool="run_pytest", target="tests/", success=False, digest="2 failed"),
    )
    assert len(ledger) == 2
    assert len(ledger.successes()) == 1
    assert len(ledger.failures()) == 1
    assert ledger.by_tool("run_pytest")[0].success is False


def test_the_ledger_serialises_for_a_transcript():
    """A transcript that carries the ledger is itself evidence."""
    ledger = _ledger(Evidence(tool="read_file", target="a.py", success=True, digest="x"))
    payload = ledger.to_dict()
    assert payload["count"] == 1
    assert payload["records"][0]["tool"] == "read_file"


# --- 4.3 adjudication against the evidence ---------------------------------


def test_a_satisfied_requirement_needs_a_successful_matching_call():
    ledger = _ledger(Evidence(tool="write_file", target="src/retry.py",
                              success=True, digest="def retry(): ..."))
    item = adjudicate(Requirement("R1", "add retry logic", ("retry",)), ledger, "")
    assert item.status == "satisfied"
    assert item.evidence


def test_a_failing_call_never_satisfies_a_requirement():
    ledger = _ledger(Evidence(tool="write_file", target="src/retry.py",
                              success=False, digest="permission denied"))
    item = adjudicate(Requirement("R1", "add retry logic", ("retry",)), ledger, "")
    assert item.status == PARTIALLY_SATISFIED, (
        "a failed call is not evidence that the work happened")


def test_an_claim_with_no_supporting_call_is_an_unsupported_claim():
    """The hallucination class. It must never be rounded up to a pass."""
    ledger = _ledger(Evidence(tool="list_directory", target="src", success=True,
                              digest="a.py b.py"))
    item = adjudicate(Requirement("R1", "update the docs", ("docs",)), ledger,
                      "I updated the docs.")
    assert item.status == UNSUPPORTED_CLAIM
    assert "asserts" in item.reason


def test_a_requirement_nobody_mentions_is_missing():
    ledger = _ledger(Evidence(tool="list_directory", target="src", success=True))
    item = adjudicate(Requirement("R1", "add caching", ("caching",)), ledger, "")
    assert item.status == MISSING


def test_a_requirement_with_no_keywords_is_reported_unevidenced():
    """The honest answer when there is nothing to match on is 'I cannot show
    this', not 'assume it happened'."""
    item = adjudicate(Requirement("R1", "do the thing", ()), _ledger(), "")
    assert item.status == MISSING
    assert "no distinctive terms" in item.reason


# --- contradictions run before any model ------------------------------------


def test_claiming_all_tests_pass_with_a_recorded_failure_is_a_contradiction():
    ledger = _ledger(Evidence(tool="run_pytest", target="tests/", success=False,
                              digest="3 failed"))
    found = find_contradictions("All tests pass.", ledger)
    assert found, "a summary contradicting the ledger must be caught"
    assert "all tests pass" in found[0]


def test_a_claim_with_no_matching_evidence_is_not_a_contradiction():
    """The gate must not cry wolf: with no recorded failure, the claim stands."""
    ledger = _ledger(Evidence(tool="run_pytest", target="tests/", success=True,
                              digest="583 passed"))
    assert find_contradictions("All tests pass.", ledger) == []


def test_each_contradiction_is_reported_once():
    ledger = _ledger(
        Evidence(tool="run_pytest", target="tests/", success=False, digest="3 failed"),
        Evidence(tool="run_pytest", target="tests/", success=False, digest="2 failed"),
    )
    found = find_contradictions("All tests pass.", ledger)
    assert len(found) == len(set(found))


# --- 4.3 the verdict --------------------------------------------------------


def test_a_fully_evidenced_summary_is_accepted():
    requirements = extract_requirements("add retry logic")
    ledger = _ledger(Evidence(tool="write_file", target="src/retry.py", success=True,
                              digest="def retry(): ..."))
    verdict = evaluate(requirements, ledger, "Added retry logic in src/retry.py.")
    assert verdict.verdict == ACCEPT
    assert verdict.ok


def test_a_hallucination_never_passes():
    """The single most important property of this module."""
    requirements = extract_requirements("update the docs")
    ledger = _ledger(Evidence(tool="read_file", target="README.md", success=True,
                              digest="hello"))
    verdict = evaluate(requirements, ledger, "I updated the docs.")
    assert verdict.verdict != ACCEPT
    assert any(item.status == UNSUPPORTED_CLAIM for item in verdict.adjudications)


def test_a_contradiction_never_passes():
    requirements = extract_requirements("run the tests")
    ledger = _ledger(Evidence(tool="run_pytest", target="tests/", success=False,
                              digest="3 failed"))
    verdict = evaluate(requirements, ledger, "All tests pass.")
    assert verdict.verdict != ACCEPT
    assert verdict.contradictions


def test_partial_evidence_asks_for_a_revision_not_a_pass():
    requirements = extract_requirements("add caching")
    ledger = _ledger(Evidence(tool="edit_file", target="src/cache.py", success=False,
                              digest="pattern not found"))
    verdict = evaluate(requirements, ledger, "Added caching.")
    assert verdict.verdict == REVISE


def test_an_empty_requirement_set_with_no_evidence_is_accepted():
    """Nothing was asked and nothing was done; that is not a failure."""
    assert evaluate(RequirementSet(), EvidenceLedger(), "Hello.").verdict == ACCEPT


# --- 4.4 what a non-accept verdict does -------------------------------------


def test_a_revision_names_the_specific_deficiency():
    requirements = extract_requirements("add retry logic and update the docs")
    ledger = _ledger(Evidence(tool="write_file", target="src/retry.py", success=True,
                              digest="def retry(): ..."))
    verdict = evaluate(requirements, ledger, "Added retry logic and updated the docs.")
    instruction = revision_instruction(verdict)
    assert "R2" in instruction, "the revision must name which requirement failed"
    assert "docs" in instruction
    assert "retry logic" not in instruction.split("R2")[0], (
        "already-evidenced work should not be asked for again")


def test_a_revision_tells_the_model_it_may_report_failure():
    """A model that cannot finish needs an honest way out, or it will claim
    success rather than admit the gap."""
    verdict = evaluate(extract_requirements("do the impossible"),
                       EvidenceLedger(), "done")
    assert "say which one and why" in revision_instruction(verdict)


def test_the_gate_never_rewrites_the_model_text():
    """It reports; the model authors. A gate that edited the answer would be
    publishing a claim the model never made."""
    summary = "I updated the docs."
    verdict = evaluate(extract_requirements("update the docs"),
                       EvidenceLedger(), summary)
    assert summary not in verdict.to_dict()["reason"]
    assert not hasattr(verdict, "rewritten")
    # The verdict carries the model's own words only inside adjudications it
    # built from the requirements, never a corrected summary.
    assert "corrected" not in verdict.to_dict()


# --- 4.6 a classifier may add findings, never remove them -------------------


def test_a_classifier_cannot_clear_a_deterministic_finding():
    class SilentClassifier:
        def adjudications(self):
            return []

    requirements = extract_requirements("update the docs")
    ledger = _ledger(Evidence(tool="read_file", target="README.md", success=True))
    baseline = evaluate(requirements, ledger, "I updated the docs.")
    with_classifier = evaluate(requirements, ledger, "I updated the docs.",
                               relevance=SilentClassifier())
    assert with_classifier.verdict == baseline.verdict, (
        "a classifier must not be able to turn a failure into a pass")


def test_a_classifier_that_raises_cannot_break_the_gate():
    class Exploding:
        def adjudications(self):
            raise RuntimeError("classifier is down")

    verdict = evaluate(extract_requirements("add caching"),
                       _ledger(Evidence(tool="write_file", target="src/cache.py",
                                        success=True, digest="cache")),
                       "Added caching.", relevance=Exploding())
    assert verdict.verdict == ACCEPT, (
        "a broken classifier must not turn a good run into a failure")


def test_a_classifier_may_add_a_finding():
    class StrictClassifier:
        def adjudications(self):
            return [Adjudication("R9", "an obligation the classifier noticed",
                                  MISSING, (), "no evidence")]

    verdict = evaluate(extract_requirements("add caching"),
                       _ledger(Evidence(tool="write_file", target="src/cache.py",
                                        success=True, digest="cache")),
                       "Added caching.", relevance=StrictClassifier())
    assert any(item.requirement_id == "R9" for item in verdict.adjudications)


# --- the verdict is serialisable, because a transcript carries it ------------


def test_a_verdict_serialises_for_the_transcript_and_the_exit_code():
    verdict = evaluate(extract_requirements("add caching"), EvidenceLedger(), "done")
    payload = verdict.to_dict()
    assert payload["verdict"] in {ACCEPT, REVISE, REJECT}
    assert "adjudications" in payload and isinstance(payload["adjudications"], list)
