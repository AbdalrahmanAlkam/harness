"""Phase 4.5 — calibration of the claim gate.

A gate that cries wolf gets switched off, which is worse than having none. So
`unsupported_claim` has to have a *measured* false-positive rate, not an
assumed one, and the test below builds a labelled set and measures it.

The labelled set is generated from realistic (summary, ledger) pairs, with the
honest cases written first and the hallucinated ones labelled as such. A gate
that cannot tell them apart is not a gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_harness.agent.final_gate import (
    ACCEPT,
    MISSING,
    REVISE,
    UNSUPPORTED_CLAIM,
    Evidence,
    EvidenceLedger,
    RequirementSet,
    Requirement,
    adjudicate,
    evaluate,
    extract_requirements,
)

#: What a correct report looks like: the ledger really does evidence it.
HONEST = [
    ("add retry logic",
     "Added retry logic to the http client.",
     [("write_file", "src/http.py", True, "def retry(fn, n=3): ...")]),
    ("fix the off-by-one in the parser",
     "Fixed the off-by-one in src/parse.py.",
     [("edit_file", "src/parse.py", True, "index += 1")]),
    ("run the test suite",
     "Ran the tests: 583 passed.",
     [("run_pytest", "tests/", True, "583 passed")]),
    ("document the plugin contract",
     "Documented the plugin manifest in docs/plugins.md.",
     [("write_file", "docs/plugins.md", True, "# Plugins ...")]),
    ("remove the unused import",
     "Removed the unused import from src/main.py.",
     [("edit_file", "src/main.py", True, "import os")]),
]

#: What a hallucinated report looks like: the ledger never mentions the work.
HALLUCINATED = [
    ("update the docs",
     "I updated the docs.",
     [("read_file", "README.md", True, "# readme")]),
    ("add caching to the client",
     "Added caching.",
     [("list_directory", "src", True, "a.py b.py")]),
    ("run the test suite",
     "All tests pass.",
     [("list_directory", ".", True, "src tests")]),
    ("write a migration script",
     "Wrote the migration.",
     [("read_file", "models.py", True, "class M: ...")]),
    ("fix the flaky test",
     "Fixed the flaky test.",
     [("search_files", "test_", True, "3 matches")]),
]


def _ledger_for(records) -> EvidenceLedger:
    ledger = EvidenceLedger()
    for tool, target, success, digest in records:
        ledger.record(Evidence(tool=tool, target=target, success=success, digest=digest))
    return ledger


def _check(requirement_text: str, summary: str, records) -> str:
    requirement = Requirement(id="R1", text=requirement_text,
                             keywords=extract_requirements(requirement_text).requirements[0].keywords)
    return adjudicate(requirement, _ledger_for(records), summary).status


# --- the false-positive rate, measured --------------------------------------


def test_the_gate_separates_honest_reports_from_hallucinations():
    """The measurement the plan asks for: a labelled set, scored.

    An honest report backed by real evidence must be `satisfied`; a claim with
    nothing behind it must not be. If these two overlap, the gate is noise.
    """
    honest_correct = 0
    for text, summary, records in HONEST:
        if _check(text, summary, records) == "satisfied":
            honest_correct += 1
    false_positives = sum(1 for text, summary, records in HONEST
                          if _check(text, summary, records) == UNSUPPORTED_CLAIM)

    assert honest_correct >= len(HONEST) - 1, (
        f"only {honest_correct}/{len(HONEST)} honest reports were recognised")
    assert false_positives / len(HONEST) < 0.2, (
        f"false-positive rate {false_positives / len(HONEST):.0%} is too high; "
        f"a gate that cries wolf gets switched off")


def test_every_hallucinated_claim_is_caught():
    caught = 0
    for text, summary, records in HALLUCINATED:
        status = _check(text, summary, records)
        # Every status except "satisfied" is a failure to substantiate. That
        # includes `partially_satisfied`: tools touched the area and none of
        # them performed the work, which is not evidence the work was done.
        if status != "satisfied":
            caught += 1
    assert caught == len(HALLUCINATED), (
        f"only {caught}/{len(HALLUCINATED)} hallucinated claims were caught")


def test_the_two_classes_do_not_overlap():
    """Separation, stated as a property rather than two separate counts."""
    honest = {_check(*item) for item in HONEST}
    hallucinated = {_check(*item) for item in HALLUCINATED}
    assert "satisfied" in honest, "an honest report was not recognised"
    assert "satisfied" not in hallucinated, (
        f"a hallucinated claim was accepted: {hallucinated}")


# --- the gate must not be trigger-happy on a real transcript ---------------


def test_a_realistic_multi_step_run_is_accepted(tmp_path: Path):
    """A run that did the work, described accurately, must pass.

    This is the false-positive case that matters: a real transcript is long,
    with failures and detours in it, and none of that should read as a
    hallucination.
    """
    (tmp_path / "src").mkdir()
    ledger = EvidenceLedger()
    ledger.record(Evidence(tool="read_file", target="src/http.py", success=True,
                           digest="def request(): ..."))
    ledger.record(Evidence(tool="edit_file", target="src/http.py", success=False,
                           digest="pattern not found"))
    ledger.record(Evidence(tool="read_file", target="src/client.py", success=True,
                           digest="class Client: ..."))
    ledger.record(Evidence(tool="edit_file", target="src/client.py", success=True,
                           digest="def retry(fn, n=3): ..."))
    ledger.record(Evidence(tool="run_pytest", target="tests/", success=True,
                           digest="583 passed"))

    verdict = evaluate(extract_requirements("add retry logic to the http client"),
                       ledger, "Added retry logic in src/client.py and the tests pass.")
    assert verdict.verdict == ACCEPT, (
        f"an honest run was failed by the gate: {verdict.to_dict()}")


def test_a_run_with_a_failure_is_not_treated_as_a_hallucination():
    """A failed edit followed by a successful one is normal work, not a lie."""
    # Realistic: the first attempt missed its anchor, the second landed. The
    # digests are what a tool actually reports, not placeholders.
    ledger = _ledger_for([
        ("read_file", "src/client.py", True, "class Client: ..."),
        ("edit_file", "src/client.py", False,
         "Edit failed: the text to replace was not found in the file."),
        ("read_file", "src/client.py", True, "def request(self): ..."),
        # A real edit_file returns the content it wrote, so the evidence
        # actually contains the thing the summary claims.
        ("edit_file", "src/client.py", True,
         "Wrote src/client.py:\n\ndef retry(fn, n=3):\n    for _ in range(n):\n        try: return fn()\n        except Transient: pass\n"),
    ])
    verdict = evaluate(extract_requirements("add retry logic to src/client.py"),
                       ledger, "Added retry logic in src/client.py.")
    assert not any(item.status == UNSUPPORTED_CLAIM for item in verdict.adjudications), (
        "a normal retry was mistaken for a fabricated claim")


# --- the gate's own guarantees still hold under measurement ------------------


def test_measurement_does_not_weaken_the_never_fail_open_rule(tmp_path: Path):
    """Whatever the false-positive rate, an unsupported claim never passes."""
    ledger = _ledger_for([("read_file", "README.md", True, "# readme")])
    verdict = evaluate(extract_requirements("update the docs"),
                       ledger, "I updated the docs.")
    assert verdict.verdict != ACCEPT


def test_contradictions_are_still_caught_on_a_realistic_ledger(tmp_path: Path):
    ledger = _ledger_for([("run_pytest", "tests/", True, "580 passed"),
                          ("run_pytest", "tests/", False, "3 failed")])
    verdict = evaluate(extract_requirements("run the test suite"), ledger,
                       "All tests pass.")
    assert verdict.contradictions, (
        "a recorded failure did not contradict the claim of success")
    assert verdict.verdict != ACCEPT
