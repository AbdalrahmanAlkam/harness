"""Open-ended topics: the Director formulates claims, the kernel decides them.

The contract these tests protect is narrow and important: a model may state a
claim, but it may never write the script that judges it. Every proposition that
reaches the gate carries a kernel-generated script, so the verdict comes from an
exit code. A model proposing a false identity gets it refuted, not believed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from adaptive_harness.research import ResearchSwarm, SwarmConfig
from adaptive_harness.research.claim import Verdict
from adaptive_harness.research.formulate import (Formulation, formulate, propositions_from_payload,
                                                render_spec_section)

GOOD_FORMULATION = {
    "propositions": [
        {"id": "P1", "name": "Geometric pairing",
         "claim": "(x - y) * (x + y) == x**2 - y**2",
         "symbols": ["x", "y"],
         "hypotheses": ["x and y are real or complex"],
         "statement": "A difference of squares factors as a product of a sum and a difference.",
         "proof_sketch": "Expanding the product cancels the cross terms.",
         "consequence": "Justifies a factorisation-based check for the identity."},
        {"id": "P2", "name": "Doubling",
         "claim": "2 * n * (n + 1) == n * (n + 1) + n * (n + 1)",
         "symbols": ["n"],
         "hypotheses": ["n is an integer"],
         "statement": "Doubling a product distributes.",
         "proof_sketch": "Both sides are the same sum written twice.",
         "consequence": "A sanity check on the kernel's simplifier."},
    ]
}

# A claim that is false. The kernel must refute it, not adopt it.
FALSE_FORMULATION = {
    "propositions": [
        {"id": "P1", "name": "Wrong claim",
         "claim": "(x + y)**2 == x**2 + y**2",
         "symbols": ["x", "y"],
         "hypotheses": ["x and y are real"],
         "statement": "The square of a sum omits the cross term (false).",
         "proof_sketch": "This is the error the harness must catch.",
         "consequence": "A refuted claim is still a result."},
    ]
}


class FormulatingClient:
    """A client that replies with a fixed formulation payload, then goes quiet."""

    is_mock = True
    default_model = "formulating"
    provider = "scripted"
    api_key = None
    base_url = None
    provider_keys: dict = {}
    backup_providers: tuple = ()

    def __init__(self, payload: dict[str, Any] | None):
        self.payload = payload
        self.turn = 0
        self.asked: list[str] = []

    def complete(self, messages, tools=None, model=None, **kwargs):
        from adaptive_harness.llm.mock_client import LLMResponse
        self.asked.append(str(messages[-1].get("content", "")))
        self.turn += 1
        if self.payload is None:
            return LLMResponse(content="I decline to formulate anything.",
                                model=model or self.default_model)
        return LLMResponse(content=json.dumps(self.payload),
                            model=model or self.default_model)


def _swarm(tmp_path: Path, payload: dict | None, **kwargs) -> ResearchSwarm:
    return ResearchSwarm(
        "an uncovered topic about weighted harmonic means of nested loops",
        root=tmp_path,
        config=SwarmConfig(max_cycles=kwargs.pop("max_cycles", 2),
                           llm_client_factory=lambda: FormulatingClient(payload),
                           worker_max_steps=2, **kwargs))


# -- the kernel still owns the deciding script -----------------------------

def test_only_the_claim_comes_from_the_model():
    props = propositions_from_payload(GOOD_FORMULATION)
    assert [item.prop_id for item in props] == ["PROP-D01", "PROP-D02"]
    for proposition in props:
        # The script is generated, so it carries the kernel's own contract and
        # cannot be model-authored code.
        assert "Exit-code contract" in proposition.script
        assert "def hold(" in proposition.script and "def refute(" in proposition.script
        assert proposition.rationale.startswith("Formulated by the Theoretical Lead")


def test_unusable_claims_are_dropped_rather_than_admitted():
    payload = {"propositions": [
        {"claim": "x**2 == 2.0", "symbols": ["x"]},            # float literal
        {"claim": "N(pi) == 3", "symbols": []},                 # approximating
        {"claim": "the sky is blue", "symbols": []},             # not an identity
        {"claim": "x == 1 == 2", "symbols": ["x"]},             # malformed
        {"claim": "x**2 == x**2", "symbols": ["x"]},            # the one good claim
    ]}
    props = propositions_from_payload(payload)
    assert len(props) == 1
    # The claim survives into the kernel's display form, translated to Typst math.
    assert props[0].display == ("x^2 = x^2",)


def test_a_reply_that_ignores_the_contract_yields_nothing():
    for reply in ("", "I think the topic is interesting.", "{not json}", "[]"):
        assert propositions_from_payload({}) == ()
    formulation = formulate("topic", client=FormulatingClient(None))
    assert formulation.requested is True
    assert formulation.empty


def test_no_model_means_not_requested_rather_than_declined():
    formulation = formulate("topic")
    assert formulation.requested is False
    assert "not tested" in formulation.notes


# -- end to end ------------------------------------------------------------

def test_uncovered_topic_is_formulated_and_decided(tmp_path: Path):
    swarm = _swarm(tmp_path, GOOD_FORMULATION)
    assert swarm.plan.strategy == "dynamic-formulation"
    assert len(swarm.plan.propositions) == 2
    # The library really did not cover it, which is why formulation ran.
    assert "No derivation strategy matched" in swarm.plan.notes

    outcome = swarm.run()
    assert swarm.claims.headline is Verdict.PROVEN, swarm.claims.summary()
    assert all(item.exit_code == 0 for item in swarm.claims.adjudications)
    # The kernel wrote the deciding scripts.
    scripts = sorted(path.name for path in swarm.workspace.proof_dir.glob("*.py"))
    assert scripts == ["prop-d01.py", "prop-d02.py"]
    assert Path(outcome.pdf).is_file()


def test_a_formulated_but_false_claim_is_refused_not_believed(tmp_path: Path):
    """The decisive property: a model's false claim is refuted by exit code."""
    swarm = _swarm(tmp_path, FALSE_FORMULATION)
    assert swarm.plan.strategy == "dynamic-formulation"
    outcome = swarm.run()
    assert swarm.claims.headline is Verdict.DISPROVEN, swarm.claims.summary()
    witness = swarm.claims.disproven[0].finding
    assert "counterexample at" in witness
    # A refutation is a settled result, so the run completes.
    assert outcome.solved, outcome.render()
    assert "DISPROVEN" in Path(swarm.workspace.paper_typ).read_text()


def test_the_formulation_is_recorded_in_the_objective_spec(tmp_path: Path):
    swarm = _swarm(tmp_path, GOOD_FORMULATION)
    swarm.run()
    spec = Path(swarm.workspace.objective_spec).read_text()
    assert "formulated by the Director" in spec
    assert "PROP-D01" in spec
    assert "A difference of squares factors" in spec


def test_a_declined_formulation_is_reported_not_hidden(tmp_path: Path):
    swarm = _swarm(tmp_path, None)
    outcome = swarm.run()
    assert not outcome.solved
    assert swarm.claims.headline is Verdict.UNTESTED
    satisfied, detail, _ = swarm._evaluate_claim()
    assert not satisfied
    # The paper states the negative result rather than implying coverage.
    assert "No proposition was derived" in Path(swarm.workspace.paper_typ).read_text()


def test_formulation_can_be_disabled(tmp_path: Path):
    swarm = _swarm(tmp_path, GOOD_FORMULATION, formulate_unknown=False)
    assert swarm.plan.strategy == "none"
    assert swarm.plan.propositions == ()


def test_covered_topics_are_not_re_formulated(tmp_path: Path):
    """A topic the library covers must not be second-guessed by the model."""
    client_holder: list[FormulatingClient] = []

    def factory():
        client = FormulatingClient(GOOD_FORMULATION)
        client_holder.append(client)
        return client

    swarm = ResearchSwarm("pareto heavy tailed delay critical index", root=tmp_path,
                          config=SwarmConfig(max_cycles=1, llm_client_factory=factory))
    assert swarm.plan.strategy != "dynamic-formulation"
    swarm.run()
    assert not client_holder or not client_holder[0].asked, "the model was consulted needlessly"


def test_spec_section_renders_an_empty_formulation_honestly():
    text = render_spec_section(Formulation((), "no checkable identity was available", True))
    assert "no checkable identity" in text
    assert "## Formulated Claims" in text


# ---------------------------------------------------------------------------
# The exact decider is fed model-authored text, so it must be total
# ---------------------------------------------------------------------------

def _run_decider(expression: str) -> int:
    import subprocess
    import sys
    import tempfile

    from adaptive_harness.research.claim import exact_decider
    source = exact_decider(expression, statement=expression)
    assert source, f"expected a decider for {expression!r}"
    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / "decider.py"
        script.write_text(source, encoding="utf-8")
        return subprocess.run([sys.executable, str(script)], capture_output=True,
                              text=True, timeout=120).returncode


def test_the_decider_never_raises_on_anything_a_model_might_write():
    """A parser crash on model output is a single point of failure for the run."""
    from adaptive_harness.research.claim import exact_decider
    hostile = [
        "", "   ", "==", "== 4", "4 ==", "a == b == c", "x == 1 and y == 2",
        "not math at all", "import os", "x* == y", "a<b == c>d", "() == ()",
        '""" == """', "None == None", "True == True", "foo(bar) == baz(qux)",
        "|N++(v)| >= |N+(v)|", "\\frac{1}{2} == 0.5", "A[0][1] == 1",
        "sum(1 for k in range(4)) == 6", "v1.out_deg == v2.out_deg",
        "\n==\n", "0 == 0.0", "1/0 == 1", "lambda: 1 == 2", "x == y",
        "f(x) == f(x) for all x", "set() == {1,2}",
    ]
    for expression in hostile:
        result = exact_decider(expression, statement=expression)
        assert isinstance(result, str), expression


def test_an_unevaluable_expression_is_never_reported_as_a_counterexample():
    """Undefined is not false. A refutation here would be a false accusation."""
    from adaptive_harness.research.claim import exact_decider
    for expression in ("1/0 == 1", "lambda: 1 == 2", "a == b == c",
                       "sum(1 for k in range(4)) == 6", "0 == 0.0"):
        assert exact_decider(expression, statement=expression) == "", expression


def test_a_genuine_identity_is_still_decided_by_the_kernel():
    from adaptive_harness.research.claim import EXIT_COUNTEREXAMPLE, EXIT_HOLDS
    assert _run_decider("2 + 2 == 4") == EXIT_HOLDS
    assert _run_decider("(x + y)**2 == x**2 + 2*x*y + y**2") == EXIT_HOLDS
    assert _run_decider("sqrt(2)**2 == 2") == EXIT_HOLDS
    # A closed false claim is a refutation, and that one is real.
    assert _run_decider("2 + 2 == 5") == EXIT_COUNTEREXAMPLE
    # A false identity with free symbols is caught by the exact grid.
    assert _run_decider("(x + y)**2 == x**2 + y**2") == EXIT_COUNTEREXAMPLE
