"""The Quality Controller.

> The classifier is always fed the model's final summary and checks whether what
> the model did fulfils what the user asked for.

This is the part of the harness that closes the loop between what was *asked*
and what was *claimed*, using the same principle the `SkillVerifier` already
uses: an LLM assertion alone never satisfies a check. The evidence is observed
during the run, not narrated afterwards, because asking the model what it did
is precisely the hallucination surface this exists to remove.

The gate classifies each requirement against the evidence ledger and emits one
of three verdicts:

- **accept** — every requirement is supported by recorded evidence.
- **revise** — something is asserted that the ledger does not support. The
  specific deficiencies go back to the model as a targeted instruction, so it
  repairs rather than restarts.
- **reject** — the run did not do the work.

Two properties are non-negotiable and are enforced here rather than trusted:

1. **It never fails open on a hallucination-class verdict.** An unsupported
   claim is not a pass with a note attached.
2. **It never rewrites the model's text.** It reports; the model authors.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# --- requirement extraction -------------------------------------------------

#: Connectives that mark a second obligation in one sentence. A request like
#: "add retry logic and update the docs" is two requirements, and treating it as
#: one is how a run reports itself done having done half of it.
_CONJUNCTIONS = (
    r"and then", r"and also", r"and", r"then", r"also", r"plus",
    r"as well as", r";", r",",
)

#: Verbs that usually begin an obligation, used when a fragment is too short to
#: be a clause of its own.
_OBLIGATION_VERBS = (
    "add", "update", "fix", "remove", "write", "create", "implement", "refactor",
    "document", "test", "migrate", "rename", "delete", "configure", "optimize",
    "optimise", "support", "handle", "validate", "verify", "benchmark",
)

#: Words that mark a claim the run is asserting rather than a thing to do. Used
#: to tell a requirement ("add retry") from a statement about the result.
_RESULT_MARKERS = re.compile(
    r"\b(?:should|does|works|passes|passing|now|already|fixed|done|complete)\b", re.I)


@dataclass(frozen=True)
class Requirement:
    """One atomic, checkable obligation extracted from the user's request."""

    id: str
    text: str
    #: Words a matching piece of evidence should contain. Extracted from the
    #: obligation itself rather than invented, so a requirement always has some
    #: way of being evidenced.
    keywords: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "text": self.text, "keywords": list(self.keywords)}


@dataclass
class RequirementSet:
    """The ground truth for one turn. Persisted so a resumed session keeps it."""

    requirements: List[Requirement] = field(default_factory=list)
    source: str = ""

    def __len__(self) -> int:
        return len(self.requirements)

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source,
                "requirements": [item.to_dict() for item in self.requirements]}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RequirementSet":
        return cls(
            requirements=[Requirement(id=str(item.get("id", "")),
                                       text=str(item.get("text", "")),
                                       keywords=tuple(item.get("keywords", ())))
                          for item in payload.get("requirements", [])],
            source=str(payload.get("source", "")))


#: Words too common to be evidence of anything.
_STOPWORDS = frozenset("""
a an the and or but if then to of in on at for with without from by as is are was were be been
being do does did done this that these those it its their there here all any some each every
no not so such than too very can will just only also into over under about after before
""".split())


def extract_requirements(request: str) -> RequirementSet:
    """Split a request into atomic, checkable obligations.

    Deliberately syntactic and local. A classifier or a model would do this
    better, and would also cost a foundation-model token on every single turn to
    do it — which sacred invariant 1 forbids. The split is conservative: it
    errs toward more requirements, because a spurious requirement costs a
    revision prompt and a missing one costs an unfulfilled ask.
    """
    text = " ".join(str(request or "").split())
    if not text:
        return RequirementSet(source=text)

    # Split on sentence boundaries first, then on conjunctions inside each.
    fragments: List[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        pieces = re.split(rf"\s+(?:{'|'.join(_CONJUNCTIONS)})\s+", sentence)
        fragments.extend(piece.strip(" ,;.") for piece in pieces if piece and piece.strip(" ,;."))

    requirements: List[Requirement] = []
    for fragment in fragments:
        if len(fragment) < 4:
            continue
        requirements.append(Requirement(
            id=f"R{len(requirements) + 1}",
            text=fragment,
            keywords=_keywords(fragment),
        ))
    return RequirementSet(requirements=requirements, source=text)


def _keywords(fragment: str) -> tuple[str, ...]:
    """The words a piece of evidence would plausibly contain for this obligation.

    Derived from the obligation itself, so there is always *some* way for it to
    be evidenced. Content words and anything in backticks or quotes count; the
    first few are the distinctive ones.
    """
    distinctive = [word.lower() for word in re.findall(r"[A-Za-z_][A-Za-z0-9_./-]{2,}", fragment)
                   if word.lower() not in _STOPWORDS and not word.isdigit()]
    # Quoted or backticked terms are almost always the thing being asked for.
    for quoted in re.findall(r"[`\"']([^`\"']{2,40})[`\"']", fragment):
        distinctive.insert(0, quoted.lower())
    seen: List[str] = []
    for word in distinctive:
        if word not in seen:
            seen.append(word)
    return tuple(seen[:8])


# --- the evidence ledger ----------------------------------------------------

@dataclass
class Evidence:
    """One observed tool result, recorded during the run rather than recalled."""

    tool: str
    target: str
    success: bool
    #: A short digest of what came back, for matching against a claim.
    digest: str = ""
    #: Filesystem change observed around the call, when the tool could write.
    workspace_delta: int = 0
    step: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "target": self.target, "success": self.success,
                "digest": self.digest[:400], "workspace_delta": self.workspace_delta,
                "step": self.step}


class EvidenceLedger:
    """Everything the run actually did, in order.

    Built by observation, not by asking. A model asked to narrate its own work
    will narrate it correctly most of the time, which is the problem: the rate
    at which it is wrong is exactly the rate the gate exists to catch.
    """

    def __init__(self) -> None:
        self.records: List[Evidence] = []

    def record(self, evidence: Evidence) -> None:
        self.records.append(evidence)

    def __len__(self) -> int:
        return len(self.records)

    def successes(self) -> List[Evidence]:
        return [record for record in self.records if record.success]

    def failures(self) -> List[Evidence]:
        return [record for record in self.records if not record.success]

    def by_tool(self, tool: str) -> List[Evidence]:
        return [record for record in self.records if record.tool == tool]

    def to_dict(self) -> Dict[str, Any]:
        return {"count": len(self.records),
                "records": [record.to_dict() for record in self.records]}


# --- requirement adjudication -----------------------------------------------

SATISFIED = "satisfied"
PARTIALLY_SATISFIED = "partially_satisfied"
UNSUPPORTED_CLAIM = "unsupported_claim"
MISSING = "missing"

ACCEPT = "accept"
REVISE = "revise"
REJECT = "reject"


@dataclass
class Adjudication:
    """What the gate concluded about one requirement."""

    requirement_id: str
    requirement: str
    status: str
    evidence: tuple[str, ...] = ()
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.requirement_id, "requirement": self.requirement,
                "status": self.status, "evidence": list(self.evidence),
                "reason": self.reason}


@dataclass
class GateVerdict:
    """The gate's overall conclusion for a turn."""

    verdict: str
    adjudications: List[Adjudication] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict == ACCEPT

    def deficiencies(self) -> List[Adjudication]:
        return [item for item in self.adjudications
                if item.status in {UNSUPPORTED_CLAIM, MISSING, PARTIALLY_SATISFIED}]

    def to_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict, "reason": self.reason,
                "adjudications": [item.to_dict() for item in self.adjudications],
                "contradictions": list(self.contradictions)}


#: Claims a summary can make that the ledger can deterministically contradict.
#: These run before any classifier, because a string match is a fact and a
#: classifier is an opinion. Each is reported once, however many ways it matches.
_CONTRADICTIONS = (
    (re.compile(r"\ball tests (?:pass|passed|succeed)\b", re.I), "failed_tests",
     "the summary says all tests pass, but a test run is recorded as failed"),
    (re.compile(r"\beverything (?:passes|works)\b", re.I), "any_failure",
     "the summary says everything works, but a tool call is recorded as failed"),
    (re.compile(r"\bno (?:errors|failures)\b", re.I), "any_failure",
     "the summary says there are no errors, but a tool call is recorded as failed"),
    (re.compile(r"\b(?:all|every) (?:checks?|tests?) (?:pass|passed)\b", re.I),
     "failed_tests",
     "the summary says every check passed, but a run is recorded as failed"),
)


def find_contradictions(summary: str, ledger: EvidenceLedger) -> List[str]:
    """Deterministic summary-vs-ledger contradictions, found before any model.

    This is the cheapest and highest-confidence part of the gate, and it is
    entirely local. Each distinct contradiction is reported once.
    """
    found: List[str] = []
    text = summary or ""
    has_failed_tests = any(record.tool in {"run_pytest", "run_bash"}
                          and not record.success for record in ledger.records)
    has_any_failure = bool(ledger.failures())
    seen: set[str] = set()
    for pattern, kind, message in _CONTRADICTIONS:
        if message in seen or not pattern.search(text):
            continue
        if kind == "failed_tests" and has_failed_tests:
            seen.add(message)
            found.append(message)
        elif kind == "any_failure" and has_any_failure:
            seen.add(message)
            found.append(message)
    return found


def adjudicate(requirement: Requirement, ledger: EvidenceLedger,
               summary: str) -> Adjudication:
    """Judge one requirement against the evidence.

    A requirement is satisfied when a recorded, successful tool result mentions
    something the obligation was about. Anything else is a deficiency, and a
    deficiency is never rounded up to a pass.
    """
    summary_lower = (summary or "").lower()
    if not requirement.keywords:
        # Nothing distinctive to match on. The only honest thing is to say the
        # requirement could not be evidenced rather than assume it was met.
        return Adjudication(requirement.id, requirement.text, MISSING, (),
                            "the obligation has no distinctive terms to evidence it")

    evidence_ids: List[str] = []
    matched_any = False
    for record in ledger.records:
        haystack_words = set(re.findall(r"[a-z0-9_.]+", f"{record.target} {record.digest}".lower()))
        if any(any(_same_word(keyword, word) for word in haystack_words)
               for keyword in requirement.keywords):
            evidence_ids.append(f"{record.tool}:{record.target[:60]}")
            if record.success:
                matched_any = True

    if matched_any:
        return Adjudication(requirement.id, requirement.text, SATISFIED,
                            tuple(evidence_ids),
                            "a successful tool result evidences this obligation")
    if evidence_ids:
        return Adjudication(requirement.id, requirement.text, PARTIALLY_SATISFIED,
                            tuple(evidence_ids),
                            "tools touched this area but every one of them failed")
    if _summary_claims(summary_lower, requirement):
        # The model says it did the thing, and nothing supports it. This is the
        # hallucination class, and it must not pass.
        return Adjudication(requirement.id, requirement.text, UNSUPPORTED_CLAIM, (),
                            "the summary asserts this was done and no tool call "
                            "evidences it")
    return Adjudication(requirement.id, requirement.text, MISSING, (),
                        "no recorded tool call mentions this obligation")


def _summary_claims(summary_lower: str, requirement: Requirement) -> bool:
    """Whether the summary asserts this obligation was completed.

    Compared on the obligation's *keywords* rather than its exact text: a
    requirement reads "update the docs" and a summary reads "I updated the
    docs", so a literal substring test would miss the most ordinary phrasing
    there is. Stems are compared, so update/updated and add/added match.
    """
    if not summary_lower or not requirement.keywords:
        return False
    words = set(re.findall(r"[a-z]+", summary_lower))
    # A requirement with several keywords is claimed when most of them appear:
    # one word in common is a coincidence, most of them is a statement.
    hits = sum(1 for keyword in requirement.keywords
               if any(_same_word(keyword, word) for word in words))
    return hits >= max(1, (len(requirement.keywords) + 1) // 2)


def _stem(word: str) -> str:
    """A crude but sufficient stem.

    Strips the common English verb endings, then compares by prefix, because
    suffix stripping alone does not relate the forms a codebase actually
    produces: "caching" and "cache" must match, and "update"/"updated" must.
    """
    word = word.lower()
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _same_word(left: str, right: str) -> bool:
    """Whether two words are the same word in different inflections."""
    a, b = _stem(left), _stem(right)
    if a == b:
        return True
    # A shared prefix covers derivational pairs the suffix rules miss
    # (cach/cache, doc/document, migrat/migration).
    shorter = min(len(a), len(b))
    return shorter >= 4 and a[:shorter] == b[:shorter]


def evaluate(requirements: RequirementSet, ledger: EvidenceLedger, summary: str,
             *, relevance: Optional[Any] = None) -> GateVerdict:
    """Run the whole gate: adjudicate, then contradict, then decide."""
    adjudications = [adjudicate(item, ledger, summary) for item in requirements.requirements]
    contradictions = find_contradictions(summary, ledger)

    unsupported = [item for item in adjudications if item.status == UNSUPPORTED_CLAIM]
    missing = [item for item in adjudications if item.status == MISSING]
    partial = [item for item in adjudications if item.status == PARTIALLY_SATISFIED]

    if unsupported or contradictions:
        # Fails closed. A hallucination-class finding is never downgraded to a
        # note on an otherwise-passing run.
        verdict = REVISE if (unsupported or partial or missing) else REJECT
        reason = (f"{len(unsupported)} claim(s) are not supported by any recorded "
                  f"tool call"
                  + (f", and {len(contradictions)} contradiction(s) with the "
                     f"evidence ledger" if contradictions else ""))
    elif missing or partial:
        verdict = REVISE
        reason = (f"{len(missing)} obligation(s) unevidenced and "
                  f"{len(partial)} only partially evidenced")
    else:
        verdict = ACCEPT
        reason = f"all {len(adjudications)} obligation(s) are supported by recorded evidence"

    if relevance is not None:
        # A classifier may *add* findings, never remove them. The deterministic
        # pass has already spoken.
        try:
            for item in getattr(relevance, "adjudications", lambda: [])():
                adjudications.append(item)
        except Exception:  # noqa: BLE001 - a classifier cannot break the gate
            pass

    return GateVerdict(verdict=verdict, adjudications=adjudications,
                       contradictions=contradictions, reason=reason)


def revision_instruction(verdict: GateVerdict) -> str:
    """The targeted instruction a `revise` verdict returns to the model.

    Itemised and specific. "Try again" would restart the work; naming the exact
    unsupported claims lets the model repair the gap.
    """
    lines = ["The final answer claims work the recorded tool calls do not support.",
             "Address exactly these, and do not restate what is already evidenced:"]
    for item in verdict.deficiencies():
        lines.append(f"  - {item.requirement_id} ({item.status}): {item.requirement}")
        lines.append(f"      why: {item.reason}")
    for contradiction in verdict.contradictions:
        lines.append(f"  - contradiction: {contradiction}")
    lines.append("If an obligation genuinely cannot be completed, say which one and why, "
                 "rather than reporting success.")
    return "\n".join(lines)
