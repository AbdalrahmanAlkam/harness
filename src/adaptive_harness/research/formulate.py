"""Dynamic topic formulation for topics no template covers.

When a topic matches nothing in the derivation library, the swarm does not give
up and it does not invent a result. It asks the Director to *formulate candidate
claims*, and those claims are then decided by the kernel.

The division of labour is the whole point:

* the model states the claim, in SymPy syntax, as an identity;
* the kernel writes the script that decides it.

So no model-authored code is ever executed. A proposal is admitted only because
``identity_proposition`` can turn ``lhs == rhs`` into a self-adjudicating script,
and that script's exit code — not the model's assertion — becomes the verdict. A
model that proposes a false identity gets it refuted, exactly as a human would.

What this cannot do is stated plainly: it can only settle claims expressible as
exact symbolic identities. A topic needing real analysis is reported as
unformulated rather than dressed up in a checkable-looking but vacuous identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable, Mapping, Sequence

from adaptive_harness.research.claim import ParsedClaim, Proposition, parse_claim
from adaptive_harness.research.synthesis import identity_proposition

MAX_PROPOSALS = 6
# A generous cap on identity length: long enough for a real closed form, short
# enough that a model cannot smuggle a program in where a claim belongs.
MAX_IDENTITIES = 200

FORMULATION_REQUEST = """\
You are the Theoretical Lead of an autonomous research institute. A topic has been \
proposed that no derivation in your library covers. Propose candidate mathematical \
claims about it that can be decided by exact symbolic computation.

Return ONLY a JSON object of this exact shape:

{"propositions": [
  {"id": "P1",
   "name": "a short human title",
   "claim": "<lhs> == <rhs> in SymPy syntax, e.g. (x+y)**2 == x**2 + 2*x*y + y**2>",
   "symbols": ["x", "y"],
   "hypotheses": ["the conditions under which the identity holds, in words"],
   "statement": "the claim in one sentence of prose",
   "proof_sketch": "why it should hold, in two or three sentences",
   "consequence": "what it implies for the topic"}
]}

Rules:
- Every "claim" must be a single exact identity using only SymPy syntax and the \
symbols you declare. It must be checkable by symbolic simplification alone.
- Do not use decimals, floats, or the functions N, evalf, or float.
- Prefer claims that are true. If you are unsure whether a claim holds, still \
propose it: the harness will decide, and a refutation is a useful result.
- Propose at most {max_proposals} claims. Return an empty list if the topic admits \
no such claim.
"""

_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


class FormulationError(RuntimeError):
    """The model could not be asked, or its reply could not be used."""


@dataclass(frozen=True)
class Formulation:
    """The outcome of asking the Director to formulate claims for a topic."""

    propositions: tuple[Proposition, ...]
    notes: str
    requested: bool

    @property
    def empty(self) -> bool:
        return not self.propositions


def _extract_json(reply: str) -> dict[str, Any] | None:
    """Pull the JSON object out of a reply that may be fenced or chatty."""
    if not reply:
        return None
    text = reply.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = _JSON_OBJECT.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _clean_identity(claim: Any) -> str | None:
    """Reject anything that is not a single, exact, checkable identity."""
    if not isinstance(claim, str) or "==" not in claim:
        return None
    text = " ".join(claim.split())
    if len(text) > MAX_IDENTITIES or text.count("==") != 1:
        return None
    # Approximation is inadmissible: a claim stated with N() or a decimal is not
    # an exact identity and would be rejected by the proof gate anyway.
    if re.search(r"(?<![\w.])N\s*\(", text) or re.search(r"(?<![\w.])evalf", text):
        return None
    if re.search(r"\d+\.\d", text) or re.search(r"(?<![\w.])float\s*\(", text):
        return None
    if "#" in text or "\n" in text:
        return None
    return text


def _symbols_from(entry: Mapping[str, Any], identity: str) -> tuple[str, ...]:
    declared = entry.get("symbols")
    names: list[str] = []
    if isinstance(declared, (list, tuple)):
        for item in declared:
            if isinstance(item, str) and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", item.strip()):
                cleaned = item.strip()
                if cleaned not in names:
                    names.append(cleaned)
    if names:
        return tuple(names)
    # Fall back to the free names appearing in the identity itself.
    reserved = {"sqrt", "exp", "log", "pi", "sin", "cos", "tan", "Abs", "Rational", "oo",
                "integrate", "simplify", "sum", "symbols", "oo", "I", "ln"}
    found: list[str] = []
    for token in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", identity):
        if token not in reserved and token not in found:
            found.append(token)
    return tuple(found)


def propositions_from_payload(payload: Mapping[str, Any],
                             prefix: str = "PROP-D") -> tuple[Proposition, ...]:
    """Turn a model reply into kernel-checkable propositions.

    Every entry must yield an identity the kernel can decide; anything else is
    dropped, so a reply that ignores the contract produces no propositions rather
    than uncheckable ones.
    """
    raw = payload.get("propositions")
    if not isinstance(raw, (list, tuple)):
        return ()
    built: list[Proposition] = []
    for index, entry in enumerate(raw[:MAX_PROPOSALS], start=1):
        if not isinstance(entry, Mapping):
            continue
        identity = _clean_identity(entry.get("claim"))
        if identity is None:
            continue
        symbols = _symbols_from(entry, identity)
        parsed: ParsedClaim = parse_claim(identity, symbols)
        if not parsed.machine_checkable:
            continue
        try:
            proposition = identity_proposition(parsed, prop_id=f"{prefix}{index:02d}")
        except ValueError:
            continue
        # Carry the model's prose so the paper can state a real theorem, but keep
        # the kernel-generated script untouched.
        name = entry.get("name")
        built.append(Proposition(
            prop_id=proposition.prop_id,
            kind=proposition.kind,
            name=str(name).strip() if isinstance(name, str) and name.strip()
            else f"Formulated claim {proposition.prop_id}",
            hypotheses=_string_tuple(entry.get("hypotheses")) or proposition.hypotheses,
            statement=str(entry.get("statement") or "").strip() or proposition.statement,
            proof_sketch=str(entry.get("proof_sketch") or "").strip() or proposition.proof_sketch,
            consequence=str(entry.get("consequence") or "").strip() or proposition.consequence,
            display=proposition.display,
            notation=proposition.notation,
            script=proposition.script,
            rationale=("Formulated by the Theoretical Lead for an uncovered topic, then decided "
                       "by a kernel-generated script. Only the claim came from the model; the "
                       "deciding script and the verdict did not.")))
    return tuple(built)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return ()


def formulate(topic: str, *, client: Any = None,
              author: Callable[[Any, str, str], str] | None = None,
              max_proposals: int = MAX_PROPOSALS) -> Formulation:
    """Ask for candidate claims about ``topic`` and turn them into propositions.

    Returns a :class:`Formulation` whose ``requested`` flag distinguishes "no
    model was available" from "the model was asked and declined", because those
    are different situations and the run should report them differently.
    """
    if client is None and author is None:
        return Formulation((), "No live model was available to formulate claims for this topic, "
                               "so the claim was not tested.", False)
    # A plain replace, not str.format: the request embeds a literal JSON example
    # whose braces would be read as format fields.
    request = FORMULATION_REQUEST.replace("{max_proposals}", str(max_proposals))
    reply = ""
    try:
        if client is not None:
            # Name a model when the client advertises one; a provider that
            # validates its response model field rejects an explicit None.
            named = getattr(client, "default_model", None)
            response = client.complete(
                messages=[{"role": "system", "content": request},
                          {"role": "user", "content": f"Topic: {topic}"}],
                **({"model": named} if isinstance(named, str) and named else {}))
            reply = response.content or ""
        else:
            reply = author(  # type: ignore[misc]
                None, "propositions",
                f"{request}\n\nTopic: {topic}") or ""
    except Exception as exc:  # noqa: BLE001 - a failed request is a negative result
        return Formulation((), f"Formulating claims for '{topic}' failed: "
                               f"{type(exc).__name__}: {exc}".replace("\n", " ")[:300], True)

    payload = _extract_json(reply)
    if payload is None:
        return Formulation((), f"The reply for '{topic}' was not a usable JSON proposition list, "
                               f"so no claim was formulated. First 120 characters: "
                               f"{reply.strip()[:120]!r}", True)
    propositions = propositions_from_payload(payload)
    if not propositions:
        return Formulation((), f"The Director formulated no checkable identity for '{topic}'. "
                               f"An exact symbolic identity is required for the kernel to decide "
                               f"it; a claim needing real analysis cannot be settled this way.", True)
    return Formulation(propositions,
                       f"The Director formulated {len(propositions)} candidate claim(s) for "
                       f"'{topic}'. Each is decided by a kernel-generated script, so the verdict "
                       f"comes from the exit code and not from the model's assertion.", True)


def render_spec_section(formulation: Formulation) -> str:
    """The objective-spec text describing dynamically formulated claims."""
    lines = ["## Formulated Claims", ""]
    if formulation.empty:
        lines += [formulation.notes, ""]
        return "\n".join(lines)
    lines += [formulation.notes, ""]
    for proposition in formulation.propositions:
        lines.append(f"- `{proposition.prop_id}` — {proposition.name}")
        lines.append(f"  - Statement: {proposition.statement}")
        if proposition.hypotheses:
            lines.append(f"  - Hypotheses: {'; '.join(proposition.hypotheses)}")
    lines.append("")
    return "\n".join(lines)
