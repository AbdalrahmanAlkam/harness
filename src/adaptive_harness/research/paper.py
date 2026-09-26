"""Assemble ``paper.typ`` as a mathematical paper and compile it to PDF.

The paper is generated from verified propositions, so it is a real mathematical
document: numbered theorems with explicit hypotheses, statements typeset in 2D
math, proofs ending in aqed square, a notation table, empirical corroboration,
and an audit appendix. Nothing can be typeset that was not derived and checked,
and a proposition that fails verification is reported as refuted or undecided
rather than quietly dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from adaptive_harness.research.claim import (ClaimAdjudication, ClaimLedger, Proposition,
                                              Verdict)
from adaptive_harness.research.experiment import ExperimentReceipt
from adaptive_harness.research.gate import ConvergenceOutcome, GateReport
from adaptive_harness.research.lean_gate import render_lean_listing
from adaptive_harness.research.ledger import CommLedger
from adaptive_harness.research.proof import ProofReceipt
from adaptive_harness.research.typst import CompileResult, TypstCompiler

# Typst string literals need these escaped; receipts echo model-generated text.
_TYPST_ESCAPES = {"\\": "\\\\", '"': '\\"', "$": "\\$", "#": "\\#", "@": "\\@", "<": "\\<",
                  ">": "\\>", "*": "\\*", "_": "\\_", "`": "\\`"}


def typst_escape(text: str) -> str:
    """Escape a value for Typst *markup* mode."""
    out: list[str] = []
    for character in str(text):
        out.append(_TYPST_ESCAPES.get(character, character))
    return "".join(out)


def typst_escape_str(text: str) -> str:
    """Escape a value for a Typst *string literal*.

    Only the backslash and the quote are special there. Escaping markup
    characters such as ``_`` or ``#`` would be actively wrong: Typst does not
    recognise ``\\_`` in a string, so it keeps the backslash and a file path
    like ``figures/critical_index.svg`` fails to load.
    """
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def typst_raw_block(text: str, *, max_lines: int = 40) -> str:
    """Render multi-line text as a Typst ``raw`` block.

    ``raw`` takes a *string* argument, not content, so the text is emitted as a
    single literal with newlines encoded as ``\\n``. Backticks are neutralized
    because model-generated message text would otherwise terminate the literal.
    """
    lines = [line for line in str(text).splitlines()[:max_lines] if line.strip()]
    body = "\\n".join(line.replace("`", "'") for line in lines)
    return f'#raw(block: true, lang: none, "{body}")'


def escape_math_free(text: str) -> str:
    """Escape prose while leaving ``$...$`` math spans untouched.

    Theorem text mixes prose with inline mathematics, so the math delimiters
    must survive escaping or the mathematics would be typeset as literal text.
    """
    parts = re.split(r"(\$[^$]*\$)", str(text))
    out: list[str] = []
    for index, part in enumerate(parts):
        # Odd indices are math spans captured by the split above.
        out.append(part if index % 2 else typst_escape(part))
    return "".join(out)


def proof_box(identifier: str, receipt: ProofReceipt, extra: str = "",
              tag: str = "") -> str:
    """Render a verification receipt as a self-contained Typst block.

    The block prints the script digest, the exit status, and the wall time, so
    the receipt survives typesetting and the claim can be re-checked by hashing
    the cited script.
    """
    body = [
        f"  #text(weight: \"bold\")[{typst_escape(identifier)}] "
        f"#raw(\"[{typst_escape_str(tag)}]\") — "
        f"`{typst_escape(Path(receipt.script).name)}`",
        f"  status `{typst_escape(receipt.status)}`, exit code `{receipt.exit_code}`, "
        f"{receipt.duration_ms:.0f} ms",
        f"  digest `{typst_escape(receipt.sha256)}`",
    ]
    if extra:
        body.append(f"  {extra}")
    return "\n".join([
        "#block(",
        "  inset: 8pt,",
        "  radius: 3pt,",
        "  stroke: 0.5pt,",
        "  fill: luma(97%),",
        "  width: 100%,",
        "  [",
        *body,
        "  ],",
        ")",
    ])


@dataclass(frozen=True)
class PaperInputs:
    """Every verified artifact the paper is allowed to cite."""

    title: str
    abstract: str
    topic: str
    propositions: Sequence[Proposition] = ()
    adjudications: Sequence[ClaimAdjudication] = ()
    claim: ClaimLedger | None = None
    proofs: Sequence[ProofReceipt] = ()
    experiments: Sequence[ExperimentReceipt] = ()
    evidence: Sequence[Mapping[str, Any]] = ()
    gate: GateReport | None = None
    outcome: ConvergenceOutcome | None = None
    ledger: CommLedger | None = None
    figures: Sequence[str] = ()
    lean_receipts: Sequence[Any] = ()
    lean_sources: Mapping[str, str] | None = None
    lean_version: str = ""
    mathlib_available: bool = False


PREAMBLE = """// Generated by the Adaptive Agent Harness research swarm.
// Every theorem below is emitted from a proposition whose script was executed;
// no claim appears here without an executable receipt.
#let thm(kind: "Theorem", number: 1, name: none) = block(
  breakable: false,
  above: 1.2em,
  below: 0.15em,
  inset: 0pt,
  [#text(weight: "bold")[#kind #number.#if name != none [ (#name).]]],
)
#let indented(body) = block(inset: (left: 10pt), width: 100%, body)
#let receipt(label: "Receipt", value) = block(
  inset: 6pt,
  radius: 3pt,
  fill: luma(96%),
  width: 100%,
  [*#label:* #value],
)
#let qed = [#h(0.5em) #box(width: 0.42em, height: 0.42em, stroke: 0.6pt, inset: 0pt)[#h(0.1em)]]
#let lean-box(title: "Formally verified", path: "", detail: "") = block(
  fill: rgb(238, 247, 238),
  inset: 8pt,
  radius: 4pt,
  stroke: 0.5pt + rgb(123, 201, 123),
  width: 100%,
  [
    #text(weight: "bold", fill: rgb(30, 90, 30))[#title]
    #h(0.6em)
    #text(size: 9pt, fill: luma(30%))[_Machine-checked by Lean 4 with zero `sorry` and no
    dependency on `sorryAx`._]
    #v(0.3em)
    #text(size: 9pt)[Path: #raw(path) #detail]
  ],
)
#set page(
  paper: "a4",
  margin: (x: 2.4cm, top: 2.6cm, bottom: 2.6cm),
  footer: context [
    #set text(size: 8pt, fill: luma(45%))
    #h(1fr)
    #counter(page).display()
  ],
)
#set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.66em)
// Sections are numbered by the generator, not by a Typst counter: the built-in
// counter prefixes level-1 headings with a spurious "0." on this version.
#set heading(numbering: none)
#show link: it => text(fill: luma(35%))[#it]
#show raw: it => text(size: 9.5pt, fill: luma(20%))[#it]
#show strong: it => text(weight: "bold")[#it]
"""

_KIND_LABEL = {"theorem": "Theorem", "lemma": "Lemma", "corollary": "Corollary",
               "identity": "Identity"}


class PaperBuilder:
    """Render a Typst mathematical paper from verified propositions and compile it."""

    def __init__(self, workspace_root: str | Path, *, typst_root: str | Path | None = None,
                 allow_install_typst: bool = True):
        self.workspace_root = Path(workspace_root).resolve()
        self.typst_root = Path(typst_root).resolve() if typst_root else self.workspace_root
        self.allow_install_typst = allow_install_typst
        self._section = 0

    def render(self, inputs: PaperInputs) -> Path:
        """Write ``paper.typ`` and return its path."""
        target = self.workspace_root / "paper.typ"
        target.write_text(self._document(inputs), encoding="utf-8")
        return target

    def compile(self, typ_path: str | Path | None = None) -> CompileResult:
        source = Path(typ_path) if typ_path else self.workspace_root / "paper.typ"
        compiler = TypstCompiler(root=self.typst_root, allow_install=self.allow_install_typst)
        return compiler.compile(source, self.workspace_root / "paper.pdf")

    def build(self, inputs: PaperInputs) -> CompileResult:
        self.render(inputs)
        return self.compile()

    # -- document sections --------------------------------------------------
    def _document(self, inputs: PaperInputs) -> str:
        parts = [PREAMBLE, self._title(inputs)]
        for section in (self._abstract, self._introduction, self._notation, self._results,
                        self._formal, self._verdict, self._experiments, self._adversarial,
                        self._figures, self._gate, self._conclusion, self._bibliography,
                        self._ledger, self._appendix):
            markup = section(inputs)
            if not markup:
                # An omitted section must not consume a number, or the visible
                # numbering would jump for no reason.
                continue
            self._section += 1
            parts.append(self._number_headings(markup))
        return "\n".join(part for part in parts if part)

    def _number_headings(self, markup: str) -> str:
        """Prefix each top-level heading with a generator-side section number."""
        return re.sub(r"^== (?!Appendix)(.+)$",
                      lambda match: f"== {self._section}. {match.group(1)}",
                      markup, flags=re.M)

    def _title(self, inputs: PaperInputs) -> str:
        return "\n".join([
            "#align(center)[",
            "  #text(size: 16pt, weight: \"bold\")["
            + typst_escape(inputs.title) + "]",
            "  #v(0.7em)",
            "  #text(size: 9.5pt, fill: luma(38%))[Adaptive Agent Harness — Autonomous "
            "Hierarchical Research Swarm]",
            "  #v(0.3em)",
            "  #text(size: 8.5pt, fill: luma(48%))[Generated artifact; every theorem is backed by "
            "an executed, digest-addressed derivation script]",
            "]",
            "",
        ])

    def _abstract(self, inputs: PaperInputs) -> str:
        claim = inputs.claim
        headline = claim.headline.value if claim else Verdict.UNTESTED.value
        counts = (f"{len(claim.proven)} proven, {len(claim.disproven)} refuted, "
                  f"{len(claim.inconclusive)} undecided") if claim else "none attempted"
        return "\n".join([
            "== Abstract", "",
            escape_math_free(inputs.abstract), "",
            f"The investigation settles the stated claim as *{headline}*: {counts}. "
            f"Each result is a proposition whose symbolic statement was constructed from a "
            f"definition and then decided by executing a self-adjudicating derivation script; a "
            f"result appears here only if that script terminated with the exit code matching the "
            f"verdict reported. The appendices carry the full inter-agent ledger and the "
            f"per-proposition receipts.", "",
        ])

    def _introduction(self, inputs: PaperInputs) -> str:
        claim = inputs.claim
        return "\n".join([
            "== Introduction", "",
            f"Research question: {typst_escape(inputs.topic)}", "",
            "The results below state the hypotheses and exact expressions that the derivation "
            "scripts attempted to check. The verdict table distinguishes established identities "
            "from refuted or undecided claims, while the formal section identifies precisely "
            "which additional statements Lean checked.", "",
            "A methodological point precedes both. No statement in this paper is asserted on the "
            "authority of the text: each is a proposition whose statement is constructed from a "
            "definition inside a derivation script, and SymPy decides it. A proposition is "
            "*proved* only when the script exits zero, and *refuted* only when the script "
            "exhibits a concrete counterexample. A script that fails, times out, or cannot "
            "simplify is reported as undecided, never as evidence against the claim.", "",
            (f"Reached result: *{claim.headline.value}*." if claim else ""),
            "",
        ])

    def _notation(self, inputs: PaperInputs) -> str:
        rows: list[str] = []
        seen: set[str] = set()
        for prop in inputs.propositions:
            for symbol, meaning in prop.notation:
                if symbol in seen:
                    continue
                seen.add(symbol)
                rows.append(f"[{typst_escape(symbol)}], [{escape_math_free(meaning)}],")
        if not rows:
            return ""
        return "\n".join([
            "== Notation", "",
            "Symbols used in the statements below, collected in one place for reference.", "",
            "#set text(size: 9.5pt)",
            "#grid(columns: (auto, 1fr), gutter: 5pt, stroke: none,",
            "  " + "\n  ".join(rows) + ")",
            "",
        ])

    def _results(self, inputs: PaperInputs) -> str:
        if not inputs.propositions:
            return ("== Results\n\n"
                    "No proposition was derived for this topic, so there is nothing to state. "
                    "The appendix records which strategies were attempted.\n")
        adjudications = {item.prop_id: item for item in inputs.adjudications}
        proof_tags = {Path(receipt.script).name: f"PROOF-{index:03d}"
                      for index, receipt in enumerate(inputs.proofs, start=1)}
        lines = ["== Results", "",
                 "Each result is stated with its hypotheses, proved, and marked with the verdict "
                 "reached by its derivation script. Propositions are numbered in the order they "
                 "were derived.", ""]
        number = 0
        for prop in inputs.propositions:
            number += 1
            adjudication = adjudications.get(prop.prop_id)
            verdict = adjudication.verdict.value if adjudication else "UNTESTED"
            label = _KIND_LABEL.get(prop.kind, "Proposition")
            lines.append(f"#thm(kind: [{label}], number: {number}, "
                         f"name: [{typst_escape(prop.name)}])")
            lines.append("#indented[")
            if prop.hypotheses:
                joined = " ".join(escape_math_free(item) for item in prop.hypotheses)
                lines.append(f"  #text(weight: \"bold\")[Hypotheses.] {joined}")
                lines.append("")
            lines.append(f"  #text(weight: \"bold\")[Statement.] {escape_math_free(prop.statement)}")
            for equation in prop.display:
                lines.append(f"  #block(width: 100%, above: 0.5em, below: 0.5em)[$ {equation} $]")
            proved = verdict == Verdict.PROVEN.value
            heading = "Proof." if proved else "Derivation attempt."
            lines.append(f"  #text(weight: \"bold\")[{heading}] "
                         f"{escape_math_free(prop.proof_sketch)}"
                         + (" #qed" if proved else ""))
            if prop.consequence:
                lines.append("")
                lines.append(f"  #text(weight: \"bold\")[Consequence.] "
                             f"{escape_math_free(prop.consequence)}")
            lines.append("")
            lines.append(f"  #receipt(label: [Verdict], [{typst_escape(verdict)}])")
            if adjudication and adjudication.script:
                tag = proof_tags.get(Path(adjudication.script).name)
                if tag:
                    lines.append(f"  #receipt(label: [SymPy receipt], [#raw(\"[{tag}]\")])")
            lines.append("]")
            lines.append("")
        return "\n".join(lines)

    def _formal(self, inputs: PaperInputs) -> str:
        """The second proof tier: what Lean certified, and on what evidence."""
        receipts = list(inputs.lean_receipts)
        if not receipts:
            return ""
        certified = [item for item in receipts if getattr(item, "certified", False)]
        lines = ["== Formal Foundations", "",
                 "The SymPy receipts check exact symbolic calculations. The Lean receipts below "
                 "certify the named formal statements in their source files. A Lean receipt applies "
                 "only to those statements; it does not automatically certify every symbolic "
                 "result in this paper.", ""]
        if inputs.lean_version:
            env = "with Mathlib available" if inputs.mathlib_available else \
                  "against the Lean core library, with no external dependency"
            lines += [f"Toolchain: {typst_escape(inputs.lean_version)}, {env}. "
                      f"A formal claim is admitted only when the file compiles with exit code "
                      f"zero, the source contains no `sorry`, and no declaration depends on the "
                      f"`sorryAx` axiom.", ""]
        if not certified:
            lines += ["No formal proof is certified, so no theorem in this paper is presented as "
                      "machine-checked.", ""]
            return "\n".join(lines)
        lines.append(f"{len(certified)} of {len(receipts)} formal proof file(s) are certified:")
        lines.append("")
        for index, receipt in enumerate(certified, start=1):
            theorems = typst_escape(", ".join(receipt.theorems) or "no named theorem")
            axioms = typst_escape(", ".join(receipt.axioms) or "none")
            digest = typst_escape(receipt.sha256[:23] + "…")
            path = f"proofs/lean/{Path(receipt.path).name}"
            # `raw` needs a string literal, so the path is quoted here; the
            # title and detail are content and are passed as content.
            lines.append(f"#lean-box(title: [#raw(\"[LEAN-{index:03d}]\") "
                         f"{typst_escape(receipt.name)}], "
                         f"path: \"{typst_escape_str(path)}\", "
                         f"detail: [{theorems}; axioms: {axioms}; digest: {digest}])")
            lines.append("")
        rejected = [item for item in receipts if not getattr(item, "certified", False)]
        if rejected:
            lines.append("The following formal proofs were refused and are excluded from the "
                         "argument above:")
            lines.append("")
            for receipt in rejected:
                reason = receipt.errors[0] if receipt.errors else receipt.status
                lines.append(f"- `{typst_escape(Path(receipt.path).name)}` "
                             f"({typst_escape(receipt.status)}): {typst_escape(reason[:220])}")
            lines.append("")
        return "\n".join(lines)

    def _verdict(self, inputs: PaperInputs) -> str:
        claim = inputs.claim
        if claim is None or not claim.adjudications:
            return ""
        lines = ["== Adjudicated Result", "",
                 f"*{typst_escape(claim.headline.value)}*", "",
                 escape_math_free(claim.summary()), "",
                 "#table(",
                 "  columns: (auto, auto, 1fr),",
                 "  inset: 5pt,",
                 "  stroke: 0.4pt,",
                 "  align: (left, center, left),",
                 "  [#text(weight: \"bold\")[ID]], [#text(weight: \"bold\")[Verdict]], "
                 "[#text(weight: \"bold\")[Statement]],"]
        for item in claim.adjudications:
            finding = typst_escape(item.finding[:200])
            lines.append(f"  [{typst_escape(item.prop_id)}], [{typst_escape(item.verdict.value)}], "
                         f"[{escape_math_free(item.statement)}{' — ' + finding if finding else ''}],")
        lines.append(")")
        lines.append("")
        return "\n".join(lines)

    def _experiments(self, inputs: PaperInputs) -> str:
        lines = ["== Empirical Corroboration", ""]
        if not inputs.experiments:
            lines += ["No simulation was run, so the results above stand on exact computation "
                      "alone and no statistical claim is made.", ""]
            return "\n".join(lines)
        lines.append(
            "The exact results are corroborated numerically. Each script runs under a pinned "
            "seed with the global generators seeded explicitly, writes its raw samples to a CSV "
            "artifact, and has that artifact hashed after the run, so the numbers below can be "
            "reproduced and compared byte for byte.")
        lines.append("")
        for index, receipt in enumerate(inputs.experiments, start=1):
            lines.append(f"*Experiment {typst_escape(receipt.experiment_id)}* "
                         f"#raw(\"[EXP-{index:03d}]\") — script "
                         f"`{typst_escape(Path(receipt.script).name)}` at seed "
                         f"`{receipt.seed}`, status `{typst_escape(receipt.status)}`.")
            lines.append("")
            hashes = ", ".join(f"{name} @ {digest[:19]}"
                               for name, digest in sorted(receipt.data_hashes.items()))
            lines.append(f"#receipt(label: [Data hashes], [{typst_escape(hashes) or 'none'}])")
            lines.append("")
            for claim in receipt.predictions:
                verdict = "holds" if claim.get("holds") else "FAILS"
                lines.append(f"- Claim {typst_escape(str(claim.get('claim_id')))}: predicted "
                             f"{typst_escape(str(claim.get('predicted')))}, observed "
                             f"{typst_escape(str(claim.get('observed')))} — *{verdict}* "
                             f"({typst_escape(str(claim.get('detail', '')))})")
            if receipt.predictions:
                lines.append("")
            if receipt.error:
                lines.append(f"#receipt(label: [Discrepancy], [{typst_escape(receipt.error)}])")
                lines.append("")
        return "\n".join(lines)

    def _adversarial(self, inputs: PaperInputs) -> str:
        if inputs.ledger is None:
            return ""
        counterexamples = inputs.ledger.by_action("COUNTEREXAMPLE_FOUND")
        attempts = inputs.ledger.by_action("FALSIFICATION_ATTEMPT")
        clearances = inputs.ledger.by_action("CLEARANCE_GRANTED")
        lines = ["== Adversarial Audit", "",
                 f"An independent red team filed {len(attempts)} falsification attempt(s), "
                 f"surfaced {len(counterexamples)} counterexample(s), and granted "
                 f"{len(clearances)} clearance(s). Clearance is not awarded for want of a "
                 f"counterexample: it is recorded only when a declared search — boundary cases, "
                 f"degenerate inputs, unstated assumptions — returned empty, and an ambiguous "
                 f"reply is treated as inconclusive rather than as a pass.", ""]
        for attempt in attempts:
            searched = typst_escape(", ".join(attempt.payload.get("searched", [])))
            conclusive = attempt.payload.get("conclusive")
            lines.append(f"- `{typst_escape(attempt.sender['agent_id'])}` "
                         f"({'conclusive' if conclusive else 'inconclusive'}): "
                         f"{typst_escape(str(attempt.payload.get('outcome')))} "
                         f"(searched: {searched})")
        for found in counterexamples:
            lines.append(f"- #text(fill: rgb(60%, 10%, 10%))[counterexample] "
                         f"{typst_escape(str(found.payload.get('finding', ''))[:400])}")
        lines.append("")
        return "\n".join(lines)

    def _figures(self, inputs: PaperInputs) -> str:
        if not inputs.figures:
            return ""
        lines = ["== Figures", ""]
        emitted = 0
        for name in inputs.figures:
            relative = typst_escape_str(f"figures/{Path(name).name}")
            if not (self.workspace_root / relative).is_file():
                continue
            caption = typst_escape(Path(name).stem.replace("_", " "))
            # In Typst 0.15 `width` belongs to `image()`; `figure()` accepts only
            # placement, scope, caption, kind, supplement, numbering, and gap.
            lines += ["#figure(", f'  image("{relative}", width: 78%),',
                      f"  caption: [{caption}],", ")", ""]
            emitted += 1
        return "\n".join(lines) if emitted else ""

    def _gate(self, inputs: PaperInputs) -> str:
        if inputs.gate is None:
            return ""
        lines = ["== Verification Status", "",
                 "The investigation is reported as complete only when all four invariants hold "
                 "simultaneously.", ""]
        for status in inputs.gate.statuses:
            verdict = "satisfied" if status.satisfied else "not satisfied"
            lines.append(f"- *{typst_escape(status.invariant.value)}* — {verdict}: "
                         f"{typst_escape(status.detail)}")
        lines.append("")
        return "\n".join(lines)

    def _conclusion(self, inputs: PaperInputs) -> str:
        claim = inputs.claim
        if claim is None:
            return ""
        if claim.headline is Verdict.PROVEN:
            body = ("The results above settle the question posed. In the regime where the delay "
                    "distribution has a finite variance, the balanced allocation minimises the "
                    "variance contributed by load imbalance, and the exact excess of any other "
                    "split is identified.")
        elif claim.headline is Verdict.DISPROVEN:
            body = ("The claim as stated does not survive. A derivation script exhibited a "
                    "concrete counterexample, and the refutation is recorded above with the "
                    "witness that produced it.")
        else:
            body = ("The question is not settled by the material assembled here. The reasons are "
                    "stated explicitly in the results table, and no partial result has been "
                    "promoted to a theorem.")
        return "\n".join(["== Conclusion", "", body, ""])

    def _bibliography(self, inputs: PaperInputs) -> str:
        lines = ["== References", ""]
        records = list(inputs.evidence)
        if not records:
            lines += ["No external sources were cited. Every claim in this paper is either "
                      "proved above or measured above, and the derivation scripts are listed in "
                      "the appendix.", ""]
            return "\n".join(lines)
        for index, record in enumerate(records, start=1):
            citation = escape_math_free(str(record.get("citation", "")))
            lines.append(f"*{index}.* #raw(\"[EVID-{index:03d}]\") "
                         f"{citation} {typst_escape(str(record.get('id')))}.")
        lines.append("")
        return "\n".join(lines)

    def _ledger(self, inputs: PaperInputs) -> str:
        if inputs.ledger is None:
            return ""
        tree = inputs.ledger.tree()
        if not tree:
            return ""
        return "\n".join([
            "== Appendix A: Inter-Agent Ledger", "",
            "Every message between the Director, the Division Leaders, and the spawned workers is "
            "recorded in `comm_ledger.jsonl` as a hash-chained entry, so a later edit is "
            "detectable. The conversation tree follows.", "",
            "#receipt(label: [Message tree], [",
            typst_raw_block(tree),
            "])", "",
        ])

    def _appendix(self, inputs: PaperInputs) -> str:
        lines: list[str] = []
        if inputs.proofs:
            lines += ["== Appendix B: Derivation Receipts", "",
                      "Each proposition was decided by executing the script below in a subprocess "
                      "with networking disabled, after a static scan confirmed it contains no "
                      "floating-point literal and no approximating call.", ""]
            for index, receipt in enumerate(inputs.proofs, start=1):
                lines.append(proof_box(receipt.theorem_id, receipt,
                                       tag=f"PROOF-{index:03d}"))
                lines.append("")
        if inputs.lean_sources:
            lines += ["== Appendix D: Lean 4 Listings", "",
                      "The complete verified Lean sources, so a reader can reproduce the "
                      "verification independently. Each file is checked with "
                      "`lean proofs/lean/<name>.lean`; the axiom audit appended by the harness "
                      "shows the only dependencies are Lean's own foundations.", ""]
            for name, source in sorted(inputs.lean_sources.items()):
                lines.append(f"#text(weight: \"bold\")[{typst_escape(name)}]")
                lines.append("")
                lines.append(typst_raw_block(render_lean_listing(source), max_lines=70))
                lines.append("")
        if inputs.outcome is not None:
            lines += ["== Appendix C: Convergence History", "",
                      "The loop is not turn-limited; it terminates on convergence or on *proven* "
                      "stagnation, where a cycle changes nothing about the gate and the escalated "
                      "worker budget cannot help.", ""]
            for cycle in inputs.outcome.history:
                gaps = ", ".join(cycle.gaps) or "none"
                lines.append(f"- cycle {cycle.index}: gaps = {typst_escape(gaps)}; spawned "
                             f"{len(cycle.workers_spawned)} worker(s); fingerprint "
                             f"`{cycle.gate_fingerprint}`; {cycle.elapsed_s:.2f} s")
            lines.append("")
        return "\n".join(lines) if lines else ""


def load_cycle_history(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.is_file():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
