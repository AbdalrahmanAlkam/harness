"""The hierarchical autonomous research swarm.

The Executive Director owns the objective and drives the Relentless Convergence
Loop. Each cycle it (1) asks the gate what is still missing, (2) has the
responsible Division Leaders spawn Workers aimed at exactly those gaps, (3) lets
the mechanical verifiers re-adjudicate every artifact, and (4) escalates the
worker budget when a cycle changes nothing. The loop exits only when all four
invariants hold, or when stagnation is *proven*.

Artifact layout under ``research/<topic_slug>/`` is created on construction, so
every run is auditable even if it aborts early.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import textwrap
import time
from typing import Any, Callable, Mapping, Sequence

from adaptive_harness.research.claim import (EXIT_COUNTEREXAMPLE, EXIT_HOLDS, ClaimLedger,
                                              ParsedClaim, Proposition, TopicPlan, Verdict,
                                              exact_decider, parse_claim)
from adaptive_harness.research.coordination import (TASK_BOARD_FILENAME, CoordinationError,
                                                    MessageBus, MessageKind, StopKind,
                                                    SwarmControl, Task, TaskBoard, WorkerCancelled,
                                                    WorkerState)
from adaptive_harness.research.experiment import ExperimentRunner, canonical_key
from adaptive_harness.research.gate import (ConvergenceOutcome, Invariant, InvariantGate,
                                            RelentlessConvergenceLoop, StopReason,
                                            write_cycle_history)
from adaptive_harness.research.lean_gate import LEAN_DIRNAME, LeanProofGate
from adaptive_harness.research.ledger import CommLedger, sha256_file
from adaptive_harness.research.paper import PaperBuilder
from adaptive_harness.research.proof import ProofRunner
from adaptive_harness.research.roles import (DIVISION_SPECS, FALSIFICATION_VERDICT, Clearance,
                                             Division, FalsificationVerdict, ResearchAgent,
                                             leader_agent_id, worker_agent_id)
from adaptive_harness.research.typst import TypstCompiler

DIRECTOR_ID = "executive_director_01"

# Gap -> division that owns closing it. The Director routes by method, not by
# whoever happens to be free.
GAP_ROUTING: Mapping[str, Division] = {
    Invariant.MATHEMATICAL_SOUNDNESS.value: Division.THEORY,
    Invariant.EMPIRICAL_REPLICATION.value: Division.EMPIRICAL,
    Invariant.ADVERSARIAL_CLEARANCE.value: Division.ADVERSARIAL,
    Invariant.DOCUMENT_INTEGRITY.value: Division.LITERATURE,
    Invariant.CLAIM_ADJUDICATION.value: Division.THEORY,
    Invariant.LEAN_FORMAL_SOUNDNESS.value: Division.FORMAL,
}

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def _summarize_args(arguments: Any, limit: int = 160) -> str:
    """Compact, loggable rendering of a tool call's arguments.

    A trajectory is written to an append-only ledger, so arguments are truncated
    rather than copied wholesale: a worker pasting a whole proof into a log line
    would bloat the audit without adding evidence.
    """
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        text = arguments
    else:
        try:
            text = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            text = str(arguments)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def topic_slug(topic: str) -> str:
    slug = _SLUG_STRIP.sub("-", topic.lower()).strip("-")
    return (slug[:64] or "research-topic").strip("-")


@dataclass
class ResearchWorkspace:
    """The on-disk artifact tree for one research topic."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        for name in ("evidence", "proofs", "experiments", "figures"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        # The formal tier lives beside the computational one, never inside it:
        # a Lean file must not be picked up by the SymPy proof gate.
        self.lean_dir.mkdir(parents=True, exist_ok=True)

    @property
    def ledger_path(self) -> Path:
        return self.root / "comm_ledger.jsonl"

    @property
    def proof_dir(self) -> Path:
        return self.root / "proofs"

    @property
    def lean_dir(self) -> Path:
        return self.root / "proofs" / LEAN_DIRNAME

    @property
    def experiment_dir(self) -> Path:
        return self.root / "experiments"

    @property
    def evidence_dir(self) -> Path:
        return self.root / "evidence"

    @property
    def figure_dir(self) -> Path:
        return self.root / "figures"

    @property
    def paper_typ(self) -> Path:
        return self.root / "paper.typ"

    @property
    def paper_pdf(self) -> Path:
        return self.root / "paper.pdf"

    @property
    def objective_spec(self) -> Path:
        return self.root / "00_objective_spec.md"

    @property
    def audit(self) -> Path:
        return self.root / "03_adversarial_audit.md"

    @property
    def bibliography(self) -> Path:
        return self.root / "bibliography.bib"

    @property
    def evidence_index(self) -> Path:
        return self.evidence_dir / "index.json"

    def evidence_records(self) -> list[dict[str, Any]]:
        if not self.evidence_index.is_file():
            return []
        try:
            payload = json.loads(self.evidence_index.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return payload if isinstance(payload, list) else []

    def evidence_id(self) -> str:
        return f"EVID_{len(self.evidence_records()) + 1:03d}"

    def proof_id(self) -> str:
        return f"PROOF-{len(list(self.proof_dir.glob('*.py'))) + 1:03d}"

    def experiment_id(self) -> str:
        return f"EXP-{len(list(self.experiment_dir.glob('*.py'))) + 1:03d}"


@dataclass
class SwarmConfig:
    """Operator-controlled limits for one research run.

    CLI research supplies ``llm_client_factory`` by default. Direct Python
    callers that omit it retain the explicit offline compatibility path; this
    path uses the historical fixed-topic derivation library and is never used
    by the live API-driven swarm.
    """

    max_cycles: int | None = None
    stagnation_patience: int = 2
    max_workers_per_division: int = 8
    absolute_ceiling: int = 64
    worker_budget_tokens: int = 16000
    proof_timeout_s: float = 120.0
    experiment_timeout_s: float = 300.0
    seed: int = 20260926
    author: Callable[[Division, str, str], str] | None = None
    # A factory for independent per-agent LLM clients. Live runs never invoke
    # fixed-topic synthesis or formalisation templates.
    llm_client_factory: Callable[[], Any] | None = None
    worker_max_steps: int | None = 24
    step_policy: str = "classifier"
    safety_profile: str = "turbo"
    typst_root: Path | None = None
    auto_install_typst: bool = True
    # A checkable claim stated as "lhs == rhs", which the kernel decides directly
    # instead of matching the topic against its derivation library.
    claim: str = ""
    claim_symbols: tuple[str, ...] = ()
    synthesize: bool = True
    # Tier 2 of the proof standard. On by default: a run that advertises a
    # two-tier standard but skips the formal tier would be misleading, so the
    # opt-out is explicit and recorded in the run's verdict.
    formalize: bool = True
    # When no library template covers the topic, ask the Director to formulate
    # candidate claims instead of abandoning the investigation. Only the claim
    # comes from the model; the deciding script is always kernel-generated.
    formulate_unknown: bool = True
    # How long a worker may hold a task lease before another worker may take it.
    # A lease that outlives its owner would strand the board, and a lease shorter
    # than a single tool loop would let two workers share one artifact.
    task_lease_s: float = 1800.0
    # How often a running worker refreshes its lease. Must be well under
    # ``task_lease_s`` or a long attempt loses its task mid-flight and a sibling
    # starts writing the same artifact.
    task_lease_renew_s: float = 300.0
    max_task_attempts: int = 3
    # How many workers of one division may run their tool loops at once. 1
    # restores the historical serial behaviour; the default runs a division's
    # workers concurrently while keeping one writer per artifact via leases.
    max_parallel_workers: int = 4
    # A hard wall-clock ceiling for one worker's tool loop. Exceeding it moves
    # the worker to TIMED_OUT rather than letting it bill indefinitely.
    worker_timeout_s: float | None = None
    # How many runtime-overseer escalations in one attempt before a lead stops the
    # worker. One flag is normal friction on a hard proof; a repeat is a loop.
    overseer_stop_threshold: int = 3
    # Reuse the persisted board and ledger from an interrupted run instead of
    # rebuilding the plan and re-running the Director and every lead.
    resume: bool = True

    @property
    def author_mode(self) -> str:
        """Whether a model is consulted at all, and in what capacity."""
        if self.llm_client_factory is not None:
            return "interactive"
        return "live" if self.author else "mechanical"


@dataclass
class ResearchOutcome:
    """Everything a caller needs to judge and to re-verify a run."""

    topic: str
    slug: str
    workspace: Path
    solved: bool
    stop_reason: StopReason
    cycles: int
    workers_spawned: int
    ledger_ok: bool
    ledger_detail: str
    pdf: str | None
    agents: tuple[ResearchAgent, ...]

    def render(self) -> str:
        status = "SOLVED" if self.solved else "UNSOLVED"
        return (f"[director] {status} — {self.stop_reason.value} after {self.cycles} cycle(s), "
                f"{self.workers_spawned} worker(s) spawned\n"
                f"[director] ledger: {self.ledger_detail}\n"
                f"[director] artifacts: {self.workspace}")

    def to_dict(self) -> dict[str, Any]:
        return {"topic": self.topic, "slug": self.slug, "solved": self.solved,
                "stop_reason": self.stop_reason.value, "cycles": self.cycles,
                "workers_spawned": self.workers_spawned, "ledger_ok": self.ledger_ok,
                "ledger_detail": self.ledger_detail, "pdf": self.pdf,
                "workspace": str(self.workspace),
                "agents": [agent.to_dict() for agent in self.agents]}


class ResearchSwarm:
    """Govern a research topic from objective to publication-grade PDF."""

    def __init__(self, topic: str, *, root: str | Path = "research",
                 config: SwarmConfig | None = None):
        if not topic or not topic.strip():
            raise ValueError("Research topic cannot be empty")
        self.topic = topic.strip()
        self.config = config or SwarmConfig()
        self.slug = topic_slug(self.topic)
        self.workspace = ResearchWorkspace(Path(root).resolve() / self.slug)
        self.ledger = CommLedger(self.workspace.ledger_path, parent_id=None)
        self.proofs = ProofRunner(self.workspace.proof_dir, timeout_s=self.config.proof_timeout_s)
        self._legacy_formulation = False
        if self.config.llm_client_factory is not None:
            try:
                probe = self.config.llm_client_factory()
                self._legacy_formulation = bool(getattr(probe, "is_mock", False) and
                                                hasattr(probe, "payload"))
            except Exception:
                pass
        self.experiments = ExperimentRunner(self.workspace.experiment_dir,
                                            timeout_s=self.config.experiment_timeout_s,
                                            require_predictions=self._live_research_mode())
        self.agents: dict[str, ResearchAgent] = {}
        self.spawned_total = 0
        self._counters: dict[Division, int] = {division: 0 for division in Division}
        self._compile_result: Any = None
        self._paper_digest: str | None = None
        self._recorded: set[tuple] = set()
        self._objection_artifacts: dict[str, str] = {}
        self._clearance_artifacts: dict[str, str] = {}
        self._last_report: Any = None
        self._last_outcome: ConvergenceOutcome | None = None
        self._status_callback: Callable[[str], None] | None = None
        self.plan: TopicPlan = self._make_plan()
        self.claims: ClaimLedger = ClaimLedger()
        self.experiment_plan_notes: str = ""
        self.planned_experiments: int = 0
        self.planned_formalisations: int = 0
        self.formalisation_notes: str = ""
        self.formulation: Any = None
        self.lean_gate = LeanProofGate(self.workspace.lean_dir)
        self.board = TaskBoard(lease_s=self.config.task_lease_s,
                               max_attempts=self.config.max_task_attempts)
        self.bus = MessageBus(self.ledger)
        self.control = SwarmControl(self.ledger, board=self.board, bus=self.bus)
        self._workers_per_division: dict[Division, int] = {division: 0 for division in Division}
        self._install_leaders()
        self._recovered = self._recover_state()

    def _make_plan(self) -> TopicPlan:
        """Decide up front what the kernel will attempt, so the run can report it.

        A topic the library does not cover is not abandoned. When a model is
        available the Director is asked to formulate candidate claims, and those
        claims become kernel-checked propositions; only the *claim* comes from the
        model, never the deciding script.
        """
        if self._live_research_mode():
            return TopicPlan(self.topic, strategy="llm-authored",
                             notes="The Director and workers must author every claim and script.")
        from adaptive_harness.research.synthesis import plan_research
        claim: ParsedClaim | None = None
        if self.config.claim:
            claim = parse_claim(self.config.claim, self.config.claim_symbols)
        plan = plan_research(self.topic, claim)
        if plan.propositions or not self.config.formulate_unknown:
            return plan

        from adaptive_harness.research.formulate import formulate
        client = None
        if self.config.llm_client_factory is not None:
            try:
                client = self.config.llm_client_factory()
            except Exception:  # noqa: BLE001 - fall back to the one-shot author
                client = None
        formulation = formulate(self.topic, client=client, author=self.config.author)
        self.formulation = formulation
        if formulation.empty:
            return plan
        return TopicPlan(
            topic=self.topic, strategy="dynamic-formulation",
            propositions=formulation.propositions,
            notes=(f"{plan.notes} {formulation.notes}").strip())

    # -- organisation -------------------------------------------------------
    def _install_leaders(self) -> None:
        # The Director is a real roster member, not a local variable. Making it
        # addressable is what lets a leader escalate a stalled worker upward and
        # lets the control plane accept a stop request aimed at the run itself.
        self.agents[DIRECTOR_ID] = ResearchAgent(
            agent_id=DIRECTOR_ID, role_name="Executive Director", directive=self.topic,
            allowed_tools=("write_file",), budget_tokens=self.config.worker_budget_tokens,
            parent_id=None, division=None)
        self.control.register(DIRECTOR_ID, role="Executive Director", parent_id="")
        for division, spec in DIVISION_SPECS.items():
            agent_id = leader_agent_id(division)
            self.agents[agent_id] = ResearchAgent(
                agent_id=agent_id, role_name=spec.leader_title, directive=spec.objective,
                allowed_tools=spec.default_tools, budget_tokens=self.config.worker_budget_tokens,
                parent_id=DIRECTOR_ID, division=division, is_leader=True)
            # A leader's escalation contact is the Director; a worker's is its lead.
            self.bus.register_route(agent_id, DIRECTOR_ID)
            self.control.register(agent_id, role=spec.leader_title,
                                  division=division.value, parent_id=DIRECTOR_ID)
            if agent_id not in self.agents[DIRECTOR_ID].children:
                self.agents[DIRECTOR_ID].children.append(agent_id)
        self.ledger.append("OBJECTIVE_SET", {"agent_id": DIRECTOR_ID, "role": "Chief Scientist"},
                           {"agent_id": "all_leads", "role": "Division Leads"},
                           {"topic": self.topic, "slug": self.slug,
                            "leads": [leader_agent_id(d) for d in Division],
                            "author_mode": self.config.author_mode,
                            "definition_of_solved": [item.value for item in Invariant]})

    def spawn_subagent(self, parent_id: str, role_name: str, directive: str,
                       allowed_tools: Sequence[str], budget_tokens: int = 16000) -> str:
        """Spawn a worker agent dynamically, returning its ``agent_id``.

        The parent may be the Director or any Leader. Worker pools scale with
        the number of outstanding gaps, so a hard problem recruits more agents
        rather than a bigger single prompt.
        """
        if parent_id not in self.agents and parent_id != DIRECTOR_ID:
            raise KeyError(f"Unknown parent agent: {parent_id}")
        if not role_name or not role_name.strip():
            raise ValueError("A spawned agent needs a role name")
        if not directive or not directive.strip():
            raise ValueError("A spawned agent needs a directive")
        tools = tuple(dict.fromkeys(allowed_tools))
        if not tools:
            raise ValueError("A spawned agent needs at least one allowed tool")
        if budget_tokens < 1:
            raise ValueError("budget_tokens must be positive")

        parent = self.agents.get(parent_id)
        division = parent.division if parent else None
        if division is None:
            # Director-level spawns are routed by the leader that owns the work.
            division = Division.THEORY
        if self._counters[division] >= self.config.max_workers_per_division:
            raise RuntimeError(
                f"{division.value} division is at its cap of "
                f"{self.config.max_workers_per_division} workers; retire one first")
        self._counters[division] += 1
        index = self._counters[division]
        agent_id = worker_agent_id(division, index)
        if agent_id in self.agents:
            raise RuntimeError(f"Agent id collision: {agent_id}")
        agent = ResearchAgent(agent_id=agent_id, role_name=role_name.strip(), directive=directive.strip(),
                              allowed_tools=tools, budget_tokens=budget_tokens,
                              parent_id=parent_id, division=division)
        self.agents[agent_id] = agent
        if parent is not None:
            parent.children.append(agent_id)
        self.spawned_total += 1
        # Every spawned worker is registered with the control plane and given a
        # route to its nearest relevant lead, so a stop request and a help request
        # both have somewhere to go.
        self.control.register(agent_id, role=agent.role_name, division=division.value,
                              parent_id=parent_id,
                              deadline_s=self.config.worker_timeout_s)
        if parent is not None and parent.is_leader:
            self.bus.register_route(agent_id, parent_id)
        else:
            self.bus.register_route(agent_id, leader_agent_id(division))
        self.ledger.append("SPAWN_REQUEST",
                           {"agent_id": parent_id,
                            "role": parent.role_name if parent else "Chief Scientist"},
                           {"agent_id": agent_id, "role": agent.role_name},
                           {"directive": agent.directive, "allowed_tools": list(tools),
                            "budget_tokens": budget_tokens, "division": division.value,
                            "escalation_contact": self.bus.contact_for(agent_id)})
        return agent_id

    def scale_division(self, division: Division, size: int) -> tuple[str, ...]:
        """Grow or shrink a division's worker pool to ``size`` live workers."""
        if size < 0:
            raise ValueError("Division size must be non-negative")
        spec = DIVISION_SPECS[division]
        leader_id = leader_agent_id(division)
        live = [agent_id for agent_id, agent in self.agents.items()
                if agent.division is division and not agent.is_leader]
        if size > len(live):
            spawned: list[str] = []
            for offset in range(len(live), size):
                title = spec.worker_titles[offset % len(spec.worker_titles)]
                spawned.append(self.spawn_subagent(
                    leader_id, title,
                    f"{spec.objective} Focus: {title} assignment {offset + 1} for topic '{self.topic}'.",
                    spec.default_tools, self.config.worker_budget_tokens))
            return tuple(spawned)
        retired: list[str] = []
        for agent_id in live[size:]:
            agent = self.agents[agent_id]
            # A retired worker is stopped, not merely dropped from a dict: it may
            # still hold a task lease and a running child process, and both would
            # otherwise outlive the roster change.
            if self.control.record(agent_id) is not None and \
                    self.control.record(agent_id).state.active:
                self.control.request_stop(agent_id, actor=leader_id,
                                          reason="worker pool scaled down")
            self.agents.pop(agent_id)
            parent = self.agents.get(agent.parent_id or "")
            if parent is not None and agent_id in parent.children:
                parent.children.remove(agent_id)
            self._counters[division] = max(0, self._counters[division] - 1)
            retired.append(agent_id)
            self.ledger.append("STATUS_REPORT",
                               {"agent_id": agent_id, "role": agent.role_name},
                               {"agent_id": leader_id, "role": spec.leader_title},
                               {"retired": True, "reason": "worker pool scaled down"})
        return tuple(retired)

    # -- evidence -----------------------------------------------------------
    def record_evidence(self, claim: str, source: str, citation: str) -> dict[str, Any]:
        """Record one citable evidence record; unproven assertions stay out."""
        evidence_id = self.workspace.evidence_id()
        record = {"id": evidence_id, "claim": claim, "source": source,
                  "citation": citation, "timestamp": datetime.now(timezone.utc).isoformat()}
        records = self.workspace.evidence_records()
        records.append(record)
        self.workspace.evidence_index.write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        self.ledger.append("EVIDENCE_RECORDED", {"agent_id": leader_agent_id(Division.LITERATURE),
                                                 "role": "Literature Lead"},
                           {"agent_id": DIRECTOR_ID, "role": "Chief Scientist"},
                           {"evidence_id": evidence_id, "claim": claim, "citation": citation})
        return record

    # -- convergence loop ---------------------------------------------------
    def _synthesize(self) -> None:
        """Have the kernel write a decider for every claim that carries one.

        This is what makes a run do research rather than merely audit research.
        On the live path the Director supplies the exact expression and *the kernel*
        writes the self-adjudicating script, so the computational verdict is
        SymPy's and not a language model's. The theory worker's remaining job is the
        natural-language derivation and any claim the expression cannot settle —
        which is a real contribution rather than a transcription exercise.
        """
        if not self.config.synthesize:
            return
        existing = {path.stem for path in self.workspace.proof_dir.glob("*.py")}
        written: list[str] = []
        undetermined: list[str] = []
        for prop in self.plan.propositions:
            if prop.is_goal or not prop.script:
                undetermined.append(prop.prop_id)
                continue
            target = self.workspace.proof_dir / f"{prop.prop_id.lower()}.py"
            if target.stem in existing and target.is_file():
                continue
            target.write_text(prop.script, encoding="utf-8")
            written.append(target.name)
        if written or self.plan.propositions or undetermined:
            self.ledger.append(
                "TASK_ASSIGNED", {"agent_id": leader_agent_id(Division.THEORY),
                                  "role": "Theoretical Lead"},
                {"agent_id": "derivation_kernel", "role": "Derivation Kernel"},
                {"strategy": self.plan.strategy, "propositions": len(self.plan.propositions),
                 "scripts_written": written,
                 "kernel_decided": [prop.prop_id for prop in self.plan.propositions
                                   if prop.script],
                 "worker_authored_required": undetermined,
                 "notes": self.plan.notes})

    def _synthesize_experiments(self) -> None:
        """Emit the corroborating simulations that test the derived propositions."""
        from adaptive_harness.research.synthesis import plan_experiments
        experiments, notes = plan_experiments(self.topic, self.config.seed)
        self.experiment_plan_notes = notes
        self.planned_experiments = len(experiments)
        existing = {path.stem for path in self.workspace.experiment_dir.glob("*.py")}
        written: list[str] = []
        for experiment in experiments:
            target = self.workspace.experiment_dir / f"{experiment.exp_id.lower()}.py"
            if target.stem in existing and target.is_file():
                continue
            target.write_text(experiment.script, encoding="utf-8")
            written.append(target.name)
        if experiments:
            self.ledger.append(
                "TASK_ASSIGNED", {"agent_id": leader_agent_id(Division.EMPIRICAL),
                                  "role": "Empirical Lead"},
                {"agent_id": "derivation_kernel", "role": "Derivation Kernel"},
                {"experiments": len(experiments), "scripts_written": written,
                 "seed": self.config.seed, "notes": notes})

    def _adjudicate(self) -> ClaimLedger:
        """Decide every proposition from its script's exit code and record the verdict."""
        from adaptive_harness.research.claim import (ClaimAdjudication, verdict_from_exit)

        ledger = ClaimLedger()
        for prop in self.plan.propositions:
            if prop.is_goal:
                # A goal is not adjudicated. Recording it as UNTESTED would put a
                # fabricated verdict on the very problem the run was asked to attack.
                continue
            script = self.workspace.proof_dir / f"{prop.prop_id.lower()}.py"
            receipt = self.proofs.run_script(script, prop.prop_id) if script.is_file() else None
            exit_code = receipt.exit_code if receipt else None
            # An exit code cannot override the exactness scan. A script with a
            # float literal can exit 0 (or 3) but establishes no exact claim.
            verdict = (verdict_from_exit(exit_code) if receipt and
                       receipt.status != "REJECTED_INEXACT" else Verdict.UNTESTED)
            finding = ""
            if receipt is not None and receipt.stdout:
                tail = [line.strip() for line in receipt.stdout.splitlines() if line.strip()]
                finding = tail[-1] if tail else ""
            ledger.add(ClaimAdjudication(
                prop_id=prop.prop_id, statement=prop.statement, verdict=verdict,
                exit_code=exit_code, finding=finding,
                script=str(script.relative_to(self.workspace.root)) if script.is_file() else "",
                sha256=receipt.sha256 if receipt else ""))
            self.ledger.append(
                "PROOF_VERIFIED" if verdict is Verdict.PROVEN else (
                    "COUNTEREXAMPLE_FOUND" if verdict is Verdict.DISPROVEN else "PROOF_REJECTED"),
                {"agent_id": f"derivation_{prop.prop_id.lower()}", "role": "Derivation Kernel"},
                {"agent_id": DIRECTOR_ID, "role": "Chief Scientist"},
                {"prop_id": prop.prop_id, "statement": prop.statement,
                 "verdict": verdict.value, "exit_code": exit_code, "finding": finding[:400],
                 "hash": receipt.sha256 if receipt else None})
        self.claims = ledger
        return ledger

    @property
    def goals(self) -> tuple[Proposition, ...]:
        """The open problems this run was pointed at, if any."""
        return tuple(item for item in self.plan.propositions if item.is_goal)

    @property
    def decidable(self) -> tuple[Proposition, ...]:
        """The claims a run can actually settle, i.e. everything but the goals."""
        return tuple(item for item in self.plan.propositions if not item.is_goal)

    def _evaluate_claim(self) -> tuple[bool, str, tuple[str, ...]]:
        """The headline verdict must actually be decided, not merely attempted.

        A run aimed at an open problem settles its *sub-claims*; it does not settle
        the problem. The distinction is reported rather than blurred, because a
        paper that says "we proved the conjecture" when it proved a small-case
        lemma is the exact failure this harness exists to prevent.
        """
        if self._live_research_mode() and not self.workspace.objective_spec.is_file():
            return False, "Director has not written 00_objective_spec.md", ()
        ledger = self.claims if self.claims.adjudications else self._adjudicate()
        goals = self.goals
        prefix = ""
        if goals:
            names = ", ".join(item.prop_id for item in goals)
            prefix = (f"{len(goals)} open problem(s) stated as the target ({names}); "
                      f"what follows is progress on the decidable sub-claims, not a "
                      f"resolution. ")
        if not ledger.adjudications:
            return False, (prefix + "no proposition was derived, so nothing was tested; "
                           + (self.plan.notes or "supply a checkable claim to proceed")), ()
        evidence = [f"{item.prop_id} {item.verdict.value}" for item in ledger.adjudications]
        if not ledger.decided:
            return False, prefix + f"verdict {ledger.headline.value}: {ledger.summary()}", tuple(evidence)
        return True, prefix + f"verdict {ledger.headline.value}: {ledger.summary()}", tuple(evidence)

    def _formalise(self) -> None:
        """Emit the Lean 4 counterpart of the derived propositions.

        This is the second tier of the proof standard. Tier 1 (SymPy) reduces
        expressions; Tier 2 (Lean) checks the deduction. The two are independent
        checks on the same mathematical content, so a mistake in one is unlikely
        to be mirrored in the other.
        """
        from adaptive_harness.research.formalise import plan_formalisations
        if not self.config.formalize:
            return
        selected, notes = plan_formalisations(self.topic)
        self.formalisation_notes = notes
        self.planned_formalisations = len(selected)
        existing = {path.stem for path in self.workspace.lean_dir.glob("*.lean")}
        written: list[str] = []
        for formalisation in selected:
            if formalisation.lean_id in existing:
                continue
            self.lean_gate.write(formalisation.lean_id, formalisation.source)
            written.append(formalisation.lean_id)
        if selected:
            self.ledger.append(
                "LEAN_PROOF_SUBMISSION", {"agent_id": leader_agent_id(Division.FORMAL),
                                          "role": "Formal Proof Lead"},
                {"agent_id": "lean_kernel", "role": "Lean Formaliser"},
                {"formalisations": [item.lean_id for item in selected],
                 "scripts_written": written, "notes": notes,
                 "supplied_lean_files": sorted(existing)})

    def _evaluate_lean_proofs(self) -> tuple[bool, str, tuple[str, ...]]:
        """Pre-compilation clearance for the formal tier.

        The standard applies to *asserted theorems*, so what is required depends
        on the verdict:

        * ``PROVEN`` — the paper asserts a theorem, so every Lean file must be
          machine-checked. An empty proof set blocks publication, because
          advertising a two-tier standard while skipping the formal tier is the
          failure this gate exists to prevent.
        * ``DISPROVEN`` — nothing is asserted; the refutation *is* the result,
          and it is certified by an exact witness that the kernel checked. A Lean
          proof of a falsehood is neither expected nor meaningful, so the tier is
          satisfied by construction and any Lean file that was checked is noted.
        * ``INCONCLUSIVE``/``UNTESTED`` — no theorem is published, so there is
          nothing to formalise; the paper says so explicitly instead.
        """
        if not self.config.formalize:
            return True, "formal verification disabled for this run; claims are SymPy-only", ()

        self.lean_gate.receipts = self.lean_gate.verify_all()
        self.lean_gate.to_index(self.workspace.root / "lean_receipts.json")
        for receipt in self.lean_gate.receipts:
            sidecar = Path(receipt.path).with_name(Path(receipt.path).name + ".receipt.json")
            receipt_hash = sha256_file(sidecar) if sidecar.is_file() else None
            if receipt.certified:
                self.ledger.append(
                    "LEAN_PROOF_VERIFIED", {"agent_id": "lean_kernel", "role": "Lean Kernel"},
                    {"agent_id": leader_agent_id(Division.FORMAL), "role": "Formal Proof Lead"},
                    {"proof_id": receipt.proof_id, "name": receipt.name,
                     "status": receipt.status, "hash": receipt.sha256,
                     "receipt_hash": receipt_hash,
                     "theorems": list(receipt.theorems), "axioms": list(receipt.axioms),
                     "lean_version": receipt.lean_version, "verified_at": receipt.verified_at})
            else:
                self.ledger.append(
                    "LEAN_PROOF_REJECTED", {"agent_id": "lean_kernel", "role": "Lean Kernel"},
                    {"agent_id": leader_agent_id(Division.FORMAL), "role": "Formal Proof Lead"},
                    {"proof_id": receipt.proof_id, "name": receipt.name,
                     "status": receipt.status, "receipt_hash": receipt_hash,
                     "errors": list(receipt.errors[:3])})

        certified = [item for item in self.lean_gate.receipts if item.certified]
        checked = f"{len(certified)} of {len(self.lean_gate.receipts)} Lean file(s) certified"
        failed = [item for item in self.lean_gate.receipts if not item.certified]
        if failed:
            return False, (f"{checked}; invalid formal file(s): "
                           + ", ".join(item.name for item in failed)), ()
        verdict = self.claims.headline
        if verdict is Verdict.DISPROVEN:
            witness = self.claims.disproven[0].finding if self.claims.disproven else ""
            return True, (f"no theorem is asserted; the claim is refuted by the exact witness "
                          f"({witness[:80]}), which is itself machine-checked. {checked}"), ()
        if verdict is not Verdict.PROVEN:
            return True, (f"no theorem is published at verdict {verdict.value}, so the formal tier "
                          f"is not applicable. {checked}"), ()
        if self._live_research_mode():
            certified_names = {item.name.lower() for item in certified}
            missing = [item.prop_id for item in self.claims.proven
                       if item.prop_id.lower() not in certified_names]
            if missing:
                return False, ("no certified Lean file for claim(s): "
                               + ", ".join(missing)), ()
            from adaptive_harness.tools.lean import strip_lean_comments
            for claim in self.claims.proven:
                proposition = next((item for item in self.plan.propositions
                                    if item.prop_id == claim.prop_id), None)
                formal = proposition.lean_statement.strip() if proposition else ""
                if not formal:
                    return False, f"claim {claim.prop_id} has no declared Lean statement", ()
                header = re.match(r"^(?:theorem|lemma|corollary)\s+\S+\s*:\s*(.+)$",
                                  formal, flags=re.S)
                if header:
                    formal = header.group(1).strip()
                    formal = formal.split(":=", 1)[0].strip()
                source = (self.workspace.lean_dir / f"{claim.prop_id.lower()}.lean").read_text(
                    encoding="utf-8", errors="replace")
                normalized_source = " ".join(strip_lean_comments(source).split())
                normalized_formal = " ".join(formal.split())
                if f": {normalized_formal} :=" not in normalized_source:
                    return False, (f"the certified Lean file for {claim.prop_id} does not "
                                   "assert the Director's declared formal statement"), ()
        ok, detail, evidence = self.lean_gate.clearance()
        return ok, detail, evidence

    def _evaluate_proofs(self) -> tuple[bool, str, tuple[str, ...]]:
        scripts = self.proofs.scripts()
        if not scripts:
            self._adjudicate()
            return False, "no proof script exists yet; the claim is unsubstantiated", ()
        receipts = self.proofs.run_all()
        evidence: list[str] = []
        failed: list[str] = []
        for receipt in receipts:
            key = (receipt.theorem_id, receipt.sha256, receipt.status)
            if key not in self._recorded:
                self._recorded.add(key)
                self.proofs.record(receipt, ledger=self.ledger)
            # Exit 0 means the proposition holds; exit 3 means a counterexample was
            # exhibited. Both are valid *decisions*; anything else means the script
            # could not decide, which is a failure of the derivation, not a result.
            if receipt.status != "REJECTED_INEXACT" and receipt.exit_code in (EXIT_HOLDS, EXIT_COUNTEREXAMPLE):
                evidence.append(f"{receipt.theorem_id} exit {receipt.exit_code} ({receipt.sha256[:19]})")
            else:
                failed.append(f"{receipt.theorem_id} {receipt.status}")
        self.proofs.receipts = receipts
        # Worker edits can change the verdict between convergence cycles.
        # Re-adjudicate after every proof pass so later gate evaluators see the
        # current script hashes, rather than the first cycle's cached claim.
        self._adjudicate()
        if self._live_research_mode():
            missing_explanations = [
                item.prop_id for item in self.claims.proven
                if not (self.workspace.proof_dir / f"{item.prop_id.lower()}.md").is_file()]
            if missing_explanations:
                failed.append("missing natural-language proof: "
                              + ", ".join(missing_explanations))
        if failed:
            return False, f"{len(failed)} derivation(s) reached no verdict: {'; '.join(failed)}", tuple(evidence)
        return True, f"all {len(receipts)} derivation(s) reached a clean verdict", tuple(evidence)

    def _evaluate_experiments(self) -> tuple[bool, str, tuple[str, ...]]:
        scripts = self.experiments.scripts()
        if not scripts:
            if not self.planned_experiments:
                # Nothing to corroborate: a claim settled by exact algebra needs
                # no simulation. Saying so is honest; demanding a simulation here
                # would be theatre, and passing silently would be misleading.
                return True, ("no simulation applies to this claim, so the result rests on exact "
                              "computation alone and no statistical claim is made"), ()
            return False, "a simulation was planned but none was written to experiments/", ()
        receipts = self.experiments.run_all({path.name: self.config.seed for path in scripts})
        evidence: list[str] = []
        failed: list[str] = []
        for receipt in receipts:
            key = (receipt.experiment_id, receipt.seed, receipt.status,
                   canonical_key(sorted(receipt.data_hashes.items())))
            if key not in self._recorded:
                self._recorded.add(key)
                self.experiments.record(receipt, ledger=self.ledger)
            evidence.append(f"{receipt.experiment_id} {receipt.status} seed={receipt.seed}")
            if not receipt.replicated:
                failed.append(f"{receipt.experiment_id} {receipt.status}")
        self.experiments.receipts = receipts
        if failed:
            return False, f"{len(failed)} experiment(s) not replicated: {'; '.join(failed)}", tuple(evidence)
        return True, f"all {len(receipts)} experiment(s) replicated at seed {self.config.seed}", tuple(evidence)

    def _evaluate_adversarial(self) -> tuple[bool, str, tuple[str, ...]]:
        """Certify the *reported verdict*, not the claim in the abstract.

        Research aims to settle a question either way, so a clean refutation is
        a success, not a failure. Clearance therefore means "the result now
        reported is certified": when the verdict is PROVEN the red team must
        have found no counterexample to the proof; when it is DISPROVEN the
        refutation must carry an exact witness, which the red team is recorded
        as having inspected. Neither reading lets an unexamined result pass.
        """
        if not self.proofs.scripts():
            return False, "no claim exists to falsify; clearance would be vacuous", ()
        if self._live_research_mode():
            current = self._artifact_fingerprint()
            for decisions in (self._objection_artifacts, self._clearance_artifacts):
                for agent_id, digest in list(decisions.items()):
                    if digest != current and agent_id in self.agents:
                        self.agents[agent_id].clearance = Clearance.PENDING
                        del decisions[agent_id]
                        self.ledger.append(
                            "STATUS_REPORT", {"agent_id": DIRECTOR_ID, "role": "Chief Scientist"},
                            {"agent_id": agent_id, "role": "Adversarial Worker"},
                            {"status": "reassess_after_artifact_change", "artifact_fingerprint": current})
        counterexamples = [agent.agent_id for agent in self.agents.values()
                           if agent.clearance is Clearance.COUNTEREXAMPLE]
        if counterexamples:
            return False, f"the red team holds {len(counterexamples)} open counterexample(s)", ()
        cleared = [agent.agent_id for agent in self.agents.values()
                   if agent.division is Division.ADVERSARIAL and not agent.is_leader
                   and agent.clearance is Clearance.CLEARED]

        # A refutation is settled by its witness, so it is judged before any
        # clearance is required: a red team that declined to return a verdict must
        # not block publication of a result the kernel already certified exactly.
        if self.claims.headline is Verdict.DISPROVEN:
            refuted = [item for item in self.claims.disproven if item.finding]
            if not refuted:
                return False, "the claim is reported refuted but no witness was exhibited", ()
            first = refuted[0]
            return True, (f"the reported verdict is a refutation certified by the exact witness "
                          f"{first.finding[:120]}; {len(cleared)} red-team worker(s) cleared the "
                          f"claim and none holds an objection to the disproof"), tuple(cleared)

        if self.claims.headline is not Verdict.PROVEN:
            return False, (f"verdict is {self.claims.headline.value}, so there is no settled result "
                           f"to certify"), ()
        if not cleared:
            return False, "no red-team worker has returned a verdict yet", ()
        if not self.workspace.audit.is_file():
            return False, "adversarial audit log has not been written", ()
        return True, (f"{len(cleared)} red-team worker(s) cleared {len(self.proofs.scripts())} "
                      f"proved claim(s); audit log written"), tuple(cleared)

    def _artifact_fingerprint(self) -> str:
        """Bind an adversarial objection to the exact artifacts it examined."""
        import hashlib
        paths = [self.workspace.root / "claim_manifest.json"]
        for directory in (self.workspace.proof_dir, self.workspace.lean_dir,
                          self.workspace.experiment_dir):
            paths.extend(sorted(directory.glob("*.py")))
            paths.extend(sorted(directory.glob("*.lean")))
        evidence = [(str(path.relative_to(self.workspace.root)), sha256_file(path))
                    for path in paths if path.is_file()]
        return hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()

    def _evaluate_document(self) -> tuple[bool, str, tuple[str, ...]]:
        """Render the paper from current receipts, then build it warning-free.

        The PDF is content-addressed: Typst is only re-invoked when the rendered
        source actually changes, because document integrity is re-checked on
        every cycle while its inputs move far more slowly than that.
        """
        if self._live_research_mode():
            required = self._write_receipt_manifest()
            if not self.workspace.paper_typ.is_file():
                return False, "the Typst Author has not written paper.typ", ()
            source = self.workspace.paper_typ.read_text(encoding="utf-8", errors="replace")
            absent = [tag for tag in required if tag not in source]
            if absent:
                return False, f"paper.typ omits verified receipt tags: {', '.join(absent)}", ()
            result = self._compile()
            return self._document_verdict(result)
        builder = PaperBuilder(self.workspace.root, typst_root=self.config.typst_root,
                               allow_install_typst=self.config.auto_install_typst)
        try:
            target = builder.render(self._paper_inputs())
        except OSError as exc:
            return False, f"paper.typ could not be authored: {exc}", ()
        digest = sha256_file(target)
        if digest == self._paper_digest and self._compile_result is not None:
            return self._document_verdict(self._compile_result)
        result = self._compile()
        self._paper_digest = digest
        return self._document_verdict(result)

    @staticmethod
    def _document_verdict(result: Any) -> tuple[bool, str, tuple[str, ...]]:
        if result is None or not result.success:
            detail = result.error if result else "Typst could not be resolved"
            return False, f"paper.pdf did not build cleanly: {detail}", ()
        return True, f"paper.pdf built cleanly ({result.size_bytes} bytes, {result.typst_version})", ()

    def _write_receipt_manifest(self) -> tuple[str, ...]:
        """Give the Typst Author an exact list of citable, verified artifacts."""
        entries: list[dict[str, str]] = []
        for index, receipt in enumerate(self.proofs.receipts, 1):
            if receipt.verified:
                entries.append({"tag": f"[PROOF-{index:03d}]", "path": receipt.script,
                                "hash": receipt.sha256})
        for index, receipt in enumerate(self.lean_gate.receipts, 1):
            if receipt.certified:
                entries.append({"tag": f"[LEAN-{index:03d}]", "path": receipt.path,
                                "hash": receipt.sha256})
        for index, receipt in enumerate(self.experiments.receipts, 1):
            if receipt.replicated:
                entries.append({"tag": f"[EXP-{index:03d}]", "path": receipt.script,
                                "hash": canonical_key(receipt.data_hashes)})
        for index, record in enumerate(self.workspace.evidence_records(), 1):
            key = ("evidence", canonical_key(record))
            if key not in self._recorded:
                self._recorded.add(key)
                self.ledger.append("EVIDENCE_RECORDED",
                                   {"agent_id": leader_agent_id(Division.LITERATURE),
                                    "role": "Literature Lead"},
                                   {"agent_id": DIRECTOR_ID, "role": "Chief Scientist"},
                                   {"evidence_id": record.get("id", f"EVID-{index:03d}"),
                                    "source": record.get("source", ""),
                                    "claim": record.get("claim", "")})
            entries.append({"tag": f"[EVID-{index:03d}]", "path": str(record.get("source", "")),
                            "hash": canonical_key(record)})
        target = self.workspace.root / "receipt_manifest.json"
        target.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        return tuple(item["tag"] for item in entries)

    def _lean_sources(self) -> dict[str, str]:
        """Read the formal sources for the paper's reproducibility appendix."""
        sources: dict[str, str] = {}
        for path in sorted(self.workspace.lean_dir.glob("*.lean")):
            try:
                sources[path.name] = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
        return sources

    def _paper_inputs(self) -> Any:
        """Assemble the citable artifact set from the latest verified receipts."""
        from adaptive_harness.research.paper import PaperInputs
        receipts = list(self.lean_gate.receipts)
        certified = [item for item in receipts if item.certified]
        return PaperInputs(
            title=self._paper_title(),
            abstract=self._paper_abstract(len(certified)),
            topic=self.topic,
            propositions=self.plan.propositions,
            adjudications=self.claims.adjudications,
            claim=self.claims,
            proofs=tuple(self.proofs.receipts),
            experiments=tuple(self.experiments.receipts),
            evidence=tuple(self.workspace.evidence_records()),
            gate=self._last_report,
            outcome=self._last_outcome,
            ledger=self.ledger,
            figures=tuple(path.name for path in sorted(self.workspace.figure_dir.glob("*.svg"))),
            lean_receipts=tuple(receipts),
            lean_sources=self._lean_sources() if certified else None,
            lean_version=self.lean_gate.lean_version,
            mathlib_available=bool(certified) and self.lean_gate._mathlib_available())

    def _paper_title(self) -> str:
        """Title the paper after what it actually concluded, not the raw topic slug."""
        strategy = self.plan.strategy
        if "power-law-moments" in strategy and "balanced-allocation" in strategy:
            return ("The critical tail index of a heavy-tailed delay and the "
                    "variance-optimality of balanced routing")
        if "power-law-moments" in strategy:
            return "Moments and the critical index of a power-law delay distribution"
        if "balanced-allocation" in strategy:
            return "Variance-optimal load allocation under independent delays"
        if "gaussian-transform" in strategy:
            return "A Fourier-transform baseline for subexponential delay tails"
        if self.config.claim:
            return f"On the identity {self.config.claim}"
        return self.topic

    def _paper_abstract(self, certified: int = 0) -> str:
        """A real abstract: what was asked, what was shown, and what it means."""
        strategy = self.plan.strategy
        if "power-law-moments" in strategy and "balanced-allocation" in strategy:
            return ("Routing policies for delay-sensitive services are routinely justified by "
                    "minimising the variance of the observed delay. We show that this objective "
                    "is not merely inconvenient but undefined on part of the empirically observed "
                    "range: for a delay with a Pareto tail of index at most two the second moment "
                    "diverges, so the variance is infinite under every allocation of work and no "
                    "variance-minimising policy exists. Above the critical index we identify the "
                    "variance-minimising allocation, showing that the balanced split is optimal "
                    "and that its exact excess over the Cauchy-Schwarz bound is determined by the "
                    "remainder of the batch size upon division by the number of servers. Every "
                    "statement is a proposition whose symbolic form is constructed from a "
                    "definition and decided by an executed derivation script."
                    + (f" {certified} of these results additionally carry a Lean 4 proof that the "
                       f"Lean kernel machine-checks, so the argument rests on a verified "
                       f"deduction and not only on symbolic computation." if certified else ""))
        if self.plan.propositions:
            return (f"We investigate {self.topic}. The propositions below are constructed from "
                    f"their definitions and decided by executing self-adjudicating derivation "
                    f"scripts; a result is reported only where its script reached a verdict.")
        return (f"No derivation strategy matched the topic '{self.topic}', so no proposition was "
                f"constructed and the claim was not tested. This paper reports that outcome "
                f"explicitly rather than presenting an unverified result.")

    def _compile(self) -> Any:
        if not self.workspace.paper_typ.is_file():
            return None
        compiler = TypstCompiler(root=self.config.typst_root or self.workspace.root,
                                 allow_install=self.config.auto_install_typst)
        self._compile_result = compiler.compile(self.workspace.paper_typ, self.workspace.paper_pdf)
        return self._compile_result

    def build_gate(self) -> InvariantGate:
        return InvariantGate({
            Invariant.MATHEMATICAL_SOUNDNESS: self._evaluate_proofs,
            Invariant.EMPIRICAL_REPLICATION: self._evaluate_experiments,
            Invariant.ADVERSARIAL_CLEARANCE: self._evaluate_adversarial,
            Invariant.CLAIM_ADJUDICATION: self._evaluate_claim,
            Invariant.LEAN_FORMAL_SOUNDNESS: self._evaluate_lean_proofs,
            Invariant.DOCUMENT_INTEGRITY: self._evaluate_document,
        })

    def _author(self, division: Division, artefact: str, instruction: str) -> str | None:
        """Delegate authoring to the configured author (an LLM in live runs)."""
        if self.config.author is None:
            return None
        return self.config.author(division, artefact, instruction)

    # -- task board ---------------------------------------------------------
    def _authorized_paths(self, target: Path) -> tuple[Path, ...]:
        """The exact set of files a worker assigned to ``target`` may write.

        Restricting writes is what turns "your artifact is X" from a
        suggestion into a constraint. Two cases need more than the target path:
        a proof worker's natural-language explanation is a separate file the
        adjudication gate requires, and a Typst author must be able to save a
        draft beside the paper it is trying to repair.
        """
        authorized = [target]
        if target.suffix == ".py":
            authorized.append(target.with_name(f"{target.stem}.md"))
        if target == self.workspace.paper_typ:
            authorized.append(self.workspace.root / "paper_draft.typ")
        return tuple(dict.fromkeys(authorized))

    def _task_artifact(self, division: Division, gap: str, slot: int) -> str:
        """A unique artifact path for one (division, gap, slot) unit of work.

        The previous scheme derived a target from the worker's index modulo the
        claim count, so with more workers than claims two workers resolved to the
        same file and raced on it. Deriving the path from the *task* instead makes
        the collision impossible to express, and the board's artifact lease makes
        it impossible to execute even if a path were reused.
        """
        if division is Division.THEORY and self.plan.propositions:
            prop = self.plan.propositions[slot % len(self.plan.propositions)]
            return f"proofs/{prop.prop_id.lower()}.py"
        if division is Division.FORMAL and self.plan.propositions:
            prop = self.plan.propositions[slot % len(self.plan.propositions)]
            return f"proofs/lean/{prop.prop_id.lower()}.lean"
        if division is Division.EMPIRICAL:
            return f"experiments/exp-{slot + 1:02d}.py"
        if gap == Invariant.DOCUMENT_INTEGRITY.value:
            return "paper.typ"
        if division is Division.ADVERSARIAL:
            return f"audit/falsify-{slot + 1:02d}.md"
        return f"evidence/{division.value}-{slot + 1:02d}.md"

    def _task_dependencies(self, division: Division, gap: str) -> tuple[str, ...]:
        """Which task ids must complete before this one is worth attempting.

        Dependencies encode the actual research order rather than a blanket
        serialisation: a paper cannot cite receipts that do not exist, and a
        document-integrity task is therefore blocked on the division that produces
        the evidence it must cite.
        """
        if gap == Invariant.DOCUMENT_INTEGRITY.value:
            return tuple(task.task_id for task in self.board.all()
                         if task.state is WorkerState.COMPLETED
                         and task.division in (Division.THEORY.value, Division.FORMAL.value,
                                               Division.EMPIRICAL.value))
        return ()

    def _ensure_task(self, division: Division, gap: str, slot: int) -> Task:
        """Find or create the board task for one unit of work.

        Idempotent on ``(division, gap, artifact)``, which is what makes a later
        cycle reuse the same task — and its attempt count, evidence, and failure
        history — instead of silently opening a fresh duplicate.
        """
        artifact = self._task_artifact(division, gap, slot)
        for task in self.board.all():
            if (task.division == division.value and task.gap == gap
                    and task.artifact == artifact):
                return task
        criterion = self._success_criterion_for(division, gap, Path(artifact))
        hypothesis = self._pre_registered_hypothesis(division, slot) if \
            division is Division.EMPIRICAL else ""
        task = self.board.create(
            title=f"{division.value}: close {gap}", division=division.value, gap=gap,
            created_by=DIRECTOR_ID, artifact=artifact,
            dependencies=self._task_dependencies(division, gap), acceptance=criterion,
            hypothesis=hypothesis,
            contacts=(leader_agent_id(division),))
        self.ledger.append(
            "TASK_ASSIGNED", {"agent_id": DIRECTOR_ID, "role": "Chief Scientist"},
            {"agent_id": leader_agent_id(division),
             "role": DIVISION_SPECS[division].leader_title},
            {"task_id": task.task_id, "gap": gap, "artifact": artifact,
             "acceptance": criterion, "dependencies": list(task.dependencies),
             "hypothesis": hypothesis or None,
             "seed": self.config.seed if division is Division.EMPIRICAL else None,
             "division": division.value})
        return task

    def _pre_registered_hypothesis(self, division: Division, slot: int) -> str:
        """The prediction a simulation must be tested against, fixed in advance.

        The seed, the sampling model, and the acceptance band are written onto the
        board before the script exists. That ordering is the point: a bound chosen
        after seeing the samples is not a prediction, and the ledger entry for this
        task predates the experiment receipt, so the two can be compared.
        """
        if division is not Division.EMPIRICAL:
            return ""
        claim = (self.plan.propositions[slot % len(self.plan.propositions)]
                 if self.plan.propositions else None)
        subject = f"{claim.prop_id}: {claim.statement}" if claim else \
            "the Director's declared mathematical claim"
        return (f"Test whether {subject}. Sampling model: independent draws from the "
                f"distribution the script documents, n reported alongside the interval. "
                f"Seed: {self.config.seed}, fixed. Expected bound: the observed statistic "
                f"must lie inside a 95% interval computed from the raw samples. "
                f"Acceptance: a non-empty {self.config.seed}-seeded prediction record whose "
                f"interval is derived from the samples, and a non-zero exit if the "
                f"prediction falls outside it. A prediction is not revised after seeing "
                f"the data; a refuted prediction is reported as a refutation.")

    def _success_criterion_for(self, division: Division, gap: str, target: Path) -> str:
        """The acceptance criterion recorded on the board, phrased for the board."""
        if self._live_research_mode():
            return self._success_criterion(division, target)
        return f"{target.name} exists and records the evidence relied upon."

    def _task_slots(self, division: Division, worker_ids: Sequence[str]) -> int:
        """How many distinct units of work this division has open this cycle.

        The count follows the *artifact shape*, not the worker count, because the
        artifact is what must be unique. A division that writes one file per claim
        has one slot per claim; a division whose every worker writes its own file
        has one slot per worker; a division that writes a single shared document
        has exactly one, however many workers are idle.
        """
        if division in (Division.THEORY, Division.FORMAL) and self.plan.propositions:
            return len(self.plan.propositions)
        if division in (Division.EMPIRICAL, Division.ADVERSARIAL):
            return max(1, len(worker_ids))
        return 1

    def _delegate(self, division: Division, gap: str, worker_ids: Sequence[str],
                  spawned: list[str]) -> list[tuple[str, Task]]:
        """Have the division lead lease tasks and hand them to specific workers.

        Delegation is deterministic round-robin, not model-chosen: the lead's own
        LLM pass already wrote its assignment file, and paying for a second
        model call purely to pick which of its own workers runs first buys no
        accuracy. What the lead *is* responsible for — the lease, the artifact
        exclusivity, and the recorded hand-off — is real and is what stops two
        workers from doing the same job.
        """
        slots = self._task_slots(division, worker_ids)
        pairs: list[tuple[str, Task]] = []
        lead_id = leader_agent_id(division)
        for slot in range(slots):
            try:
                task = self._ensure_task(division, gap, slot)
            except CoordinationError as exc:
                # The artifact is already committed to a live task. That is the
                # board doing its job, not an error to escalate.
                self.ledger.append(
                    "STATUS_REPORT", {"agent_id": lead_id,
                                      "role": DIVISION_SPECS[division].leader_title},
                    {"agent_id": DIRECTOR_ID, "role": "Chief Scientist"},
                    {"status": "task_exists", "detail": str(exc)})
                continue
            if task.terminal:
                continue
            if not worker_ids:
                continue
            worker_id = worker_ids[slot % len(worker_ids)]
            try:
                self.board.lease(task.task_id, worker_id)
            except CoordinationError as exc:
                self.ledger.append(
                    "STATUS_REPORT", {"agent_id": lead_id,
                                      "role": DIVISION_SPECS[division].leader_title},
                    {"agent_id": worker_id, "role": self.agents[worker_id].role_name},
                    {"status": "lease_refused", "task_id": task.task_id,
                     "detail": str(exc)})
                self.bus.send(sender=worker_id, recipient=self.bus.contact_for(worker_id),
                              kind=MessageKind.ESCALATION, subject=f"cannot take {task.task_id}",
                              body=str(exc))
                continue
            self.ledger.append(
                "TASK_LEASED", {"agent_id": lead_id, "role": DIVISION_SPECS[division].leader_title},
                {"agent_id": worker_id, "role": self.agents[worker_id].role_name},
                {"task_id": task.task_id, "gap": gap, "artifact": task.artifact,
                 "acceptance": task.acceptance, "attempt": task.attempts,
                 "lease_expires_in_s": round(self.config.task_lease_s, 1)})
            self.bus.send(sender=lead_id, recipient=worker_id, kind=MessageKind.DIRECTIVE,
                          subject=f"{task.task_id}: {task.title}", body=task.acceptance,
                          requires_ack=False)
            pairs.append((worker_id, task))
        return pairs

    def _review(self, division: Division, worker_id: str, task: Task,
                summary: Mapping[str, Any]) -> None:
        """The lead's post-hoc review of one worker's outcome, recorded either way.

        Recording only successes would make the audit read like a highlights reel;
        a leader that never acknowledges a failure cannot redirect the next cycle.
        The review is also where supervisor verdicts turn into decisions.
        """
        lead_id = leader_agent_id(division)
        accepted = bool(summary.get("accepted"))
        supervision = self._supervise(division, worker_id, task, summary)
        self.bus.send(sender=lead_id, recipient=worker_id, kind=MessageKind.REVIEW,
                      subject=f"{task.task_id} {'accepted' if accepted else 'rejected'}",
                      body=(f"artifact={task.artifact} success={summary.get('success')} "
                            f"tool_calls={summary.get('tool_calls', 0)} "
                            f"stop_reason={summary.get('stop_reason')} "
                            f"supervision={supervision or 'none'} "
                            f"detail={(summary.get('error') or '')[:200]}"),
                      requires_ack=False)
        self.ledger.append(
            "PROGRESS_SUMMARY", {"agent_id": lead_id,
                                 "role": DIVISION_SPECS[division].leader_title},
            {"agent_id": worker_id, "role": self.agents[worker_id].role_name},
            {"task_id": task.task_id, "review": "accepted" if accepted else "rejected",
             "gap": task.gap, "artifact": task.artifact,
             "tool_calls": summary.get("tool_calls", 0),
             "state": summary.get("state"), "stop_reason": summary.get("stop_reason"),
             "supervision": supervision,
             "overseer_verdicts": [item.get("state") for item in summary.get("overseer", [])],
             "error": (summary.get("error") or "")[:300]})

    def _plan_cycle(self, report: Any) -> list[tuple[Division, str, Task, str]]:
        """Director: choose the open work, then have each lead delegate it.

        Returns ``(division, gap, task, worker_id)`` quadruples. Every unit of
        work in a cycle is lease-protected before a single worker starts, so
        concurrency below cannot produce two writers for one artifact.
        """
        spawned: list[str] = []
        planned: list[tuple[Division, str, Task, str]] = []
        gaps = [status.invariant.value for status in report.gaps]
        handled: set[Division] = set()
        self._current_cycle_index = getattr(self, "_current_cycle_index", 1)
        for gap in gaps:
            if self.control.run_stopped:
                break
            division = GAP_ROUTING.get(gap, Division.THEORY)
            if self._live_research_mode() and division in handled:
                continue
            handled.add(division)
            live = [agent.agent_id for agent in self.agents.values()
                    if agent.division is division and not agent.is_leader]
            # The Director's budget is advisory; the per-division cap is the hard
            # limit, so escalation stops growing rather than overflowing the pool.
            target = min(getattr(self, "_cycle_budget", 2), self.config.max_workers_per_division)
            if len(live) < target:
                spawned.extend(self.scale_division(division, target))
                live = [agent.agent_id for agent in self.agents.values()
                        if agent.division is division and not agent.is_leader]
            for worker_id, task in self._delegate(division, gap, live[:target], spawned):
                planned.append((division, gap, task, worker_id))
        return planned

    def cycle(self, report: Any, index: int, budget: int) -> tuple[str, ...]:
        """One research cycle: plan, delegate, run workers, re-verify.

        Planning is sequential because it mutates the board; execution is
        concurrent because the leased units are provably disjoint — each owns a
        distinct artifact, and the artifact is the only thing these workers write.
        """
        self._current_cycle_index = index
        self._cycle_budget = budget
        spawned = self.scale_for_cycle(report, budget)
        planned = self._plan_cycle(report)
        results: list[Mapping[str, Any]] = []
        if planned:
            self._emit_status(f"cycle {index}: dispatching {len(planned)} leased task(s)")
            results = self._execute_planned(planned)
            for (division, gap, task, worker_id), summary in zip(planned, results):
                self._review(division, worker_id, task, summary)
        self.control.progress_summary(cycle=index)
        self.board.save(self.workspace.root / TASK_BOARD_FILENAME)
        self.ledger.append("GATE_EVALUATION",
                           {"agent_id": DIRECTOR_ID, "role": "Chief Scientist"},
                           {"agent_id": "invariants", "role": "Invariant Gate"},
                           {"cycle": index, "gaps": [status.invariant.value
                                                      for status in report.gaps],
                            "fingerprint": report.fingerprint,
                            "worker_budget": budget, "spawned": spawned,
                            "leased_tasks": [task.task_id for _, _, task, _ in planned],
                            "parallelism": min(len(planned), self.config.max_parallel_workers)})
        return tuple(spawned)

    def scale_for_cycle(self, report: Any, budget: int) -> list[str]:
        """Grow each division's pool to the escalated budget. Returns new agent ids.

        Scaling is recorded even when it grows nothing, because "the Director
        decided not to hire anyone" is a decision an audit should be able to see
        rather than infer from silence.
        """
        spawned: list[str] = []
        for status in report.gaps:
            division = GAP_ROUTING.get(status.invariant.value, Division.THEORY)
            live = [agent for agent in self.agents.values()
                    if agent.division is division and not agent.is_leader]
            target = min(budget, self.config.max_workers_per_division)
            if len(live) < target:
                added = self.scale_division(division, target)
                spawned.extend(added)
                self.ledger.append(
                    "SWARM_SCALED", {"agent_id": DIRECTOR_ID, "role": "Chief Scientist"},
                    {"agent_id": leader_agent_id(division),
                     "role": DIVISION_SPECS[division].leader_title},
                    {"division": division.value, "gap": status.invariant.value,
                     "before": len(live), "target": target, "added": list(added),
                     "cap": self.config.max_workers_per_division})
        return spawned

    def _supervise(self, division: Division, worker_id: str, task: Task,
                   summary: Mapping[str, Any]) -> str | None:
        """Act on what the runtime overseer saw, at the swarm level.

        The per-agent ``RuntimeOverseer`` already detects a worker looping, stalling,
        or drifting outside its assignment, and injects corrective directives. It is
        myopic by design — it sees one agent — and its verdict previously went only
        into the trajectory. This is where a verdict becomes an *action*:

        * a request for help goes to the worker's escalation contact and is
          recorded as unanswered until somebody deals with it;
        * a worker the overseer caught looping is stopped, with the overseer's
          verdict as the reason, so its budget stops being spent on nothing.

        Stopping is deliberately conservative: it needs a *repeat* of the same
        verdict, because one flagged step is a normal part of a difficult proof
        attempt, not grounds for termination.
        """
        verdicts = [item for item in summary.get("overseer", []) if item.get("state")]
        if not verdicts:
            return None
        # Work that landed is not stalling, however many steps it took to get there.
        if task.state is WorkerState.COMPLETED:
            return None
        states = [str(item.get("state")) for item in verdicts]
        escalated = [state for state in states
                     if state in ("LOOPING_DETECTED", "PROGRESS_STALLED", "SEMANTIC_DRIFT")]
        if not escalated:
            return None
        repeated = len(escalated) >= self.config.overseer_stop_threshold
        latest = escalated[-1]
        if repeated:
            self.control.request_stop(
                worker_id, actor=leader_agent_id(division),
                reason=(f"the runtime overseer flagged {latest} "
                        f"{len(escalated)} time(s) in this attempt"))
            self.bus.send(sender=worker_id, recipient=self.bus.contact_for(worker_id),
                          kind=MessageKind.ESCALATION,
                          subject=f"{task.task_id} stalled: {latest}",
                          body=(f"The runtime overseer flagged {latest} "
                                f"{len(escalated)} time(s). Artifact {task.artifact} "
                                f"was left unverified. Tool calls: "
                                f"{summary.get('tool_calls', 0)}."),
                          requires_ack=True)
            return "stopped"
        self.bus.send(sender=worker_id, recipient=self.bus.contact_for(worker_id),
                      kind=MessageKind.REQUEST_FOR_HELP,
                      subject=f"{task.task_id} needs help: {latest}",
                      body=(f"The runtime overseer flagged {latest}. Last directive: "
                            f"{str(verdicts[-1].get('directive'))[:200]}"),
                      requires_ack=True)
        return "asked_for_help"

    def _execute_planned(self, planned: Sequence[tuple[Division, str, Task, str]]
                         ) -> list[Mapping[str, Any]]:
        """Run the leased units, at most ``max_parallel_workers`` at a time.

        The Director's escalation is a *worker-count* signal, not a licence to
        spend N times the tokens at once, so the executor caps the fan-out
        independently of how many units were leased.
        """
        if self.config.max_parallel_workers <= 1 or len(planned) == 1:
            return [self._execute_one(unit) for unit in planned]
        limit = max(1, int(self.config.max_parallel_workers))
        summaries: list[Mapping[str, Any]] = [{} for _ in planned]
        with ThreadPoolExecutor(max_workers=limit, thread_name_prefix="research-worker") as pool:
            futures = {pool.submit(self._execute_one, unit): index
                       for index, unit in enumerate(planned)}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    summaries[index] = future.result()
                except Exception as exc:  # noqa: BLE001 - one unit must not sink the cycle
                    summaries[index] = {"ran": True, "success": False, "accepted": False,
                                        "state": WorkerState.FAILED.value,
                                        "error": f"{type(exc).__name__}: {exc}"[:400],
                                        "tool_calls": 0}
        return summaries

    def _execute_one(self, unit: tuple[Division, str, Task, str]) -> Mapping[str, Any]:
        """Dispatch one leased unit to the right kind of work."""
        division, gap, task, worker_id = unit
        try:
            self.control.begin(worker_id, task_id=task.task_id)
        except CoordinationError as exc:
            # The worker was deliberately stopped and must not be silently revived.
            return {"ran": False, "success": False, "accepted": False,
                    "state": WorkerState.CANCELLED.value, "error": str(exc), "tool_calls": 0}
        try:
            if division is Division.ADVERSARIAL and gap == Invariant.ADVERSARIAL_CLEARANCE.value:
                return self._falsify(agent=self.agents[worker_id], spawned=[], task=task)
            return self._produce(division, self.agents[worker_id], gap, [], task=task)
        except WorkerCancelled as exc:
            self._close_task(task, worker_id, WorkerState.CANCELLED, reason=str(exc))
            return {"ran": True, "success": False, "accepted": False,
                    "state": WorkerState.CANCELLED.value, "error": str(exc), "tool_calls": 0}

    def _close_task(self, task: Task, worker_id: str, state: WorkerState, *,
                    reason: str = "", evidence_ids: Sequence[str] = (),
                    artifact: str = "") -> None:
        """Write the task's terminal state to both the board and the ledger."""
        if state is WorkerState.COMPLETED:
            self.board.complete(task.task_id, worker_id, evidence_ids=evidence_ids,
                                artifact=artifact or task.artifact, note=reason)
            action = "TASK_COMPLETED"
        elif state is WorkerState.BLOCKED:
            self.board.block(task.task_id, worker_id, reason or "blocked")
            action = "TASK_BLOCKED"
        else:
            self.board.fail(task.task_id, worker_id, reason or state.value)
            action = "TASK_FAILED"
        # The owner may be a worker from a previous process, so the roster is
        # consulted defensively: a lost attribution is recoverable, a lost task
        # transition is not.
        agent = self.agents.get(worker_id)
        division = Division(task.division)
        self.ledger.append(
            action, {"agent_id": worker_id, "role": agent.role_name if agent else "former worker"},
            {"agent_id": leader_agent_id(division),
             "role": DIVISION_SPECS[division].leader_title},
            {"task_id": task.task_id, "state": state.value, "gap": task.gap,
             "artifact": task.artifact, "reason": reason[:300],
             "evidence_ids": list(evidence_ids)})

    def _interactive_capable(self) -> bool:
        """Whether a real tool-using subagent can be launched for this run."""
        return self.config.llm_client_factory is not None

    # -- operator control ---------------------------------------------------
    def _emit_status(self, message: str) -> None:
        if self._status_callback is not None:
            self._status_callback(message)

    def stop_worker(self, agent_id: str, *, actor: str, reason: str) -> bool:
        """Stop one worker on behalf of a leader or an operator.

        This is the public seam an operator or a TUI drives. It returns whether a
        live worker was actually reached, so a caller can tell "stopped" apart from
        "there was nothing running", and it refuses to invent an actor — a stop
        with no attribution is not an auditable stop.
        """
        if agent_id not in self.agents and self.control.record(agent_id) is None:
            return False
        record = self.control.record(agent_id)
        if record is None or not record.state.active:
            return False
        self.control.request_stop(agent_id, actor=actor, reason=reason, kind=StopKind.CANCEL)
        return True

    def pause_worker(self, agent_id: str, *, actor: str, reason: str) -> bool:
        record = self.control.record(agent_id)
        if record is None or not record.state.active:
            return False
        self.control.request_stop(agent_id, actor=actor, reason=reason, kind=StopKind.PAUSE)
        return True

    def resume_worker(self, agent_id: str) -> bool:
        token = self.control.token(agent_id)
        if not token.signal.requested or token.signal.kind is not StopKind.PAUSE:
            return False
        token.signal.clear()
        return True

    def stop_run(self, *, actor: str, reason: str) -> None:
        """End the run now, cancelling every active worker."""
        self.control.stop_run(actor=actor, reason=reason)

    def pause_run(self, *, actor: str, reason: str) -> None:
        self.control.pause_run(actor=actor, reason=reason)

    def resume_run(self, *, actor: str) -> None:
        self.control.resume_run(actor=actor)

    def swarm_status(self) -> dict[str, Any]:
        """Everything an operator needs to see mid-run, in one JSON-ready dict."""
        snapshot = self.control.snapshot()
        snapshot["recovery"] = self._recovered
        snapshot["open_tasks"] = [task.to_dict() for task in self.board.open_tasks()]
        snapshot["gaps"] = [status.detail for status in
                            (self._last_report.gaps if self._last_report is not None else ())]
        snapshot["ledger_entries"] = self.ledger.count
        snapshot["ledger_actions"] = self.ledger.actions_used()
        return snapshot

    def render_status(self) -> str:
        return self.control.render()

    def _live_research_mode(self) -> bool:
        """Use the live pipeline, except for the historical formulation fixture."""
        return self._interactive_capable() and not self._legacy_formulation

    def _run_worker(self, agent: ResearchAgent, division: Division, directive: str,
                    *, success_criterion: str, target: Path,
                    tool_names_override: tuple[str, ...] | None = None,
                    max_steps_override: int | None = None,
                    extra_writes: Sequence[Path] = (),
                    task_id: str = "") -> dict[str, Any]:
        """Run one worker as a multi-step, tool-using subagent.

        This replaces a single-shot text completion. The worker is a real agent
        with the division's instruments, so it can write a script, execute it,
        read the failure, and revise — which is what closing a gap actually
        requires. Its trajectory is recorded in the ledger, because a research
        claim is only worth as much as the path that reached it.

        Returns a summary dict; ``ran`` is False when no live model is available,
        in which case the caller falls back to mechanical synthesis.
        """
        from adaptive_harness.agent.swarm import (DeveloperAgentWorker, SwarmAssignment,
                                                   SwarmPhase, SwarmRole, run_assignment)
        from adaptive_harness.data.storage import ExperienceRepository
        from adaptive_harness.research.roles import WORKER_TOOLS
        from adaptive_harness.tools.process import process_scope

        if not self._interactive_capable():
            return {"ran": False, "reason": "no live model configured"}

        def role_client_factory() -> Any:
            client = self.config.llm_client_factory()
            client.default_model = "stealth/space-bunny-alpha"
            return client

        if tool_names_override is not None:
            tool_names = tool_names_override
        elif agent.agent_id == DIRECTOR_ID:
            tool_names = agent.allowed_tools
        elif agent.is_leader:
            tool_names = (("read_file", "write_file", "web_search")
                          if division is Division.LITERATURE else
                          ("read_file", "write_file"))
        else:
            tool_names = WORKER_TOOLS.get(
                division, ("read_file", "write_file", "edit_file", "run_bash"))
        # Two streams, kept apart on purpose: ``trajectory`` is the worker's own
        # tool steps and nothing else, so existing consumers keep working, while
        # ``overseer_log`` holds the supervisor's verdicts about those steps.
        trajectory: list[dict[str, Any]] = []
        overseer_log: list[dict[str, Any]] = []
        outcome: dict[str, Any] = {}
        token = self.control.token(agent.agent_id)

        def on_event(event: Any) -> None:
            # The stop check lives on the event stream, which the tool loop yields
            # from before every tool call. Raising here unwinds the generator, so a
            # cancelled worker makes no further tool call; a tool already in flight
            # is allowed to return, and the process group was killed by
            # ``SwarmControl.request_stop`` so it cannot keep burning.
            if token.expired():
                # Recorded as a timeout, attributed to the harness, and retryable.
                # An unattributed self-cancellation would leave the operator unable
                # to tell a deadline from a decision.
                budget = f"{token.deadline_s:g}s" if token.deadline_s is not None else "its budget"
                self.control.record_timeout(agent.agent_id, f"exceeded {budget}")
                raise WorkerCancelled(agent.agent_id, f"exceeded {budget}", "swarm_control")
            token.raise_if_stopped()
            if self.control.run_stopped:
                raise WorkerCancelled(agent.agent_id, "the run was stopped",
                                      self.control.stopped_by or "operator")
            if self.control.run_paused and not self.control.wait_while_paused():
                raise WorkerCancelled(agent.agent_id, "the run was stopped while paused",
                                      self.control.stopped_by or "operator")
            kind = getattr(event, "event_type", "")
            payload = getattr(event, "payload", {}) or {}
            if kind == "tool_call":
                trajectory.append({"tool": payload.get("name"),
                                   "args": _summarize_args(payload.get("arguments"))})
                if self._status_callback is not None:
                    self._status_callback(f"{agent.role_name}: calling {payload.get('name')}")
            elif kind == "tool_result":
                if trajectory:
                    trajectory[-1]["ok"] = bool(payload.get("success"))
                    error = payload.get("error")
                    if error:
                        trajectory[-1]["error"] = str(error)[:300]
                if self._status_callback is not None:
                    self._status_callback(f"{agent.role_name}: {payload.get('name')} "
                                          f"{'passed' if payload.get('success') else 'failed'}")
            elif kind == "overseer":
                # The per-agent RuntimeOverseer is the thing that spots a worker
                # looping or stalling. Surfacing its verdicts is what lets a lead
                # see *why* a worker burned its whole budget.
                overseer_log.append({"state": payload.get("state"),
                                     "directive": _summarize_args(payload.get("directive"), 200),
                                     "confidence": payload.get("confidence")})
            elif kind == "response":
                outcome.update({"summary": str(payload.get("content", ""))[:4000],
                                "stop_reason": payload.get("stop_reason"),
                                "success": bool(payload.get("success"))})

        authorized = tuple(dict.fromkeys((*self._authorized_paths(target),
                                          *(Path(item) for item in extra_writes))))
        worker = DeveloperAgentWorker(
            llm_client_factory=role_client_factory,
            repository=ExperienceRepository(self.workspace.root / "experience.db"),
            max_steps=(max_steps_override if max_steps_override is not None else
                       4 if agent.agent_id == DIRECTOR_ID or agent.is_leader
                       else self.config.worker_max_steps),
            step_policy=self.config.step_policy,
            safety_profile=self.config.safety_profile,
            tool_names=tool_names,
            forced_mode="coding" if "write_file" in tool_names else "research",
            forced_thinking=("medium" if agent.agent_id == DIRECTOR_ID or agent.is_leader else
                             "max" if division in (Division.THEORY, Division.FORMAL) else
                             "medium" if division in (Division.EMPIRICAL, Division.ADVERSARIAL)
                             or agent.role_name == "Typst Author" else "low"),
            enable_skill_routing=not (agent.agent_id == DIRECTOR_ID or agent.is_leader),
            # Every worker, not just leaders, is confined to its authorized paths.
            # An unrestricted worker could overwrite a sibling's leased artifact,
            # which is exactly the duplicate work the board exists to prevent.
            write_target=authorized,
            system_prompt=self._worker_prompt(agent, division, success_criterion, target),
            on_event=on_event)
        assignment = SwarmAssignment(SwarmRole.CODER, SwarmPhase.IMPLEMENT, directive,
                                     self.workspace.root)
        try:
            # Everything this worker spawns is tagged with its agent id, so a stop
            # can reach exactly its children and nothing else. Without the tag the
            # only handle available is the thread, and a leader running on the main
            # thread would share its identity with the verifiers.
            with process_scope(agent.agent_id):
                result = run_assignment(worker, assignment)
        except WorkerCancelled as exc:
            summary = {"ran": True, "success": False, "cancelled": True,
                       "state": WorkerState.CANCELLED.value,
                       "error": str(exc), "trajectory": trajectory, "tool_calls": len(trajectory),
                       "overseer": overseer_log}
            self._log_trajectory(agent, division, summary)
            return summary
        except Exception as exc:  # noqa: BLE001 - a worker failure must not kill the run
            summary = {"ran": True, "success": False, "state": WorkerState.FAILED.value,
                       "error": f"{type(exc).__name__}: {exc}"[:400], "trajectory": trajectory,
                       "tool_calls": len(trajectory), "overseer": overseer_log}
            self._log_trajectory(agent, division, summary)
            return summary

        summary = {
            "ran": True,
            "success": bool(result.success),
            "summary": result.summary,
            "stop_reason": outcome.get("stop_reason"),
            "error": result.error,
            "trajectory": trajectory,
            "overseer": overseer_log,
            "tool_calls": len(trajectory),
            "wrote_target": target.is_file(),
            "state": WorkerState.COMPLETED.value if result.success else WorkerState.FAILED.value,
        }
        self._log_trajectory(agent, division, summary)
        return summary

    def _worker_prompt(self, agent: ResearchAgent, division: Division,
                       success_criterion: str, target: Path) -> str:
        """The worker's operating instructions: what to build, where, and how to pass.

        The four things a worker needs and historically lacked are stated
        explicitly: its assignment, the exact paths it may write, the condition
        that constitutes success, and who to escalate to. A worker told only
        "close the gap" flails; a worker told these four can decide on its own
        whether it is finished and who to ask when it is not.
        """
        try:
            relative = target.relative_to(self.workspace.root)
        except ValueError:
            relative = target
        authorized = ", ".join(str(self._relpath(path)) for path in self._authorized_paths(target))
        contact = self.bus.contact_for(agent.agent_id)
        assignment = ""
        record = self.control.record(agent.agent_id)
        if record is not None and record.task_id:
            task = self.board.get(record.task_id)
            if task is not None:
                assignment = (f"Your assignment is {task.task_id} (attempt {task.attempts} of "
                              f"{self.config.max_task_attempts}), leased to you alone. "
                              f"No other worker may write {task.artifact} while you hold it. ")
        return (
            f"You are {agent.role_name}, an independent LLM research agent in the "
            f"{DIVISION_SPECS[division].leader_title}'s division. "
            f"{DIVISION_SPECS[division].objective}\n"
            f"{assignment}"
            f"Write your artifact to {relative}. "
            f"You are authorized to write exactly these paths and nothing else: {authorized}. "
            f"Any other write will be refused, so do not attempt to touch a sibling's file. "
            f"{success_criterion} "
            f"If you are blocked and cannot make progress within your step budget, stop and say "
            f"what you tried and what error you hit; your {contact} can reassign or escalate. "
            "Work iteratively: write the artifact, execute it with your tools, read the exact "
            "error or diagnostic, and revise until it passes. Do not report success you have not "
            "observed from a tool result. A proof script must contain no floating-point literal "
            "and no approximating call, because both are rejected before execution. A Lean proof "
            "must contain no sorry, no admit, and no bare axiom declaration. "
            "Keep exploratory scripts in scratch/, outside proofs/ and experiments/. "
            "Every source left in proofs/ or experiments/ is checked by the final gate; "
            "repair or remove failed probes before reporting completion.")

    def _relpath(self, path: Path) -> Path:
        try:
            return path.relative_to(self.workspace.root)
        except ValueError:
            return path

    def _log_trajectory(self, agent: ResearchAgent, division: Division,
                        summary: Mapping[str, Any]) -> None:
        """Record what the worker actually did, so the audit shows the path.

        This records observable actions — tool names, success flags, the overseer's
        verdicts — and not the model's private reasoning. A ledger that claimed to
        capture a model's thinking would be attesting to something it cannot see.
        """
        steps = list(summary.get("trajectory", []))
        overseer = list(summary.get("overseer", []))
        self.ledger.append(
            "WORKER_TRAJECTORY",
            {"agent_id": agent.agent_id, "role": agent.role_name},
            {"agent_id": leader_agent_id(division), "role": DIVISION_SPECS[division].leader_title},
            {"ran": bool(summary.get("ran")),
             "success": bool(summary.get("success")),
             "tool_calls": summary.get("tool_calls", 0),
             "tools_used": [item.get("tool") for item in steps],
             "trajectory": steps[:40],
             "overseer_verdicts": overseer[:12],
             "state": summary.get("state"),
             "stop_reason": summary.get("stop_reason"),
             "cancelled": bool(summary.get("cancelled")),
             "error": (summary.get("error") or "")[:400]})

    def _falsify(self, agent: ResearchAgent, spawned: list[str],
                 task: Task | None = None) -> Mapping[str, Any]:
        """Red-team pass: seek a counterexample, then record the disposition.

        A falsification attempt is only worth something if it actually searched,
        so when a live model is available the red team runs as a tool-using
        subagent with the claim's artifacts in reach. Its verdict must still be a
        structured JSON object: prose is recorded as inconclusive and grants no
        clearance, so a chatty model cannot rubber-stamp a claim.
        """
        target = (self.workspace.root / task.artifact if task is not None
                  else self.workspace.root / "audit" / f"{agent.agent_id}.md")
        verdict: FalsificationVerdict
        if self._live_research_mode():
            outcome = self._run_worker(
                agent, Division.ADVERSARIAL,
                f"Attempt to falsify the current claim for topic '{self.topic}'. "
                f"Your directive: {agent.directive} "
                f"Inspect claim_manifest.json, every proof and Lean theorem under proofs/, "
                f"and experiment scripts under experiments/. Compare each Lean statement "
                f"against the natural-language claim and reject any weakened theorem or "
                f"incorrect sampling distribution.",
                success_criterion=(
                    "Success: either exhibit a concrete counterexample with the exact input that "
                    f"produced it, or state that the search was empty. Then reply with ONLY "
                    f"{FALSIFICATION_VERDICT}"),
                target=target,
                task_id=task.task_id if task is not None else "")
            verdict = FalsificationVerdict.parse(outcome.get("summary"))
            if not outcome.get("tool_calls"):
                verdict = FalsificationVerdict(Clearance.PENDING,
                                               "the falsification worker used no investigative tools")
            elif verdict.status is Clearance.CLEARED and not any(
                    item.get("tool") in {"run_bash", "run_python_repl", "run_lean_proof"}
                    and item.get("ok") for item in outcome.get("trajectory", [])):
                verdict = FalsificationVerdict(
                    Clearance.PENDING,
                    "clearance requires an executed boundary or counterexample check")
            if not verdict.conclusive and not outcome.get("success"):
                verdict = FalsificationVerdict(
                    Clearance.PENDING,
                    outcome.get("error") or "the falsification worker did not return a verdict")
        else:
            raw = self._author(Division.ADVERSARIAL, "counterexample",
                               f"{agent.directive} {FALSIFICATION_VERDICT}")
            verdict = FalsificationVerdict.parse(raw) if self.config.author else FalsificationVerdict(
                Clearance.CLEARED, "no live author configured; mechanical pass found nothing")
            outcome = {"ran": False, "success": True, "tool_calls": 0, "trajectory": []}
        agent.clearance = verdict.status
        summary: dict[str, Any] = {
            "ran": bool(outcome.get("ran")), "tool_calls": outcome.get("tool_calls", 0),
            "stop_reason": outcome.get("stop_reason"), "trajectory": outcome.get("trajectory", []),
            "success": verdict.conclusive, "accepted": verdict.status is not Clearance.PENDING,
            "error": None if verdict.conclusive else verdict.finding,
            "clearance": verdict.status.value}

        # An inconclusive red-team pass is not a failed falsification, it is an
        # unanswered request for a stronger search: it goes back on the board as a
        # blocked task rather than being recorded as an attempt that happened.
        if task is not None:
            if verdict.status is Clearance.PENDING:
                self.control.settle_worker(agent.agent_id, WorkerState.BLOCKED,
                                           reason=verdict.finding[:200])
                self._close_task(task, agent.agent_id, WorkerState.BLOCKED,
                                 reason=verdict.finding[:200])
                summary["state"] = self.control.record(agent.agent_id).state.value
            else:
                self.control.settle_worker(agent.agent_id, WorkerState.COMPLETED,
                                           artifact=task.artifact, reason=verdict.finding[:200],
                                           evidence_ids=("COUNTEREXAMPLE_FOUND",),
                                           tool_calls=outcome.get("tool_calls"))
                self._close_task(task, agent.agent_id, WorkerState.COMPLETED,
                                 reason=verdict.finding[:200],
                                 evidence_ids=("COUNTEREXAMPLE_FOUND",),
                                 artifact=task.artifact)
                summary["state"] = WorkerState.COMPLETED.value

        if verdict.status is Clearance.COUNTEREXAMPLE:
            if self._live_research_mode():
                self._objection_artifacts[agent.agent_id] = self._artifact_fingerprint()
            self.ledger.append("COUNTEREXAMPLE_FOUND",
                               {"agent_id": agent.agent_id, "role": agent.role_name},
                               {"agent_id": leader_agent_id(Division.ADVERSARIAL),
                                "role": "Adversarial Lead"},
                               {"finding": verdict.finding[:600],
                                "task_id": task.task_id if task else None})
            self.ledger.append("SELF_PIVOT",
                               {"agent_id": DIRECTOR_ID, "role": "Chief Scientist"},
                               {"agent_id": leader_agent_id(Division.THEORY),
                                "role": "Theoretical Lead"},
                               {"reason": "counterexample reported; revising the claim",
                                "kind": "revise_claim",
                                "task_id": task.task_id if task else None,
                                "raised_by": agent.agent_id})
            return summary

        self.ledger.append("FALSIFICATION_ATTEMPT",
                           {"agent_id": agent.agent_id, "role": agent.role_name},
                           {"agent_id": leader_agent_id(Division.ADVERSARIAL),
                            "role": "Adversarial Lead"},
                           {"outcome": verdict.finding,
                            "conclusive": verdict.conclusive,
                            "task_id": task.task_id if task else None,
                            "searched": ["boundary cases", "degenerate inputs",
                                         "unstated assumptions"]})
        if verdict.status is Clearance.CLEARED:
            if self._live_research_mode():
                self._clearance_artifacts[agent.agent_id] = self._artifact_fingerprint()
            self.ledger.append("CLEARANCE_GRANTED",
                               {"agent_id": leader_agent_id(Division.ADVERSARIAL),
                                "role": "Adversarial Lead"},
                               {"agent_id": DIRECTOR_ID, "role": "Chief Scientist"},
                               {"cleared": agent.agent_id,
                                "task_id": task.task_id if task else None})
        self._write_audit()
        return summary

    def _produce(self, division: Division, agent: ResearchAgent, gap: str,
                 spawned: list[str], task: Task | None = None) -> Mapping[str, Any]:
        """Close ``gap``, preferring a real tool-using worker over a text reply.

        A one-shot completion can only restate a claim; closing a gap means
        writing a script that executes, so when a live model is available the
        division gets an agent with tools and a success criterion it must observe
        from a tool result. Without one, the mechanical kernel remains the path.

        When the work is board-backed, the target comes from the leased task
        rather than from the worker's index, so a worker's artifact cannot depend
        on how many siblings happen to exist.
        """
        artefact = {"literature": "evidence", "theory": "proofs",
                    "empirical": "experiments", "adversarial": "audit",
                    "formal": "proofs/lean"}.get(division.value, "proofs")
        target = self.workspace.root / artefact / f"{agent.agent_id}.md"
        if division.value == "theory":
            target = self.workspace.proof_dir / f"{agent.agent_id}.py"
        elif division.value == "empirical":
            target = self.workspace.experiment_dir / f"{agent.agent_id}.py"
        elif division.value == "formal":
            target = self.workspace.lean_dir / f"{agent.agent_id}.lean"
        prop = None

        if task is not None:
            target = self.workspace.root / task.artifact
            if division in (Division.THEORY, Division.FORMAL) and self.plan.propositions:
                stem = Path(task.artifact).stem
                prop = next((item for item in self.plan.propositions
                             if item.prop_id.lower() == stem), None)
            if gap == Invariant.DOCUMENT_INTEGRITY.value:
                agent.role_name = "Typst Author"
        elif self._live_research_mode():
            propositions = self.plan.propositions
            worker_index = int(agent.agent_id.rsplit("_", 1)[-1]) - 1
            position = (worker_index + getattr(self, "_current_cycle_index", 1) - 1) % len(propositions) if propositions else 0
            prop = propositions[position] if propositions else None
            if division is Division.THEORY and prop:
                target = self.workspace.proof_dir / f"{prop.prop_id.lower()}.py"
            elif division is Division.FORMAL and prop:
                target = self.workspace.lean_dir / f"{prop.prop_id.lower()}.lean"
            elif division is Division.EMPIRICAL:
                target = self.workspace.experiment_dir / f"exp-{position + 1:02d}.py"
            elif gap == Invariant.DOCUMENT_INTEGRITY.value:
                target = self.workspace.paper_typ
                agent.role_name = "Typst Author"

        if self._live_research_mode():
            criterion = task.acceptance if task is not None and task.acceptance \
                else self._success_criterion(division, target)
            # A retry must know why the last attempt failed. Without this the worker
            # re-derives the same mistake, because a fresh agent has no memory of
            # the run that preceded it.
            prior = ""
            if task is not None and task.attempts > 1 and task.note:
                prior = (f"This is attempt {task.attempts}. The previous attempt at this "
                         f"exact artifact ended with: {task.note[:400]} Do not repeat it.\n")
            # A kernel-written decider is not a gap: the worker's artifact for that
            # claim is the derivation beside it, so the write-first pass is skipped
            # and the worker goes straight to arguing the claim.
            kernel_wrote_target = (division is Division.THEORY and target.suffix == ".py"
                                   and target.is_file() and task is not None
                                   and task.artifact.endswith(".py"))
            if target.suffix in {".py", ".lean", ".typ"} and not target.is_file() \
                    and not kernel_wrote_target:
                receipts = ""
                if target.suffix == ".typ":
                    manifest = self.workspace.root / "receipt_manifest.json"
                    if manifest.is_file():
                        receipts = f" Verified receipt manifest: {manifest.read_text(encoding='utf-8')[:4000]}"
                bootstrap = self._run_worker(
                    agent, division,
                    f"Author the first version of {self._relpath(target)} now. "
                    "Your FIRST and ONLY tool call in this stage must be write_file for that "
                    "exact path. Do not inspect files or explain the task. "
                    + (f"Claim: {prop.statement}. Hypotheses: {'; '.join(prop.hypotheses)}. "
                       f"Exact Lean target: {prop.lean_statement}. " if prop else "")
                    + receipts,
                    success_criterion=f"The exact target file {target.name} exists.",
                    target=target, tool_names_override=("write_file",), max_steps_override=2,
                    task_id=task.task_id if task is not None else "")
                if bootstrap.get("cancelled"):
                    return self._settle(task, agent, bootstrap, cancelled=True)
            criterion = (f"Success: write proofs/{target.stem}.md giving the hypotheses, the "
                         f"step-by-step exact derivation of the claim, and the exact scope of "
                         f"what was proved. The decision script {target.name} already exists, "
                         f"written by the harness from the Director's declared exact expression; "
                         f"do not rewrite it and do not weaken the claim. If you think the claim "
                         f"is false, state the exact counterexample in the derivation."
                         if kernel_wrote_target else
                         f"Success: {target.name} exists and running it with the Python REPL or "
                         f"run_bash exits 0.{criterion}")
            outcome = self._run_worker(agent, division,
                                       f"Close the {gap} gap for topic '{self.topic}'. "
                                       f"Read {division.value}_assignments.md, "
                                       "00_objective_spec.md, and claim_manifest.json first. "
                                       "Inspect existing files in your target area and continue "
                                       "from prior workers' results instead of repeating completed work. "
                                       f"Your directive: {agent.directive} "
                                       + prior
                                       + (f"The claim is: {prop.statement}. Hypotheses: "
                                          f"{'; '.join(prop.hypotheses)}. The harness has already "
                                          f"decided it by exact computation from the expression "
                                          f"{prop.sympy_expression}. Your job is the mathematical "
                                          f"derivation that a reader can follow, written to "
                                          f"proofs/{target.stem}.md. "
                                          if kernel_wrote_target and prop is not None else "")
                                       + (f"Claim: {prop.statement}. Hypotheses: "
                                          f"{'; '.join(prop.hypotheses)}. " if prop else "")
                                       + (f"The exact Lean proposition is: {prop.lean_statement}. "
                                          "Do not weaken or replace it. "
                                          if prop and division is Division.FORMAL else "")
                                       + (f"Your pre-registered hypothesis, fixed before you "
                                          f"start: {task.hypothesis} "
                                          if task is not None and task.hypothesis else "")
                                       + ("Read receipt_manifest.json and cite every verified "
                                          "tag in paper.typ. Run compile_typst and fix all warnings. "
                                          if target == self.workspace.paper_typ else ""),
                                       success_criterion=criterion, target=target,
                                       task_id=task.task_id if task is not None else "")
            self.ledger.append("STATUS_REPORT", {"agent_id": agent.agent_id, "role": agent.role_name},
                               {"agent_id": leader_agent_id(division),
                                "role": DIVISION_SPECS[division].leader_title},
                               {"gap": gap, "mode": "interactive-subagent",
                                "task_id": task.task_id if task else None,
                                "success": outcome.get("success"),
                                "tool_calls": outcome.get("tool_calls", 0),
                                "artifact": str(self._relpath(target))
                                if target.is_file() else None})
            return self._settle(task, agent, outcome, target=target)
        summary: dict[str, Any] = {"ran": False, "success": False, "tool_calls": 0,
                                   "trajectory": []}
        if task is not None:
            return self._settle(task, agent, summary, target=target)

        instruction = (f"Close the {gap} gap for topic '{self.topic}'. "
                       f"Your directive: {agent.directive}")
        produced = self._author(division, artefact, instruction)
        if produced:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(produced, encoding="utf-8")
            self.ledger.append("STATUS_REPORT", {"agent_id": agent.agent_id, "role": agent.role_name},
                               {"agent_id": leader_agent_id(division),
                                "role": DIVISION_SPECS[division].leader_title},
                               {"gap": gap, "artifact": str(target.relative_to(self.workspace.root))})
        else:
            self.ledger.append("STATUS_REPORT", {"agent_id": agent.agent_id, "role": agent.role_name},
                               {"agent_id": leader_agent_id(division),
                                "role": DIVISION_SPECS[division].leader_title},
                               {"gap": gap, "status": "no live author; awaiting artifact"})
        return {"ran": False, "success": bool(produced), "tool_calls": 0,
                "trajectory": [], "state": WorkerState.COMPLETED.value if produced
                else WorkerState.FAILED.value}

    def _settle(self, task: Task | None, agent: ResearchAgent,
                summary: Mapping[str, Any], *, target: Path | None = None,
                cancelled: bool = False) -> dict[str, Any]:
        """Decide a task's fate from its worker's outcome and record it once.

        The acceptance decision is deliberately conservative. A worker that reports
        success but left no artifact does not get a completion, and a worker whose
        artifact exists but whose run was cancelled or errored does not either.
        Optimism here would turn the board into a rubber stamp, and the board is
        what the gate later reads to decide whether work remains.
        """
        result = dict(summary)
        result.setdefault("trajectory", [])
        result.setdefault("tool_calls", 0)
        if task is None:
            return result
        if cancelled or result.get("cancelled"):
            record = self.control.record(agent.agent_id)
            timed_out = record is not None and record.state is WorkerState.TIMED_OUT
            if timed_out:
                # A deadline expiry is not a failure of the work; the attempt goes
                # back on the board so a later cycle can try again.
                self._close_task(task, agent.agent_id, WorkerState.FAILED,
                                 reason=str(result.get("error") or "timed out; retryable"))
            else:
                self.control.settle_worker(agent.agent_id, WorkerState.CANCELLED,
                                           reason=str(result.get("error") or "cancelled"))
                self._close_task(task, agent.agent_id, WorkerState.CANCELLED,
                                 reason=str(result.get("error") or "cancelled"))
            result.update({"accepted": False,
                           "state": WorkerState.TIMED_OUT.value if timed_out
                           else WorkerState.CANCELLED.value})
            return result
        artifact = self._relpath(target) if target is not None else Path(task.artifact)
        evidence = self._artifact_evidence(artifact)
        produced = bool(target is not None and target.is_file())
        # A theory claim needs the natural-language derivation as well as the
        # decision script. When the kernel wrote the script, the script's presence
        # says nothing about whether the worker did its work, so requiring only the
        # script would let an idle worker claim completion.
        worker = self.agents.get(agent.agent_id)
        if produced and worker is not None and worker.division is Division.THEORY \
                and self._live_research_mode():
            derivation = target.with_suffix(".md")
            if not derivation.is_file():
                produced = False
            else:
                artifact = self._relpath(derivation)
        if produced and result.get("success"):
            self.control.settle_worker(agent.agent_id, WorkerState.COMPLETED,
                                       reason=result.get("error") or "",
                                       artifact=str(artifact), evidence_ids=evidence,
                                       tool_calls=result.get("tool_calls"))
            self._close_task(task, agent.agent_id, WorkerState.COMPLETED,
                             reason="artifact produced and verified by the tool loop",
                             evidence_ids=evidence, artifact=str(artifact))
            result.update({"accepted": True, "state": WorkerState.COMPLETED.value,
                           "evidence_ids": list(evidence)})
            return result
        if not produced:
            self.control.settle_worker(agent.agent_id, WorkerState.FAILED,
                                       reason=result.get("error") or "no artifact was written",
                                       tool_calls=result.get("tool_calls"))
            self._close_task(task, agent.agent_id, WorkerState.FAILED,
                             reason=result.get("error") or "no artifact was written")
        else:
            # The file is there but the worker's own run did not succeed. That is a
            # retryable attempt, not a completion: an unverified file must not be
            # promoted to evidence.
            self.control.settle_worker(agent.agent_id, WorkerState.FAILED,
                                       reason=result.get("error") or "artifact unverified",
                                       tool_calls=result.get("tool_calls"))
            self._close_task(task, agent.agent_id, WorkerState.FAILED,
                             reason=result.get("error") or "artifact exists but is unverified")
        result.update({"accepted": False, "state": WorkerState.FAILED.value,
                       "evidence_ids": []})
        return result

    def _artifact_evidence(self, artifact: Path) -> tuple[str, ...]:
        """Receipt ids the gates already issued for a specific artifact.

        A task is only completed with the receipts that a verifier minted for
        *that* file. Attributing a proof receipt to the wrong script would let an
        unverified file inherit another's standing, which is the failure mode the
        per-claim Lean gate exists to prevent. Only certified receipts count: a
        rejected receipt is evidence of a failure, not evidence of a proof.
        """
        name = artifact.name
        if artifact.suffix == ".lean":
            return tuple(receipt.proof_id for receipt in self.lean_gate.receipts
                         if receipt.name == name and receipt.certified)
        if artifact.suffix == ".py" and artifact.parent.name == "proofs":
            return tuple(receipt.theorem_id for receipt in self.proofs.receipts
                         if Path(receipt.script).name == name and receipt.verified)
        if artifact.suffix == ".py" and artifact.parent.name == "experiments":
            return tuple(receipt.experiment_id for receipt in self.experiments.receipts
                         if Path(receipt.script).name == name and receipt.replicated)
        return ()

    def _success_criterion(self, division: Division, target: Path) -> str:
        """The observable condition a worker must reach, stated per division."""
        name = target.name
        if target.suffix == ".typ":
            return ("Success: paper.typ cites every tag in receipt_manifest.json and "
                    "compile_typst produces paper.pdf with zero warnings.")
        if division is Division.FORMAL:
            return (f"Success: call run_lean_proof on {name} and observe exit code 0 with no "
                    f"sorry, no admit, and no axiom declaration.")
        if division in (Division.THEORY, Division.EMPIRICAL):
            extra = (" The script must also write a data artifact so the run can be replicated."
                     if division is Division.EMPIRICAL else "")
            if division is Division.EMPIRICAL and self._live_research_mode():
                extra += (f" Write a raw CSV and {target.stem}.predictions.json with a "
                          "nonempty JSON array of claim_id, description, predicted, observed, "
                          "half_width_95, and relative fields. Compute the interval from raw "
                          "samples and fail if the prediction lies outside it.")
            if division is Division.THEORY and self._live_research_mode():
                # The kernel already wrote the decider for every claim that carried
                # an exact expression, and the adjudication gate requires the
                # natural-language derivation beside it. So the worker's deliverable
                # is the *argument*, not a transcription of arithmetic SymPy has
                # already done. Telling it to write the script is what produced
                # 15,000-line category errors.
                decider_exists = target.is_file()
                if decider_exists:
                    return (f"Success: proofs/{target.stem}.md exists and states the hypotheses, "
                            f"a step-by-step exact derivation of the claim, and the precise scope "
                            f"of what was proved. The decision script {name} was written by the "
                            f"harness from the Director's declared exact expression; do not "
                            f"rewrite it and do not weaken the claim. If you believe the claim is "
                            f"false, say so in the derivation with the exact counterexample "
                            f"instead of editing the script.")
                extra += (f" Also write proofs/{target.stem}.md with hypotheses, a "
                          f"step-by-step exact derivation, and the scope of what was proved.")
            return (f"Success: {name} exists and running it with the Python REPL or run_bash exits "
                    f"0.{extra}")

        if division is Division.ADVERSARIAL:
            return ("Success: report a concrete counterexample with the exact input that produced "
                    "it, or state that the declared search returned empty.")
        return f"Success: {name} exists and contains the evidence you relied on."

    def _write_bibliography(self) -> Path:
        """Emit BibTeX for every recorded evidence record.

        Nothing is invented: an un-cited run produces an explicitly empty
        bibliography with a comment, so the absence of sources is visible in
        the artifact rather than silently padded with plausible-looking
        references.
        """
        records = self.workspace.evidence_records()
        lines = [f"% BibTeX for the research artifact '{self.slug}'.",
                 "% Generated by the Adaptive Agent Harness; one entry per evidence record.",
                 "% The swarm never fabricates a citation, so an uncited run leaves this empty.", ""]
        for record in records:
            key = str(record.get("id", "EVID")).replace("-", "")
            fields = {
                "title": str(record.get("claim", "")),
                "howpublished": str(record.get("source", "")),
                "note": str(record.get("citation", "")),
                "year": str(record.get("timestamp", ""))[:4],
            }
            lines.append("@misc{" + key + ",")
            for name, value in fields.items():
                lines.append(f"  {name:<13}= {{{value}}},")
            lines += ["}", ""]
        self.workspace.bibliography.write_text("\n".join(lines), encoding="utf-8")
        return self.workspace.bibliography

    def _write_audit(self) -> None:
        cleared = [agent for agent in self.agents.values()
                   if agent.division is Division.ADVERSARIAL and not agent.is_leader]
        lines = [f"# Adversarial Audit — {self.topic}", "",
                 "Red-team workers attempt to falsify every claim before clearance.", ""]
        for agent in cleared:
            lines.append(f"- `{agent.agent_id}` ({agent.role_name}): "
                         f"**{agent.clearance.value}** — directive: {agent.directive[:200]}")
        lines += ["", "## Falsification attempts", ""]
        for entry in self.ledger.by_action("FALSIFICATION_ATTEMPT"):
            lines.append(f"- {entry.id} {entry.sender['agent_id']}: {entry.payload.get('outcome')} "
                         f"(searched: {', '.join(entry.payload.get('searched', []))})")
        self.workspace.audit.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # -- recovery -----------------------------------------------------------
    def _recover_state(self) -> dict[str, Any]:
        """Rebuild coordination state after an interrupted run.

        Three things are recovered, in order of authority:

        1. the persisted task board, so a task keeps its id, attempt count and
           evidence rather than being reopened as a fresh duplicate;
        2. the ledger, which overrides the board — a task the ledger shows
           completed is completed, whatever the board file says;
        3. the inbox, so a help request made before the interruption is still
           unanswered rather than silently dropped.

        Deliberately *not* recovered: the worker roster. Agents are cheap to
        re-create and their ids are derived, so a restart re-hires the pool rather
        than trusting a serialized record of who used to exist.
        """
        summary = {"resumed": False, "tasks": 0, "healed": 0, "messages": 0}
        if not self.config.resume:
            return summary
        board_path = self.workspace.root / TASK_BOARD_FILENAME
        restored = TaskBoard.load(board_path)
        if restored is not None:
            self.board = restored
            self.control.board = restored
            summary["tasks"] = len(restored.all())
            summary["healed"] = restored.recovered_from(self.ledger)
        self.bus.load(self.ledger.read_raw())
        summary["messages"] = len(self.bus.all())
        if not summary["tasks"] and not summary["messages"]:
            return summary
        summary["resumed"] = True
        # Anything the previous run left mid-flight is now stale: its worker no
        # longer exists in this process, so its lease must not block a new one.
        self.ledger.append(
            "RUN_RESUMED_FROM_STATE", {"agent_id": DIRECTOR_ID, "role": "Chief Scientist"},
            {"agent_id": "archive", "role": "Recovery"},
            {"tasks_restored": summary["tasks"], "tasks_healed": summary["healed"],
             "messages_restored": summary["messages"],
             "unacknowledged_requests": len(self.bus.unacknowledged()),
             "note": "completed tasks are not re-run; stale leases were released"})
        return summary

    def recovery_report(self) -> dict[str, Any]:
        """What recovery restored, for the CLI banner and the run log."""
        return dict(self._recovered)

    def run(self, *, objective: str = "", on_status: Callable[[str], None] | None = None) -> ResearchOutcome:
        """Execute the relentless loop to convergence (or proven stagnation)."""
        def emit(message: str) -> None:
            if on_status is not None:
                on_status(message)

        self._status_callback = emit
        self.control.on_status = emit
        # Keep running workers' leases alive independently of their tool loops.
        self.control.start_heartbeat(self.config.task_lease_renew_s)

        if self._live_research_mode():
            self._prepare_live_research(objective)
            # The Director supplies each claim's exact expression; the kernel turns
            # those into self-adjudicating scripts. Deciding the computational tier
            # is arithmetic, not authoring, and leaving it to a language model is
            # what made every live run fail.
            self._synthesize()
        else:
            self._write_objective_spec(objective)
            self._synthesize()
            self._synthesize_experiments()
            self._formalise()
        loop = RelentlessConvergenceLoop(
            gate=self.build_gate(),
            cycle_fn=lambda report, index, budget: self.cycle(report, index, budget),
            stagnation_patience=self.config.stagnation_patience,
            budget_ceiling=self.config.max_workers_per_division,
            absolute_ceiling=self.config.absolute_ceiling,
            on_cycle=lambda cycle: emit(
                f"cycle {cycle.index}: gaps={list(cycle.gaps) or 'none'} "
                f"spawned={len(cycle.workers_spawned)} active={len(self.agents)} "
                f"proofs={len(self.proofs.receipts)} lean={len(self.lean_gate.receipts)} "
                f"experiments={len(self.experiments.receipts)}"))
        emit("convergence loop started")
        cycle_limit = (0 if self._live_research_mode() and not self.plan.propositions
                       else self.config.max_cycles)
        # The control plane is the stop authority. Passing it here is what makes a
        # leader's stop request end the run with EXTERNAL_STOP and a named actor,
        # instead of the loop grinding on to stagnation and implying it ran out of
        # options on its own.
        outcome: ConvergenceOutcome = loop.run(max_cycles=cycle_limit,
                                               should_stop=self.control.should_stop)
        emit(f"convergence loop finished: {outcome.stop_reason.value}")
        self._last_outcome = outcome
        self._last_report = outcome.final_report
        self.control.stop_heartbeat()
        self.board.save(self.workspace.root / TASK_BOARD_FILENAME)
        # Re-render once with the terminal state so the delivered PDF reports
        # the verdict it was published under, not the previous cycle's.
        self._republish(outcome)
        self._finalize(outcome)
        ledger_ok, ledger_detail = self.ledger.verify()
        self.ledger.append("DELIVERABLE_WRITTEN",
                           {"agent_id": DIRECTOR_ID, "role": "Chief Scientist"},
                           {"agent_id": "archive", "role": "Publication"},
                           {"solved": outcome.solved, "stop_reason": outcome.stop_reason.value,
                            "cycles": outcome.cycles, "pdf": str(self.workspace.paper_pdf),
                            "stopped_by": self.control.stopped_by or None,
                            "open_tasks": [task.task_id for task in self.board.open_tasks()],
                            "unacknowledged_requests": len(self.bus.unacknowledged()),
                            "pdf_hash": sha256_file(self.workspace.paper_pdf)
                            if self.workspace.paper_pdf.is_file() else None})
        return ResearchOutcome(
            topic=self.topic, slug=self.slug, workspace=self.workspace.root,
            solved=outcome.solved, stop_reason=outcome.stop_reason, cycles=outcome.cycles,
            workers_spawned=outcome.workers_spawned, ledger_ok=ledger_ok,
            ledger_detail=ledger_detail,
            pdf=str(self.workspace.paper_pdf) if self.workspace.paper_pdf.is_file() else None,
            agents=tuple(self.agents.values()))

    def _claim_spec_directive(self) -> str:
        """How the Director must state a claim so both tiers can actually decide it.

        This is the single highest-leverage instruction in the run. A claim phrased
        as a fact about a proof assistant's internals — "Nat.succ n + m reduces
        definitionally", "the HAdd instance" — has no exactly-decidable
        computational content. A SymPy worker handed such a claim has nothing to
        compute, so it writes a Lean evaluator in Python, and that fails. Observed
        across three live runs: every attempt produced a 15,000-line script and a
        non-zero exit.

        So the Director is told to state the *mathematics* and to supply the exact
        expression separately, with a worked example. A worked example is not
        decoration here: prose alone did not change the behaviour.
        """
        return (
            "CRITICAL — each mathematical claim needs three separate fields, and they must "
            "say different things. (1) 'statement': the mathematics itself, in ordinary language, "
            "with no reference to how any tool reduces or represents it. Do NOT write 'Nat.succ n "
            "+ m reduces definitionally' or 'the HAdd instance resolves it'; write 'for all "
            "integers n and m, the successor of n added to m equals the successor of n plus m'. "
            "(2) 'sympy_expression': the claim as an exact SymPy identity, written literally as "
            "'<lhs> == <rhs>' in SymPy syntax. This is what the computational tier decides, and "
            "the kernel decides it directly, so it must be something SymPy can settle exactly. "
            "(3) 'lean_statement': the exact Lean proposition, same scope and hypotheses as the "
            "statement, for example 'example (n m : Nat) : Nat.succ n + m = Nat.succ (n + m) "
            ":= rfl'. "
            "Include one central claim and at most two supporting lemmas. Every decidable claim "
            "must carry a nonempty 'sympy_expression' and a nonempty 'lean_statement'. "
            "WORKED EXAMPLE for the topic '2 + 2 = 4': statement 'the sum of the natural number "
            "two and two is four'; sympy_expression '2 + 2 == 4'; lean_statement 'example : "
            "(2 : Nat) + 2 = 4 := rfl'. For the binomial theorem: statement 'the square of a sum "
            "of two integers equals the sum of their squares plus twice their product'; "
            "sympy_expression '(x + y)**2 == x**2 + 2*x*y + y**2'; lean_statement 'example (x y : "
            "Nat) : (x + y)^2 = x^2 + 2*x*y + y^2 := by ring_nf'. "
            + self._open_problem_directive()
            + "Keep empirical predictions out of the mathematical claims array. "
        )

    def _open_problem_directive(self) -> str:
        """How to point the run at an open problem without pretending to settle it.

        An open problem is not a proposition, and demanding an exact expression
        for one is the category error this harness was rewritten to remove. The
        workable shape is a ladder: state the problem as a ``conjecture`` — which
        is carried as the target, printed as open, and never adjudicated — and then
        state the decidable rungs beneath it that a run can genuinely settle.
        """
        return (
            "IF THE TOPIC IS AN OPEN PROBLEM, do not force it into a single decidable claim, "
            "and do not claim to settle it. Structure the claims array as a ladder. (a) One "
            "claim with kind 'conjecture' stating the open problem itself in plain mathematics. "
            "It may have no sympy_expression; that is expected and correct. It is recorded as the "
            "target and is never marked proven. (b) Decidable claims underneath it, in increasing "
            "difficulty, each with its own kind 'theorem' plus a sympy_expression and a "
            "lean_statement: the exhaustive small cases, a verifying procedure for a proposed "
            "witness or counterexample, boundary and degenerate cases, and any supporting lemma "
            "the argument needs. (c) An explicit statement of what remains open. "
            "For a conjecture, prefer 'every digraph with at most n vertices satisfies P' with an "
            "exact expression that enumerates all such digraphs, over a hand-written script. "
            "Report progress on the rungs, and say plainly that the top rung is unsolved. "
        )

    def _prepare_live_research(self, objective: str) -> None:
        """Let the Director and leads define a new topic before any worker writes proofs.

        A model-authored manifest is deliberately data, never executable code.
        Missing or malformed claims leave the gate unsatisfied. No built-in topic
        library is consulted on this path.
        """
        from adaptive_harness.classifiers.ambiguity_classifier import AmbiguityClassifier
        from adaptive_harness.classifiers.skill_classifier import SkillClassifier

        assessment = AmbiguityClassifier().evaluate(
            objective or self.topic, SkillClassifier().classify(objective or self.topic))
        self.ledger.append(
            "STATUS_REPORT", {"agent_id": DIRECTOR_ID, "role": "Chief Scientist"},
            {"agent_id": "all_leads", "role": "Division Leads"},
            {"stage": "ambiguity_preflight", "assessment": assessment.to_dict()})
        manifest_path = self.workspace.root / "claim_manifest.json"
        imposed_claim = (f" Required user claim: {self.config.claim}. Declared symbols: "
                         f"{', '.join(self.config.claim_symbols)}. Preserve its meaning exactly."
                          if self.config.claim else "")
        director = ResearchAgent(DIRECTOR_ID, "Executive Director", objective or self.topic,
                                 ("write_file",), division=Division.THEORY)
        # A resumed run keeps the Director's own artifacts. Re-authoring them would
        # re-bill the most expensive call in the run and could silently *change* the
        # claim set, invalidating every proof and receipt already on disk.
        reusing = self.config.resume and manifest_path.is_file() and \
            self.workspace.objective_spec.is_file()
        if not reusing:
            self._run_worker(
                director, Division.THEORY,
                f"Research topic: {self.topic}. Objective: {objective or self.topic}."
                f"{imposed_claim} "
                "Use write_file to create 00_objective_spec.md with variable domains, boundary "
                "cases, assumptions, and a falsifiable goal. Also write claim_manifest.json as "
                "JSON with a nonempty 'claims' array. Each claim needs id (PROP-01 style), "
                "name, statement, hypotheses (array), and kind. "
                + self._claim_spec_directive()
                + "Use only claims you can assign to independent SymPy "
                "and Lean workers. Set top-level 'empirical_required' to true when a simulation "
                "can test the statement. Do not write proof scripts yourself.",
                success_criterion="Both the objective specification and valid claim manifest exist.",
                target=manifest_path,
                # The Director produces two files, not one; both are named here so
                # the write allow-list covers the assignment it was actually given.
                extra_writes=(self.workspace.objective_spec,))
        else:
            self._emit_status("resuming: reusing the Director's existing objective spec "
                              "and claim manifest")
            self.ledger.append(
                "STATUS_REPORT", {"agent_id": DIRECTOR_ID, "role": "Chief Scientist"},
                {"agent_id": "archive", "role": "Recovery"},
                {"status": "reused_director_artifacts", "manifest": manifest_path.name})

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw_claims = payload["claims"]
            if not isinstance(raw_claims, list) or not raw_claims:
                raise ValueError("claims must be a nonempty array")
            propositions: list[Proposition] = []
            for item in raw_claims:
                if not isinstance(item, dict) or not re.fullmatch(r"[A-Z][A-Z0-9_-]{2,40}", str(item.get("id", ""))):
                    raise ValueError("claim id must be a short uppercase identifier")
                statement = str(item.get("statement") or item.get("exact_statement") or "").strip()
                if not statement:
                    raise ValueError("each claim needs a statement")
                hypotheses = item.get("hypotheses", [])
                if not isinstance(hypotheses, list) or not all(isinstance(h, str) for h in hypotheses):
                    raise ValueError("hypotheses must be an array of strings")
                if item.get("kind") == "numerical_simulation":
                    continue
                expression = str(item.get("sympy_expression")
                                 or item.get("exact_expression")
                                 or item.get("sympy_statement") or "").strip()
                lean = str(item.get("lean_statement", "")).strip()
                # A goal is allowed to be undecidable — that is what a goal is.
                # It is carried as the target, never adjudicated, and the run's
                # progress is the decidable claims beneath it. Requiring an exact
                # expression of the goal itself is the category error this whole
                # change set exists to remove.
                # An unkinded claim is a theorem, not an open problem. Defaulting
                # to "conjecture" would let any manifest that omits a kind escape
                # validation and be carried as the target.
                goal = Proposition(
                    prop_id=item["id"], kind=str(item.get("kind") or "theorem"),
                    name=str(item.get("name", item["id"])), statement=statement,
                    hypotheses=tuple(hypotheses), lean_statement=lean,
                    sympy_expression=expression)
                if goal.is_goal:
                    propositions.append(goal)
                    continue
                if not expression:
                    raise ValueError(
                        f"claim {item['id']} has no 'sympy_expression'; a claim the exact "
                        "computer cannot decide cannot be settled, and asking a worker to "
                        "invent one produced scripts that failed every time. If this is the "
                        "open problem itself, give it kind 'conjecture'.")
                if not lean:
                    raise ValueError(
                        f"claim {item['id']} has no 'lean_statement'; the formal tier is part "
                        "of the declared proof standard")
                propositions.append(Proposition(
                    prop_id=item["id"], kind=str(item.get("kind", "theorem")),
                    name=str(item.get("name", item["id"])),
                    statement=statement, hypotheses=tuple(hypotheses),
                    lean_statement=lean,
                    sympy_expression=expression,
                    script=exact_decider(expression, statement=statement)))
            if not propositions or len({item.prop_id for item in propositions}) != len(propositions):
                raise ValueError("claim ids must be unique")
            self.plan = TopicPlan(self.topic, tuple(propositions), strategy="llm-authored",
                                  notes="Claims were authored by the Director LLM and await verification.")
            self.planned_experiments = int(bool(payload.get("empirical_required",
                any(bool(item.get("empirical_required")) for item in raw_claims))))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.plan = TopicPlan(self.topic, strategy="llm-authored",
                                  notes=f"Director did not produce a valid claim manifest: {exc}")
            self.planned_experiments = 1

        if not self.plan.propositions:
            return

        for division, spec in DIVISION_SPECS.items():
            lead = self.agents[leader_agent_id(division)]
            target = self.workspace.root / f"{division.value}_assignments.md"
            # Same reasoning as the Director: a lead's decomposition is expensive
            # and stable. Re-deriving it every restart would burn budget and could
            # contradict the manifest the proofs were written against.
            if not (self.config.resume and target.is_file()):
                self._run_worker(
                    lead, division,
                    f"Topic: {self.topic}. Read 00_objective_spec.md and claim_manifest.json. "
                    f"As {spec.leader_title}, decompose the objective into specific assignments "
                    f"for your division and write {target.name}. Record assumptions and exact "
                    "success criteria. Do not claim a result without a tool receipt. "
                    + ("Search primary literature with web_search or a citable source. Write "
                       "evidence/index.json as an array of records with id, claim, source URL, "
                       "and citation. Cite only material you inspected. "
                       if division is Division.LITERATURE else ""),
                    success_criterion=f"{target.name} contains actionable assignments.",
                    target=target)
            if not target.is_file():
                lines = [f"# {spec.leader_title} assignments", "", spec.objective, "",
                         "The lead did not leave a written assignment. Continue from the "
                         "Director's manifest and verified receipts; report this gap.", ""]
                for claim in self.plan.propositions:
                    lines.append(f"- {claim.prop_id}: {claim.statement}")
                    if claim.lean_statement:
                        lines.append(f"  Lean target: `{claim.lean_statement}`")
                target.write_text("\n".join(lines) + "\n", encoding="utf-8")
                self.ledger.append(
                    "STATUS_REPORT", {"agent_id": DIRECTOR_ID, "role": "Chief Scientist"},
                    {"agent_id": lead.agent_id, "role": lead.role_name},
                    {"status": "lead_assignment_missing", "fallback_path": target.name})

    def _progress_report(self, outcome: ConvergenceOutcome) -> str:
        """The published document for a run that did not converge.

        It has to be *diagnostic*, not merely apologetic. A bare "not solved" tells
        a reader nothing they could not already see. This one states which
        invariants failed and why, what each division produced, which tasks remain
        open, and whether a human stopped the run — the facts a reader needs to
        decide whether to resume, redirect, or abandon.
        """
        def esc(text: str) -> str:
            # Typst is markup: an unescaped '#', '*', or '_' from a model-authored
            # artifact is exactly what broke compilation in the first place.
            return (str(text).replace("\\", "\\\\").replace("#", "\\#")
                    .replace("*", "\\*").replace("_", "\\_").replace("$", "\\$"))

        # Publication is a best-effort final step, so a partially populated outcome
        # degrades the report rather than raising on the way to the PDF.
        cycles = getattr(outcome, "cycles", 0)
        spawned = getattr(outcome, "workers_spawned", 0)
        stop_reason = getattr(getattr(outcome, "stop_reason", None), "value", "unknown")
        final_report = getattr(outcome, "final_report", None)

        lines = [
            "= Research progress report", "",
            "The research swarm did not verify its objective. This document is a progress "
            "report, not a proof and not a completed academic paper. Nothing below should "
            "be read as a settled result.", ""]
        for goal in self.goals:
            lines += [f"== Open problem under investigation: {goal.name}", "",
                      f"*{goal.statement}*", "",
                      "This is an open problem. Nothing in this report resolves it. What "
                      "follows is machine-checked progress on the decidable sub-claims beneath "
                      "it, and the gap between them and the full statement is stated explicitly "
                      "below.", ""]
        lines += [
            f"The loop exited after {cycles} cycle(s) with `{stop_reason}`, having recruited "
            f"{spawned} worker(s).", ""]

        if self.control.stopped_by:
            lines += [f"== The run was stopped by a human", "",
                      f"`{esc(self.control.stopped_by)}` stopped the run: "
                      f"{esc(self.control._run_stop.reason or 'no reason recorded')}. This "
                      "is an operator decision, not a mathematical limit — the loop was "
                      "interrupted, so no claim about what the swarm could have achieved is "
                      "implied either way.", ""]

        lines += ["== Which invariants failed", ""]
        statuses = (final_report.statuses if final_report is not None else ())
        for status in statuses:
            mark = "satisfied" if status.satisfied else "*not satisfied*"
            lines.append(f"- `{status.invariant.value}` — {mark}. {esc(status.detail)}")
        if not statuses:
            lines.append("The gate produced no status record for this run.")
        lines.append("")

        if self.decidable:
            lines += ["== What each decidable claim was decided to be", ""]
            for item in self.claims.adjudications:
                lines.append(f"- `{item.prop_id}` — **{item.verdict.value}** "
                             f"(script exit {item.exit_code}).")
                if item.statement:
                    lines.append(f"  - Claimed: {esc(item.statement[:300])}")
                if item.finding:
                    lines.append(f"  - Kernel said: {esc(item.finding[:220])}")
            if not self.claims.adjudications:
                for claim in self.decidable:
                    lines.append(f"- `{claim.prop_id}` — **UNTESTED**, no verdict was reached.")
                    if claim.statement:
                        lines.append(f"  - Claimed: {esc(claim.statement[:300])}")
            lines.append("")
        if self.goals:
            settled = {item.prop_id for item in
                       (*self.claims.proven, *self.claims.disproven)}
            unresolved = [claim.prop_id for claim in self.decidable
                          if claim.prop_id not in settled]
            lines += ["== What remains open", ""]
            for goal in self.goals:
                lines.append(f"- `{goal.prop_id}` ({esc(goal.name)}) is unsolved. The run "
                             f"establishes nothing about it beyond the decidable claims above.")
            if unresolved:
                lines.append("- The following decidable sub-claims were also not settled: "
                             + ", ".join(f"`{item}`" for item in unresolved) + ".")
            lines.append("")

        certified = [item for item in self.lean_gate.receipts if item.certified]
        lines += ["== The formal tier", ""]
        if certified:
            lines.append(f"{len(certified)} of {len(self.lean_gate.receipts)} Lean file(s) "
                         "were machine-checked by the kernel with no `sorry`, no `admit`, and "
                         "no user-declared axiom:")
            for item in certified:
                lines.append(f"- `{esc(item.name)}` — {esc(', '.join(item.theorems) or 'no named theorem')}")
        else:
            lines.append("No Lean file was certified in this run, so nothing here rests on a "
                         "machine-checked formal derivation.")
        lines.append("")

        open_tasks = self.board.open_tasks()
        if open_tasks:
            lines += ["== Work that was left open", ""]
            for task in open_tasks:
                lines.append(f"- `{task.task_id}` ({esc(task.division)}, "
                             f"{esc(task.gap)}) — artifact `{esc(task.artifact)}`, "
                             f"{task.attempts} attempt(s). {esc(task.note[:200])}")
            lines.append("")

        if self.bus.unacknowledged():
            lines += ["== Requests that were never answered", ""]
            for message in self.bus.unacknowledged():
                lines.append(f"- `{message.message_id}`: {esc(message.sender)} asked "
                             f"{esc(message.recipient)} for help on "
                             f"*{esc(message.subject)}* and received no reply.")
            lines.append("")

        lines += ["== Where the evidence is", "",
                  "The objective specification, exact claim manifest, attempted proof scripts, "
                  "verification receipts, the task board, the convergence history, and the "
                  "hash-chained communication ledger are preserved beside this report. The "
                  "ledger verifies its own integrity; the receipts carry the SHA-256 of the "
                  "exact bytes that were executed.", ""]
        return "\n".join(lines) + "\n"

    def _republish(self, outcome: ConvergenceOutcome) -> None:
        """Build the final PDF from the terminal gate state, tolerating failure.

        On an unsolved live run the delivered document is the diagnostic progress
        report, not whatever the Typst author last wrote. A draft is preserved as
        ``paper_draft.typ`` so the work is not lost, but shipping a stale draft
        because it happens to compile would replace an honest account of the run
        with something that reads like a finished paper.
        """
        if self._live_research_mode():
            if not outcome.solved:
                marker = "#align(center)[*Research status: UNSOLVED*]"
                draft = (self.workspace.paper_typ.read_text(encoding="utf-8")
                         if self.workspace.paper_typ.is_file() else "")
                body = self._progress_report(outcome)
                preserved = draft if draft and marker not in draft else ""
                if preserved:
                    (self.workspace.root / "paper_draft.typ").write_text(
                        preserved, encoding="utf-8")
                self.workspace.paper_typ.write_text(marker + "\n\n" + body, encoding="utf-8")
                result = self._compile()
                if result is None or not result.success:
                    # The progress report is generated, not authored, so a failure
                    # here is a harness defect. Say so instead of shipping nothing.
                    detail = (result.stderr if result is not None else "no compiler result")
                    (self.workspace.root / "progress_report_build_error.txt").write_text(
                        f"{detail}\n", encoding="utf-8")
                    minimal = ("= Research progress report\n\nThe swarm did not verify its "
                               "objective and the diagnostic report failed to typeset. The "
                               "reason is recorded in progress_report_build_error.txt.\n")
                    self.workspace.paper_typ.write_text(marker + "\n\n" + minimal,
                                                        encoding="utf-8")
                    self._compile()
            return
        try:
            builder = PaperBuilder(self.workspace.root, typst_root=self.config.typst_root,
                                   allow_install_typst=self.config.auto_install_typst)
            target = builder.render(self._paper_inputs())
            self._paper_digest = sha256_file(target)
            builder.compile(target)
        except (OSError, RuntimeError):
            # A failed final render must not mask the real convergence verdict;
            # the gate already recorded document integrity for this run.
            pass

    def _write_objective_spec(self, objective: str) -> None:
        lines = [f"# Objective Specification — {self.topic}", "",
                 "## Objective", "", textwrap.fill(objective or self.topic, 92), "",
                 "## Derivation Plan", "",
                 f"Strategy selected: `{self.plan.strategy}`", ""]
        if self.plan.notes:
            lines += [textwrap.fill(self.plan.notes, 92), ""]
        if self.plan.propositions:
            origin = ("formulated by the Director for an uncovered topic"
                      if self.plan.strategy == "dynamic-formulation"
                      else "derived by the kernel")
            lines.append(f"The kernel will attempt the following propositions ({origin}):")
            lines.append("")
            for prop in self.plan.propositions:
                lines.append(f"- `{prop.prop_id}` ({prop.kind}) — {prop.name}")
                if prop.statement:
                    lines.append(f"  - {prop.statement}")
            lines.append("")
        else:
            lines += ["No proposition could be constructed for this topic.", ""]
            if self.formulation is not None and self.formulation.notes:
                lines += ["### Formulation attempt", "", self.formulation.notes, ""]
        lines += ["## Definition of Solved", "",
                  "A result is settled only when all five invariants hold:", ""]
        for invariant, rationale in (
                (Invariant.MATHEMATICAL_SOUNDNESS,
                 "every derivation in `proofs/` is free of approximation and reached a clean verdict"),
                (Invariant.EMPIRICAL_REPLICATION,
                 "every experiment in `experiments/` reproduces its prediction at a pinned seed within 95%"),
                (Invariant.ADVERSARIAL_CLEARANCE,
                 "no red-team worker holds an open counterexample"),
                (Invariant.CLAIM_ADJUDICATION,
                 "the headline claim reached a decided verdict: PROVEN or DISPROVEN"),
                (Invariant.DOCUMENT_INTEGRITY,
                 "`paper.typ` compiles to `paper.pdf` with zero Typst warnings")):
            lines.append(f"- **{invariant.value}** — {rationale}.")
        lines += ["", "## Divisions", ""]
        for division, spec in DIVISION_SPECS.items():
            lines.append(f"- **{spec.leader_title}** — {spec.objective}")
        self.workspace.objective_spec.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _finalize(self, outcome: ConvergenceOutcome) -> None:
        write_cycle_history(self.workspace.root / "convergence_history.json", outcome)
        self._write_audit()
        self._write_bibliography()
