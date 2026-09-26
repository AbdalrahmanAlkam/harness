"""Autonomous hierarchical research: proof receipts, seeded replication, and publication.

The contract enforced here is *no claim without a receipt*. A result enters the
deliverable only as an executable artifact — an exact SymPy script that exits 0,
a seeded simulation with a hashed data file, or a cited primary source — and the
Relentless Convergence Loop keeps running until every invariant holds.
"""

from adaptive_harness.research.experiment import (ExperimentError, ExperimentReceipt,
                                                  ExperimentRunner, Prediction,
                                                  mean_confidence_interval_95)
from adaptive_harness.research.gate import (ConvergenceCycle, ConvergenceOutcome, Invariant,
                                            InvariantGate, InvariantStatus, RelentlessConvergenceLoop,
                                            StopReason, write_cycle_history)
from adaptive_harness.research.ledger import (ACTIONS, CommLedger, LedgerEntry, LedgerError,
                                              canonical_json, sha256_file, sha256_text)
from adaptive_harness.research.claim import (ClaimAdjudication, ClaimLedger, EXIT_COUNTEREXAMPLE,
                                              EXIT_HOLDS, ParsedClaim, Proposition, TopicPlan,
                                              Verdict, parse_claim, verdict_from_exit)
from adaptive_harness.research.paper import (PaperBuilder, PaperInputs, escape_math_free,
                                              proof_box, typst_escape, typst_escape_str,
                                              typst_raw_block)
from adaptive_harness.research.synthesis import identity_proposition, plan_research
from adaptive_harness.research.proof import (ProofError, ProofReceipt, ProofRunner,
                                             scan_for_approximations)
from adaptive_harness.research.roles import (DIVISION_SPECS, FALSIFICATION_VERDICT, Clearance,
                                             Division, DivisionSpec, FalsificationVerdict,
                                             ResearchAgent, leader_agent_id, worker_agent_id)
from adaptive_harness.research.swarm import (DIRECTOR_ID, GAP_ROUTING, ResearchOutcome,
                                             ResearchSwarm, ResearchWorkspace, SwarmConfig,
                                             topic_slug)
from adaptive_harness.research.typst import CompileResult, TypstCompiler, TypstError, resolve_typst

__all__ = [
    "ACTIONS", "EXIT_COUNTEREXAMPLE", "EXIT_HOLDS", "ClaimAdjudication", "ClaimLedger", "Clearance",
    "CommLedger", "CompileResult", "ConvergenceCycle", "ConvergenceOutcome",
    "DIRECTOR_ID", "DIVISION_SPECS", "Division", "DivisionSpec", "ExperimentError",
    "ExperimentReceipt", "ExperimentRunner", "FALSIFICATION_VERDICT", "FalsificationVerdict",
    "GAP_ROUTING", "Invariant", "InvariantGate",
    "InvariantStatus", "LedgerEntry", "LedgerError", "PaperBuilder", "PaperInputs", "ParsedClaim",
    "Prediction",
    "ProofError", "ProofReceipt", "ProofRunner", "Proposition", "RelentlessConvergenceLoop",
    "ResearchAgent",
    "ResearchOutcome", "ResearchSwarm", "ResearchWorkspace", "StopReason", "SwarmConfig",
    "TopicPlan", "TypstCompiler", "TypstError", "Verdict", "canonical_json",
    "escape_math_free", "identity_proposition", "leader_agent_id",
    "mean_confidence_interval_95", "parse_claim", "plan_research",
    "proof_box", "resolve_typst", "scan_for_approximations", "sha256_file", "sha256_text",
    "topic_slug", "typst_escape", "typst_escape_str", "typst_raw_block",
    "verdict_from_exit", "worker_agent_id", "write_cycle_history",
]
