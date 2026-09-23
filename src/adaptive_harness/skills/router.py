"""Local routing across craft skills; no foundation-model tokens are used."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import time
from typing import Mapping

from adaptive_harness.skills.registry import BaseSkill


ALIASES: dict[str, tuple[str, ...]] = {
    "refactor_clean_code": ("refactor", "clean code", "dead code", "duplication"),
    "ast_migration": ("ast migration", "deprecated api", "upgrade syntax"),
    "architecture_design": ("architecture", "system design", "component design"),
    "git_worktree_manager": ("git worktree", "git bisect", "rebase", "branch"),
    "docstring_api_spec": ("docstring", "openapi", "swagger", "api spec"),
    "pytest_tdd_loop": ("pytest", "unit test", "test-driven", "failing test", "regression test"),
    "concurrency_deadlock_debugger": ("deadlock", "race condition", "asyncio", "mutex", "thread safety"),
    "fuzz_edge_case_tester": ("fuzz", "property-based", "hypothesis", "edge case"),
    "memory_leak_profiler": ("memory leak", "tracemalloc", "heap growth"),
    "symbolic_math_solver": ("sympy", "equation", "integral", "derivative", "algebra", "solve for x", "symbolic"),
    "numerical_simulation": ("simulation", "monte carlo", "convergence", "differential equation"),
    "statistical_data_analysis": ("statistics", "statistical", "anova", "t-test", "outlier", "dataset"),
    "algorithm_complexity_analyzer": ("big-o", "complexity", "asymptotic", "bottleneck"),
    "literature_synthesizer": ("literature", "papers", "citations", "sources", "research review"),
    "comparative_benchmarking": ("benchmark", "throughput", "latency comparison"),
    "technical_rfc_author": ("rfc", "proposal", "trade-off", "design document"),
    "docker_containerizer": ("docker", "container", "dockerfile", "compose"),
    "ci_cd_pipeline_builder": ("ci/cd", "github actions", "workflow", "pipeline"),
    "sql_query_optimizer": ("sql", "explain plan", "query optimization", "index"),
    "security_audit_scanner": ("security audit", "owasp", "injection", "path traversal", "vulnerability"),
    "dependency_vulnerability_checker": ("cve", "pip-audit", "dependency vulnerability", "supply chain"),
    "type_safety_enforcer": ("mypy", "pyright", "type checking", "type safety", "untyped"),
}


@dataclass(frozen=True)
class SkillSelection:
    skills: tuple[BaseSkill, ...]
    probabilities: dict[str, float]
    confidence: float
    selection: str
    backend: str
    latency_ms: float


class SkillRouter:
    def __init__(self, *, threshold: float = 0.70):
        self.threshold = threshold

    def route(self, task: str, catalog: Mapping[str, BaseSkill], backend=None,
              forced: tuple[str, ...] = ()) -> SkillSelection:
        start = time.perf_counter()
        if forced:
            missing = set(forced) - set(catalog)
            if missing:
                raise ValueError(f"Unknown skill: {', '.join(sorted(missing))}")
            return SkillSelection(tuple(catalog[name] for name in forced),
                                  {name: 1.0 / len(forced) for name in forced}, 1.0,
                                  "forced", "manual", (time.perf_counter() - start) * 1000)
        if not catalog:
            return SkillSelection((), {}, 0.0, "none", "none", 0.0)
        lexical = {name: self._score(task, skill) for name, skill in catalog.items()}
        candidates = sorted(catalog, key=lambda name: lexical[name], reverse=True)[:26]
        engine = getattr(backend, "engine", None) if backend is not None else None
        semantic = getattr(backend, "name", None) == "semif" and engine is not None
        if max(lexical.values()) <= 0 and not semantic:
            return SkillSelection((), {}, 0.0, "none", "local", (time.perf_counter() - start) * 1000)
        engine = getattr(backend, "engine", None) if backend is not None else None
        if getattr(backend, "name", None) == "semif" and engine is not None:
            try:
                decision = engine.decide(task, {name: catalog[name].trigger for name in candidates})
                probabilities = decision.probabilities
                source = "semif"
            except Exception:
                probabilities = self._lexical_probabilities(lexical)
                source = "local-fallback"
        else:
            probabilities = self._lexical_probabilities(lexical)
            source = "local"
        ranking = sorted(probabilities, key=probabilities.get, reverse=True)
        best = ranking[0]
        confidence = probabilities[best]
        # A lexical hit is meaningful even if the distribution is diffuse across
        # 22 candidate descriptions. Neural decisions use their own probabilities.
        if source != "semif" and lexical[best] > 0:
            confidence = max(confidence, min(0.99, 0.70 + 0.06 * lexical[best]))
        chain = (len(ranking) > 1 and probabilities[ranking[1]] >= 0.20 and
                 probabilities[ranking[1]] >= probabilities[best] * 0.55 and
                 probabilities[best] + probabilities[ranking[1]] >= self.threshold)
        if confidence < self.threshold and not chain:
            return SkillSelection((), probabilities, confidence, "none", source,
                                  (time.perf_counter() - start) * 1000)
        selected = [catalog[best]]
        if len(ranking) > 1:
            second = ranking[1]
            if chain:
                selected.append(catalog[second])
        return SkillSelection(tuple(selected), probabilities, confidence,
                              "auto", source, (time.perf_counter() - start) * 1000)

    @staticmethod
    def _score(task: str, skill: BaseSkill) -> float:
        text = task.casefold()
        aliases = ALIASES.get(skill.name, ()) + (skill.name.replace("_", " "),)
        matches = [alias for alias in aliases if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text)]
        if matches:
            return 3.0 + max(len(match.split()) for match in matches) + 0.2 * (len(matches) - 1)
        words = set(re.findall(r"[a-z]{4,}", text))
        trigger_words = set(re.findall(r"[a-z]{4,}", skill.trigger.casefold()))
        common = words & trigger_words
        return len(common) * 0.35 if len(common) >= 2 else 0.0

    @staticmethod
    def _lexical_probabilities(scores: Mapping[str, float]) -> dict[str, float]:
        peak = max(scores.values())
        exps = {name: math.exp(1.8 * (score - peak)) for name, score in scores.items()}
        total = sum(exps.values())
        return {name: value / total for name, value in exps.items()}
