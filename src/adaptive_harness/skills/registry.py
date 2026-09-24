"""Typed skill packages: compact instructions, tool bindings, and verifiable gates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaseSkill:
    name: str
    title: str
    category: str
    trigger: str
    instructions: str
    tools: tuple[str, ...]
    invariants: tuple[str, ...]
    icon: str = "✦"
    source: str = "built-in"

    def __post_init__(self) -> None:
        if not self.name.isidentifier() or not all((self.title, self.category, self.trigger, self.instructions)):
            raise ValueError("Skill metadata is incomplete or its name is invalid")
        if not self.tools:
            raise ValueError(f"Skill {self.name} must bind at least one tool")


def _skill(name, title, category, trigger, instructions, tools, invariants, icon="✦") -> BaseSkill:
    return BaseSkill(name, title, category, trigger, instructions,
                     tuple(tools.split()), tuple(invariants.split()), icon)


_INSPECT = "read_file search_files list_directory"
_EDIT = _INSPECT + " edit_file write_file"
_CODE = _EDIT + " run_pytest run_bash"
_SCIENCE = _INSPECT + " calculate run_python_repl verify_equation check_convergence plot_terminal"


BUILTIN_SKILLS: tuple[BaseSkill, ...] = (
    _skill("refactor_clean_code", "Clean Code Refactor", "Software Engineering",
        "Refactor source code, remove duplication or dead code, improve SOLID design and method boundaries.",
        "Map callers and tests before editing. Change one responsibility at a time, preserve observable behavior, and show a focused diff. Run the narrowest relevant test suite after each structural change.",
        _CODE, "edit_validated tests_green", "🛠"),
    _skill("ast_migration", "AST Migration", "Software Engineering",
        "Migrate deprecated Python APIs, upgrade language syntax, or automate AST transformations.",
        "Find every call site and compatibility constraint. Read precise symbols, transform syntax with AST awareness, preserve imports and semantics, and compile modified modules before targeted tests.",
        _CODE, "edit_validated tests_green", "🌳"),
    _skill("architecture_design", "Architecture Design", "Software Engineering",
        "Design multi-module architecture, interfaces, dependency boundaries, services, and component diagrams.",
        "Inspect existing contracts first. State responsibilities, data flow, failure paths, migration sequence, and trade-offs. Keep interfaces explicit and make assumptions visible rather than inventing repository facts.",
        _INSPECT + " run_bash", "evidence_checked", "🏛"),
    _skill("git_worktree_manager", "Git Worktree Manager", "Software Engineering",
        "Manage git branches, worktrees, bisect regressions, rebase, and inspect commits safely.",
        "Inspect status and branch names before changing Git state. Prefer reversible operations, preserve uncommitted work, and confirm the resulting branch/worktree topology with Git commands.",
        _INSPECT + " run_bash", "git_checked", "🌿"),
    _skill("docstring_api_spec", "API & Docstring Spec", "Software Engineering",
        "Write OpenAPI or Swagger specs, typed API documentation, Pydantic signatures, and docstrings.",
        "Read implementation signatures and examples. Document inputs, outputs, errors, and compatibility precisely; update the relevant spec or docstring and validate generated syntax or examples.",
        _CODE, "edit_validated", "📘"),

    _skill("pytest_tdd_loop", "Pytest TDD Loop", "Testing & Reliability",
        "Write pytest unit tests, practice test-driven development, fix failing assertions, and verify regressions.",
        "Start with the smallest failing behavior and a meaningful assertion. Implement only enough to pass, rerun the focused test, then refactor and report the exact pass/fail evidence.",
        _CODE, "tests_green", "🧪"),
    _skill("concurrency_deadlock_debugger", "Concurrency Debugger", "Testing & Reliability",
        "Debug asyncio, race conditions, mutex locks, deadlocks, scheduling, and thread safety.",
        "Build a timeline of actors, locks, and awaited operations. Check lock ordering and cancellation paths; create a deterministic reproducer or stress test before claiming a race is fixed.",
        _CODE + " run_python_repl", "evidence_checked tests_green", "🧵"),
    _skill("fuzz_edge_case_tester", "Fuzz & Edge Cases", "Testing & Reliability",
        "Design property-based tests, Hypothesis fuzzing, boundaries, empty inputs, NaN, and malformed data cases.",
        "Define invariants and generators before writing random cases. Cover empty, extreme, and invalid inputs; reproduce a failure with a fixed seed and run targeted tests after the fix.",
        _CODE + " run_python_repl", "tests_green", "🎲"),
    _skill("memory_leak_profiler", "Memory Leak Profiler", "Testing & Reliability",
        "Investigate tracemalloc snapshots, reference cycles, object growth, memory leaks, and heap use.",
        "Measure a baseline and repeated workload under the same conditions. Compare allocations by traceback, identify retained references, make a minimal change, and remeasure growth.",
        _CODE + " run_python_repl", "execution_checked", "📈"),

    _skill("symbolic_math_solver", "Symbolic Math Solver", "Mathematics & Science",
        "Solve algebra, calculus, equations, integrals, ODEs, exact roots, matrix identities, and symbolic proofs.",
        "Formulate the exact expression with SymPy; preserve domain restrictions and branches. Substitute proposed solutions into the original equation, check denominators, and separate exact claims from numerical approximations.",
        _SCIENCE, "symbolic_checked", "∑"),
    _skill("numerical_simulation", "Numerical Simulation", "Mathematics & Science",
        "Run NumPy or SciPy simulations, differential equations, Monte Carlo, convergence and tolerance analysis.",
        "State units, initial and boundary conditions, tolerance, and random seed. Run a deterministic baseline, vary resolution, and report residuals or confidence intervals rather than treating one run as proof.",
        _SCIENCE, "numerical_checked", "🧮"),
    _skill("statistical_data_analysis", "Statistical Analysis", "Mathematics & Science",
        "Analyze datasets with descriptive statistics, hypothesis tests, ANOVA, distributions, and outliers.",
        "Inspect sample size, missingness, and assumptions before choosing a test. Compute effect sizes and uncertainty with Python; distinguish association from causation and report reproducible parameters.",
        _SCIENCE, "numerical_checked", "📊"),
    _skill("algorithm_complexity_analyzer", "Algorithm Complexity", "Mathematics & Science",
        "Analyze Big-O time and space complexity, bottlenecks, profiling, and algorithmic scalability.",
        "Derive bounds from loops, recursion, and data structures; state input assumptions. Check worst and amortized cases and corroborate a bottleneck with focused benchmarks when code is available.",
        _INSPECT + " run_python_repl run_bash", "evidence_checked", "⏱"),

    _skill("literature_synthesizer", "Literature Synthesizer", "Research & Synthesis",
        "Synthesize papers, technical literature, citations, primary sources, and competing claims.",
        "Collect independent primary sources and record author, date, and URL or local path. Separate findings from inference, compare disagreements explicitly, and attach a source to each material claim.",
        _INSPECT + " web_search", "sources_checked", "🔎"),
    _skill("comparative_benchmarking", "Comparative Benchmarking", "Research & Synthesis",
        "Benchmark algorithms or systems across throughput, latency, memory, and reproducible comparisons.",
        "Use the same workload and environment for every candidate. Include warm-up, repetitions, dispersion, input sizes, and units; show a compact table and call out measurement limitations.",
        _INSPECT + " run_bash run_python_repl calculate plot_terminal", "execution_checked", "🏁"),
    _skill("technical_rfc_author", "Technical RFC Author", "Research & Synthesis",
        "Write engineering RFCs, proposals, design trade-offs, rollout plans, and rollback plans.",
        "Inspect current constraints and stakeholders. Present goals, alternatives, interfaces, risks, rollout, rollback, and open questions; ground repository-specific claims in inspected evidence.",
        _INSPECT + " web_search write_file", "evidence_checked", "📝"),

    _skill("docker_containerizer", "Docker Containerizer", "DevOps & Infrastructure",
        "Create Dockerfiles, multi-stage container builds, .dockerignore, compose services, and image optimization.",
        "Inspect runtime dependencies and entrypoints. Use a minimal multi-stage image, a non-root runtime, explicit health behavior, and a .dockerignore; validate the build if Docker is available.",
        _CODE, "edit_validated", "🐳"),
    _skill("ci_cd_pipeline_builder", "CI/CD Pipeline Builder", "DevOps & Infrastructure",
        "Build GitHub Actions pipelines, CI matrix tests, lint gates, caching, and release workflows.",
        "Inspect project commands and supported runtimes. Pin action versions, give jobs least privilege, cache only stable dependencies, and validate workflow syntax plus a local equivalent of key checks.",
        _CODE, "edit_validated", "⚙"),
    _skill("sql_query_optimizer", "SQL Query Optimizer", "DevOps & Infrastructure",
        "Optimize SQL EXPLAIN plans, indexes, N+1 queries, joins, and schema migrations.",
        "Inspect schema and representative queries. Compare EXPLAIN plans and selectivity before/after; preserve semantics and transaction safety, and test migrations on disposable data first.",
        _INSPECT + " run_bash run_python_repl edit_file write_file", "evidence_checked", "🗄"),

    _skill("security_audit_scanner", "Security Audit Scanner", "Security & Type Auditing",
        "Audit OWASP risks, injection, path traversal, auth boundaries, secrets, and unsafe shell use.",
        "Trace untrusted input to sensitive sinks and identify exploit conditions. Inspect code without mutating it, distinguish confirmed findings from hypotheses, and include impact plus a minimal reproduction.",
        _INSPECT + " run_bash", "evidence_checked", "🛡"),
    _skill("dependency_vulnerability_checker", "Dependency Vulnerabilities", "Security & Type Auditing",
        "Audit CVEs, pip-audit findings, lockfiles, pinned versions, and software supply-chain risks.",
        "Identify exact installed and locked versions before matching advisories. Use a vulnerability scanner when available, verify affected ranges and fixes, and avoid upgrading unrelated dependencies.",
        _INSPECT + " run_bash", "dependency_checked", "🔐"),
    _skill("type_safety_enforcer", "Type Safety Enforcer", "Security & Type Auditing",
        "Run MyPy or Pyright, strengthen Protocols and Generics, and remove unsafe Any annotations.",
        "Inspect public boundaries and current checker configuration. Add precise types without changing runtime behavior; run the strict checker or document why it is unavailable, then test affected paths.",
        _CODE, "edit_validated", "🔷"),
)

BUILTIN_BY_NAME = {skill.name: skill for skill in BUILTIN_SKILLS}
assert len(BUILTIN_BY_NAME) == 22
