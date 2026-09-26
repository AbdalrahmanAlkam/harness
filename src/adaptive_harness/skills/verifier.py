"""Evidence checks for the built-in skill invariants."""

from __future__ import annotations

from dataclasses import dataclass
import shlex
from typing import Iterable, Mapping
from adaptive_harness.skills.registry import BaseSkill


@dataclass(frozen=True)
class SkillVerification:
    skill: str
    satisfied: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def verified(self) -> bool:
        return not self.missing


class SkillVerifier:
    """Evaluate observed tool evidence; an LLM assertion alone never satisfies a check."""

    #: Which tool outcomes can discharge each invariant. Used both to verify a run
    #: and, below, to decide whether an invariant is reachable at all.
    CHECK_TOOLS: Mapping[str, frozenset[str]] = {
        "edit_validated": frozenset({"edit_file", "write_file"}),
        "tests_green": frozenset({"run_pytest", "run_bash"}),
        "evidence_checked": frozenset({"read_file", "search_files", "list_directory", "run_bash"}),
        "git_checked": frozenset({"run_bash"}),
        "symbolic_checked": frozenset({"verify_equation"}),
        "numerical_checked": frozenset({"check_convergence", "run_python_repl", "calculate"}),
        "sources_checked": frozenset({"web_search", "read_file"}),
        "execution_checked": frozenset({"run_bash", "run_python_repl", "run_pytest"}),
        "dependency_checked": frozenset({"run_bash"}),
    }

    #: Invariants that need a specific command, not merely a tool. A worker holding
    #: ``run_bash`` can run ``git status`` or ``pip-audit``, so the tool is the
    #: gate; whether the model chooses to is an evidence question, not an
    #: applicability one.
    COMMAND_INVARIANTS = frozenset({"tests_green", "git_checked", "dependency_checked"})

    def reachable_invariants(self, skills: Iterable[BaseSkill],
                             tool_names: Iterable[str]) -> frozenset[str]:
        """The invariants this tool set can possibly discharge.

        Injecting a skill whose completion checks the worker has no instrument for
        is worse than injecting nothing: the worker is told to satisfy
        ``tests_green`` with no ``run_pytest`` and no permission to run pytest, and
        then fails a verification gate it was never capable of passing. The failure
        is reported as ``skill_verification_failed``, which reads as "the agent did
        not do the work" when in fact the work was made impossible up front.
        """
        available = set(tool_names)
        reachable: set[str] = set()
        for invariant in self.CHECK_TOOLS:
            if self.CHECK_TOOLS[invariant] & available:
                reachable.add(invariant)
        return frozenset(reachable)

    def applicable(self, skills: Iterable[BaseSkill], tool_names: Iterable[str],
                   ) -> tuple[BaseSkill, ...]:
        """Skills whose every completion check is reachable with these tools.

        A skill with no invariants is always applicable. A skill with an
        unreachable invariant is dropped entirely rather than partially applied:
        half a checklist is not a checklist.
        """
        reachable = self.reachable_invariants(skills, tool_names)
        return tuple(skill for skill in skills
                     if not (set(skill.invariants) - reachable))

    def unreachable(self, skills: Iterable[BaseSkill], tool_names: Iterable[str],
                    ) -> dict[str, tuple[str, ...]]:
        """Which checks each skill could never have satisfied here, and why."""
        reachable = self.reachable_invariants(skills, tool_names)
        return {skill.name: tuple(set(skill.invariants) - reachable)
                for skill in skills if set(skill.invariants) - reachable}

    def verify(self, skills: Iterable[BaseSkill], observations: Iterable[Mapping]) -> tuple[SkillVerification, ...]:
        # A later failed check invalidates earlier evidence from that tool.
        latest = {}
        for observation in observations:
            latest[observation.get("name")] = observation
            if observation.get("name") in {"edit_file", "write_file"} and observation.get("success"):
                latest.pop("run_pytest", None)
                latest.pop("run_bash", None)
        records = tuple(latest.values())
        successful = {str(record.get("name")) for record in records if record.get("success")}
        def command_tokens(record: Mapping) -> list[str]:
            try:
                return shlex.split(str(record.get("arguments", {}).get("command", "")))
            except ValueError:
                return []

        pytest_shell = any(record.get("success") and record.get("name") == "run_bash" and
                           (tokens and (tokens[0].endswith("pytest") or tokens[:3] == ["python", "-m", "pytest"]))
                           for record in records if (tokens := command_tokens(record)))
        checks = {
            "edit_validated": bool(successful & {"edit_file", "write_file"}),
            "tests_green": "run_pytest" in successful or pytest_shell,
            "evidence_checked": bool(successful & {"read_file", "search_files", "list_directory", "run_bash"}),
            "git_checked": any(record.get("success") and record.get("name") == "run_bash" and
                               "git " in str(record.get("arguments", {}).get("command", "")) for record in records),
            "symbolic_checked": "verify_equation" in successful,
            "numerical_checked": bool(successful & {"check_convergence", "run_python_repl", "calculate"}),
            "sources_checked": bool(successful & {"web_search", "read_file"}),
            "execution_checked": bool(successful & {"run_bash", "run_python_repl", "run_pytest"}),
            "dependency_checked": any(record.get("success") and record.get("name") == "run_bash" and
                                      (tokens[:1] == ["pip-audit"] or tokens[:2] in
                                       (["safety", "check"], ["npm", "audit"], ["cargo", "audit"]))
                                      for record in records if (tokens := command_tokens(record))),
        }
        return tuple(SkillVerification(skill.name,
                     tuple(key for key in skill.invariants if checks.get(key, False)),
                     tuple(key for key in skill.invariants if not checks.get(key, False))) for skill in skills)
