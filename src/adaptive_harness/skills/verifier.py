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
