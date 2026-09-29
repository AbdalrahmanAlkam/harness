"""Permission rules: an ordered allow/deny list over tool calls.

`.harness/rules.json` holds rules that are evaluated before any tool runs. They
exist because a safety *profile* is a coarse dial — turbo asks nothing, strict
asks about everything — and a real team needs the middle: always allow `pytest`,
always refuse `rm -rf`, ask about anything touching `.env`.

Three properties matter more than the matching itself:

- **First match wins, and deny wins ties.** A rule that is easy to write wrong
  is a rule that will be written wrong.
- **Enforcement is at the safety gate**, so a permissive profile cannot route
  around a rule. A rule that a `--safety-profile turbo` launch can bypass is
  not a rule.
- **The rules are hot-reloadable.** Editing them should take effect on the next
  call without restarting a run that may be hours old.

The matching is a regex over the tool name and its *normalized* arguments, so a
rule reads like the command it is about rather than like a data structure.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

#: The rule file, in the project. A repository's own policy travels with it.
RULES_PATH = Path(".harness") / "rules.json"


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"
    #: No rule matched. Distinct from ASK: "ask" is a rule speaking, while this
    #: is the rule set declining to have an opinion. Treating them the same made
    #: an absent rules file interrupt every single tool call.
    NONE = "none"


@dataclass(frozen=True)
class Rule:
    """One rule: a pattern over a tool call, and what to do when it matches."""

    pattern: str
    decision: Decision
    #: An optional human sentence explaining why, shown when the rule bites.
    reason: str = ""
    tools: tuple[str, ...] = ()
    #: Rule order in the file. Earlier wins.
    index: int = 0

    def __post_init__(self) -> None:
        if self.decision is Decision.DENY and not self.reason:
            # A silent refusal is indistinguishable from a bug to the user, and
            # the person who wrote the rule is the one who can explain it.
            raise ValueError(
                f"Rule {self.pattern!r} denies, so it must say why. A user seeing "
                f"an unexplained refusal cannot tell a policy from a fault.")
        try:
            re.compile(self.pattern)
        except re.error as exc:
            raise ValueError(f"Rule pattern {self.pattern!r} is not a valid regex: {exc}") from exc

    def matches(self, tool: str, normalized: str) -> bool:
        if self.tools and tool not in self.tools:
            return False
        return bool(re.search(self.pattern, normalized, re.I))


@dataclass
class RuleSet:
    """An ordered rule list. Absent or malformed means no rules, not a crash."""

    rules: List[Rule] = field(default_factory=list)
    error: str = ""

    def decide(self, tool: str, arguments: Dict[str, Any]) -> Decision:
        """First match wins. Deny wins a tie on the same pattern."""
        normalized = normalize(tool, arguments)
        best: Optional[tuple[int, Rule]] = None
        for rule in self.rules:
            if not rule.matches(tool, normalized):
                continue
            if best is None or rule.index < best[0]:
                best = (rule.index, rule)
            elif rule.index == best[0] and rule.decision is Decision.DENY:
                # Two rules at the same position, one denying: the stricter wins.
                best = (rule.index, rule)
        return best[1].decision if best else Decision.NONE

    def matched_rule(self, tool: str, arguments: Dict[str, Any]) -> Optional[Rule]:
        normalized = normalize(tool, arguments)
        best: Optional[tuple[int, Rule]] = None
        for rule in self.rules:
            if rule.matches(tool, normalized) and (best is None or rule.index < best[0]):
                best = (rule.index, rule)
        return best[1] if best else None

    def __len__(self) -> int:
        return len(self.rules)

    @classmethod
    def load(cls, path: Path | str = RULES_PATH) -> "RuleSet":
        """Read rules from disk. A broken file is reported, never fatal.

        A rules file that fails to parse must not stop the harness, and it must
        not silently become "no rules" either -- that would drop a user's
        policy without telling them.
        """
        target = Path(path)
        if not target.is_file():
            return cls()
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return cls(error=f"{target} is not valid JSON: {exc}")
        if not isinstance(payload, dict) or not isinstance(payload.get("rules"), list):
            return cls(error=f"{target} must contain a 'rules' list")

        rules: List[Rule] = []
        for index, entry in enumerate(payload["rules"]):
            if not isinstance(entry, dict):
                return cls(error=f"rule {index} in {target} is not an object")
            try:
                rules.append(Rule(
                    pattern=str(entry.get("pattern", "")),
                    decision=Decision(str(entry.get("decision", "ask")).lower()),
                    reason=str(entry.get("reason", "")),
                    tools=tuple(str(name) for name in entry.get("tools", ())),
                    index=index,
                ))
            except ValueError as exc:
                return cls(error=f"rule {index} in {target}: {exc}")
        return cls(rules=rules)

    def save(self, path: Path | str = RULES_PATH) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"rules": [
            {"pattern": rule.pattern, "decision": rule.decision.value,
             "reason": rule.reason, **({"tools": list(rule.tools)} if rule.tools else {})}
            for rule in self.rules]}, indent=2) + "\n", encoding="utf-8")
        return target


#: Argument keys whose values are commands or paths, and so are what a rule
#: about a destructive action is actually written against.
_SENSITIVE_KEYS = ("command", "path", "file_path", "target", "text", "content",
                   "query", "url")

#: Values collapsed so a rule pattern cannot be defeated by an argument the
#: model padded with spaces.
_WHITESPACE = re.compile(r"\s+")


def normalize(tool: str, arguments: Dict[str, Any]) -> str:
    """Render a tool call as the single string a rule's regex is matched against.

    `run_bash {"command": "rm  -rf  /tmp/x"}` becomes
    `run_bash rm -rf /tmp/x` -- whitespace collapsed, because a rule that a
    model can defeat by typing an extra space is not a rule.
    """
    parts: List[str] = []
    for key in _SENSITIVE_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value:
            parts.append(f"{key}={_WHITESPACE.sub(' ', value.strip())}")
    for key, value in sorted(arguments.items()):
        if key in _SENSITIVE_KEYS or value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{key}={_WHITESPACE.sub(' ', str(value).strip())}")
    return _WHITESPACE.sub(" ", f"{tool} {' '.join(parts)}".strip())


def default_rules() -> List[Rule]:
    """A starting policy a user can edit, offered rather than imposed."""
    return [
        Rule(pattern=r"rm\s+-[a-z]*r", decision=Decision.DENY,
             reason="Recursive deletion is refused by policy.", index=0),
        Rule(pattern=r"\.env\b", decision=Decision.ASK,
             reason="This touches an environment file, which may hold secrets.", index=1),
        Rule(pattern=r"git\s+push", decision=Decision.ASK,
             reason="Pushing is not reversible by pulling.", index=2),
        Rule(pattern=r"\bpytest\b", decision=Decision.ALLOW,
             reason="Running the tests is the point.", index=3),
    ]
