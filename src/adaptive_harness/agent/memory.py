"""Durable memory, in three tiers with three different trust levels.

The tiers differ in *who wrote them* and therefore in how much they should be
trusted when they are put back in front of a model:

- **session** — what happened in this conversation. Already exists as
  `SessionStore`.
- **project** — `AGENTS.md` and `.harness/instructions/*.md`. Written by the
  team, committed with the code, and read as authoritative.
- **agent-authored** — the model noticed something and wrote it down. This is
  the tier that needs care, because a model that can silently change what it
  will be told next turn can, over a few turns, set its own instructions.

So a proposal never takes effect silently. It is scored for whether it is
project-general or specific to the task at hand, and the operator is shown a
one-line diff to accept. Accepted memories are namespaced, revocable, and
listed. That is the whole design: a memory the operator cannot see is not a
memory, it is a slow-motion instruction.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

#: Where accepted memories live. Under the config directory rather than the
#: workspace, so a repository never accumulates machine-specific state and a
#: memory cannot be committed by accident.
MEMORY_PATH = Path("memory.json")

#: Anything naming a secret, a credential, or a machine-specific path is not
#: worth remembering, and is not safe to write to disk.
#:
#: The separator is optional because prose writes "API key" with a space while
#: config writes "api_key" with an underscore -- a pattern that matched only one
#: of those let a key straight through.
_SENSITIVE = re.compile(
    r"(?i)\b(?:api[ _-]?key|secret|password|passwd|token|credential|bearer|"
    r"private[ _-]?key|access[ _-]?key|session[ _-]?id|auth[ _-]?token)\b")

#: A bare credential, with no word naming it. Provider key formats are long
#: runs of mixed case and digits, or a known prefix, and the sentence around
#: them is not always going to say "API key".
_KEY_SHAPED = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{12,}|sk-or-v1-[A-Za-z0-9]{12,}|"
    r"sk-ant-[A-Za-z0-9_-]{12,}|gsk_[A-Za-z0-9]{12,}|"
    r"AIza[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,})\b")

#: Absolute paths outside the workspace are machine-specific and go stale.
_ABSOLUTE_PATH = re.compile(r"(?:^|\s)(?:/home/|/Users/|/root/|[A-Z]:\\\\)")

#: A memory longer than this is a note, not a memory.
MAX_MEMORY_CHARS = 400


@dataclass
class Memory:
    """One remembered fact."""

    id: str
    text: str
    #: project | task. A task-specific memory must not be promoted to a
    #: project one, or the model generalises from a single observation.
    scope: str = "project"
    #: Which task it was learned in, for a task-scoped memory.
    origin: str = ""
    accepted: bool = False
    created: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "text": self.text, "scope": self.scope,
                "origin": self.origin, "accepted": self.accepted,
                "created": self.created}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Memory":
        return cls(id=str(payload.get("id", "")),
                   text=str(payload.get("text", "")),
                   scope=str(payload.get("scope", "project")),
                   origin=str(payload.get("origin", "")),
                   accepted=bool(payload.get("accepted", False)),
                   created=float(payload.get("created", 0.0)))


@dataclass
class Proposal:
    """A memory the model would like to keep, pending the operator's answer."""

    memory: Memory
    #: The classifier's view: how much this looks like a durable fact about the
    #: project rather than an observation about the current task.
    generality: float
    reason: str = ""
    #: The one-line diff shown to the operator.
    preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"memory": self.memory.to_dict(), "generality": self.generality,
                "reason": self.reason, "preview": self.preview}


def is_rememberable(text: str) -> tuple[bool, str]:
    """Whether a proposed memory is worth storing at all.

    A memory that records a secret writes a secret to disk and then replays it
    into every later session. Refusing here is the only place that can prevent
    it, so it happens before anything is written.
    """
    stripped = " ".join(str(text or "").split())
    if not stripped:
        return False, "empty"
    if _SENSITIVE.search(stripped):
        return False, "mentions a credential or secret"
    if _KEY_SHAPED.search(stripped):
        return False, "looks like a credential"
    if _ABSOLUTE_PATH.search(stripped):
        return False, "names a machine-specific absolute path"
    if len(stripped) > MAX_MEMORY_CHARS:
        return False, f"longer than {MAX_MEMORY_CHARS} characters"
    return True, ""


def score_generality(text: str, task: str = "") -> tuple[float, str]:
    """How much this looks like a durable project fact rather than a one-off.

    Deliberately lexical and local. A model scoring its own memory would cost a
    token on every proposal and could be talked into a high score; the point of
    the score is to route the *decision* to a human, not to replace one.
    """
    stripped = " ".join(str(text or "").split()).lower()
    score = 0.4
    reasons: List[str] = []
    if re.search(r"\b(?:always|never|prefer|convention|we use|the project)\b", stripped):
        score += 0.35
        reasons.append("states a rule rather than an observation")
    if re.search(r"\b(?:this time|today|for now|temporar|right now)\b", stripped):
        score -= 0.4
        reasons.append("sounds time-specific")
    if task and re.search(r"\b(?:it|the issue|the bug)\b", stripped) and task.lower() in stripped[:80]:
        score -= 0.25
        reasons.append("restates the current task")
    if re.search(r"\b(?:because|so that|to avoid)\b", stripped):
        score += 0.1
        reasons.append("records a reason, not just a fact")
    return max(0.0, min(1.0, score)), "; ".join(reasons) or "no strong signal either way"


class MemoryStore:
    """The accepted memories, and the proposals waiting on an answer."""

    def __init__(self, directory: Path | str | None = None) -> None:
        base = Path(directory) if directory else _default_dir()
        self.path = base / MEMORY_PATH
        self.memories: Dict[str, Memory] = {}
        self.proposals: Dict[str, Proposal] = {}
        self.load_error: str = ""
        self.load()

    # -- persistence -------------------------------------------------------

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.load_error = f"{self.path} could not be read: {exc}"
            return
        if not isinstance(payload, dict):
            self.load_error = f"{self.path} does not contain a JSON object"
            return
        for entry in payload.get("memories", []):
            if isinstance(entry, dict):
                memory = Memory.from_dict(entry)
                if memory.id and memory.accepted:
                    self.memories[memory.id] = memory

    def save(self) -> None:
        """Atomic, so a crash cannot leave a half-written memory file.

        A truncated memory file would load as empty and silently drop every
        memory the user had accepted.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"memories": [memory.to_dict()
                                for memory in self.memories.values()]}
        handle, temp = _mkstemp(self.path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.path)
            os.chmod(self.path, 0o600)
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise

    # -- proposals ---------------------------------------------------------

    def propose(self, text: str, task: str = "", origin: str = "") -> Proposal:
        """Record a memory the model asked for. It is not in effect yet."""
        ok, why = is_rememberable(text)
        if not ok:
            raise ValueError(f"Not worth remembering ({why}): {text!r}")
        generality, reason = score_generality(text, task)
        memory = Memory(id=f"m{len(self.memories) + len(self.proposals) + 1}",
                        text=" ".join(str(text).split()),
                        scope="task" if generality < 0.5 else "project",
                        origin=origin or task)
        proposal = Proposal(
            memory=memory, generality=generality, reason=reason,
            preview=_preview(memory, generality))
        self.proposals[memory.id] = proposal
        return proposal

    def accept(self, memory_id: str) -> Optional[Memory]:
        """Accept a proposal. This is the only path by which a memory applies."""
        proposal = self.proposals.pop(memory_id, None)
        if proposal is None:
            return None
        proposal.memory.accepted = True
        self.memories[proposal.memory.id] = proposal.memory
        self.save()
        return proposal.memory

    def decline(self, memory_id: str) -> bool:
        return self.proposals.pop(memory_id, None) is not None

    def forget(self, memory_id: str) -> bool:
        """Revoke a memory. Accepted memories are the operator's to remove."""
        if self.memories.pop(memory_id, None) is not None:
            self.save()
            return True
        return self.decline(memory_id)

    def clear(self) -> int:
        count = len(self.memories)
        self.memories.clear()
        self.proposals.clear()
        self.save()
        return count

    # -- what the model is told --------------------------------------------

    def recall(self, task: str = "", *, include_task_scope: bool = True) -> List[str]:
        """The memories to put in front of the model for this task.

        Project memories always; task memories only when the current task
        plausibly relates, because a task memory is an observation about one
        piece of work and repeating it for unrelated work is noise.
        """
        texts: List[str] = []
        task_lower = str(task or "").lower()
        for memory in sorted(self.memories.values(), key=lambda item: item.id):
            if memory.scope == "task":
                if not include_task_scope:
                    continue
                if task_lower and not _related(task_lower, memory.text.lower()):
                    continue
            texts.append(memory.text)
        return texts

    def as_context(self, task: str = "") -> str:
        texts = self.recall(task)
        if not texts:
            return ""
        body = "\n".join(f"- {text}" for text in texts)
        return f"[remembered for this project]\n{body}"

    def describe(self) -> str:
        lines = [f"{len(self.memories)} memory(ies) in effect"]
        for memory in sorted(self.memories.values(), key=lambda item: item.id):
            lines.append(f"  {memory.id} [{memory.scope}] {memory.text}")
        if self.proposals:
            lines.append(f"{len(self.proposals)} awaiting your answer:")
            for proposal in self.proposals.values():
                lines.append(f"  {proposal.memory.id} {proposal.preview}")
        if self.load_error:
            lines.append(f"  WARNING: {self.load_error}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.memories)


def _related(task: str, memory: str) -> bool:
    """Whether a task-scoped memory plausibly applies to this task.

    A shared content word is enough. This is a relevance hint for *including*
    context, and a false positive costs a little context -- the opposite
    failure from excluding a memory that applies.
    """
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "with",
            "is", "it", "this", "that", "be", "are", "at", "by"}
    task_words = {word for word in re.findall(r"[a-z0-9]+", task) if word not in stop and len(word) > 2}
    memory_words = {word for word in re.findall(r"[a-z0-9]+", memory) if word not in stop and len(word) > 2}
    return bool(task_words & memory_words)


def _preview(memory: Memory, generality: float) -> str:
    """The one-line diff the operator is shown.

    It has to carry the decision, not the data: what would be remembered, at
    what scope, and how confident the classifier is that it is durable.
    """
    confidence = "likely durable" if generality >= 0.5 else "looks task-specific"
    return (f"{'+'} {memory.text} "
            f"({memory.scope}, {confidence}, {generality:.2f})")


def _default_dir() -> Path:
    from adaptive_harness.data.config import DEFAULT_CONFIG_DIR

    return Path(DEFAULT_CONFIG_DIR)


def _mkstemp(directory: Path) -> tuple[int, str]:
    import tempfile

    return tempfile.mkstemp(dir=str(directory), suffix=".tmp")
