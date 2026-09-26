"""Append-only, hash-chained inter-agent communication ledger.

Every message between the Executive Director, Division Leaders, and dynamically
spawned workers lands in ``research/<topic_slug>/comm_ledger.jsonl`` as a single
JSON object per line. The ledger is tamper-evident: each record carries the
SHA-256 digest of the previous record, so :func:`CommLedger.verify` can prove
that no entry was edited, reordered, or removed after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterable, Mapping, Sequence

LEDGER_FILENAME = "comm_ledger.jsonl"
_GENESIS = "0" * 64

# Canonical action vocabulary. Unknown actions are rejected so that downstream
# consumers can rely on a closed set of transitions.
ACTIONS = frozenset({
    "OBJECTIVE_SET",
    "TASK_ASSIGNED",
    "PROOF_SUBMISSION",
    "WORKER_TRAJECTORY",
    "LEAN_PROOF_SUBMISSION",
    "LEAN_PROOF_VERIFIED",
    "LEAN_PROOF_REJECTED",
    "PROOF_VERIFIED",
    "PROOF_REJECTED",
    "EXPERIMENT_SUBMISSION",
    "EXPERIMENT_VERIFIED",
    "EXPERIMENT_REJECTED",
    "EVIDENCE_RECORDED",
    "FALSIFICATION_ATTEMPT",
    "COUNTEREXAMPLE_FOUND",
    "CLEARANCE_GRANTED",
    "SPAWN_REQUEST",
    "SELF_PIVOT",
    "GATE_EVALUATION",
    "DELIVERABLE_WRITTEN",
    "STATUS_REPORT",
    # -- coordination vocabulary -------------------------------------------
    # Added with the task board, the routed message bus, and leader-initiated
    # stop control. These record *observable* actions: an assignment that was
    # leased, a request that was routed and acknowledged, a transition a worker
    # actually underwent. Hidden model reasoning and internal tokens are
    # deliberately not logged, because the ledger cannot verify them.
    "TASK_LEASED",
    "TASK_RELEASED",
    "TASK_COMPLETED",
    "TASK_FAILED",
    "TASK_BLOCKED",
    "TASK_RETRY",
    "MESSAGE_SENT",
    "MESSAGE_ACKNOWLEDGED",
    "MESSAGE_OUTCOME",
    "HELP_REQUESTED",
    "HELP_OFFERED",
    "WORKER_STATE_CHANGE",
    "STOP_REQUESTED",
    "WORKER_CANCELLED",
    "RUN_PAUSED",
    "RUN_RESUMED",
    "RUN_TERMINATED",
    "SWARM_SCALED",
    "SWARM_PIVOT",
    "PROGRESS_SUMMARY",
    "RUN_RESUMED_FROM_STATE",
})

# Actions that mean "this run was deliberately ended by a human or a leader".
# They are separated from ordinary cycle exits so the paper can say who stopped
# the run rather than implying the loop converged or ran out of budget.
TERMINATION_ACTIONS = frozenset({
    "STOP_REQUESTED", "WORKER_CANCELLED", "RUN_TERMINATED",
})


def canonical_json(payload: Any) -> str:
    """Serialize deterministically so digests are stable across processes."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class LedgerEntry:
    """One immutable inter-agent message."""

    id: str
    timestamp: str
    parent_id: str | None
    sender: Mapping[str, str]
    recipient: Mapping[str, str]
    action: str
    payload: Mapping[str, Any]
    prev_hash: str
    entry_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "parent_id": self.parent_id,
            "sender": dict(self.sender),
            "recipient": dict(self.recipient),
            "action": self.action,
            "payload": dict(self.payload),
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
        }


class LedgerError(RuntimeError):
    """The ledger is malformed, or a write violated the append-only contract."""


@dataclass
class CommLedger:
    """Append-only writer and verifier for the inter-agent message ledger."""

    path: Path
    parent_id: str | None = None
    _count: int = field(default=0, init=False)
    _tip: str = field(default=_GENESIS, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _written_signature: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _parse_cache: tuple[tuple[int, int], list[dict[str, Any]]] | None = field(
        default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._replay()
        elif not self.path.is_file():
            self.path.touch()
        self._written_signature = self._file_signature()

    # -- reading ------------------------------------------------------------
    def _file_signature(self) -> tuple[int, int] | None:
        """Size and mtime of the ledger file, or None when it does not exist."""
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        return (stat.st_size, stat.st_mtime_ns)

    def _resync(self) -> None:
        """Re-derive the chain tip if something else appended to the file.

        The in-memory counter is the fast path, but a second ``CommLedger`` on
        the same path — another thread, another ``ResearchSwarm``, another
        process — can advance the chain underneath us. Reusing a stale
        ``_count`` would mint a duplicate ``MSG-NNNN`` id and fork the hash
        chain, so every append re-checks the file signature and rebuilds the tip
        when it moved. One ``stat`` per append is cheap next to the parse it
        usually avoids.
        """
        signature = self._file_signature()
        if signature == self._written_signature:
            return
        self._parse_cache = None
        self._replay()
        self._written_signature = signature

    def _replay(self) -> None:
        """Rebuild the chain tip and counter from an existing ledger file.

        A broken chain is deliberately *not* raised here. Opening a tampered
        ledger is the auditor's normal path, so :meth:`verify` is the single
        adjudicator and must be reachable; refusing to open the file would hide
        the very evidence of tampering.
        """
        count = 0
        tip = _GENESIS
        for entry in self.read_raw():
            tip = str(entry.get("entry_hash", ""))
            count += 1
        self._count = count
        self._tip = tip

    def read_raw(self) -> list[dict[str, Any]]:
        """Return every entry, reusing the last parse when the file is unchanged.

        The ledger is append-only, so a repeated read of an unchanged file must
        be O(1) rather than a full re-parse: the audit log renders the message
        tree on every convergence cycle, and a quadratic parse would dominate
        the run. The cache is keyed on size and mtime, so an external edit or a
        truncation invalidates it.
        """
        if not self.path.exists():
            return []
        with self._lock:
            stat = self.path.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            cached = self._parse_cache
            if cached is not None and cached[0] == signature:
                return cached[1]
            entries: list[dict[str, Any]] = []
            for lineno, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LedgerError(f"Corrupt ledger line {lineno}: {exc}") from exc
                if not isinstance(parsed, dict):
                    raise LedgerError(f"Ledger line {lineno} is not a JSON object")
                entries.append(parsed)
            self._parse_cache = (signature, entries)
            return entries

    @property
    def count(self) -> int:
        return self._count

    @property
    def tip(self) -> str:
        return self._tip

    def next_id(self) -> str:
        return f"MSG-{self._count + 1:04d}"

    # -- writing ------------------------------------------------------------
    def append(self, action: str, sender: Mapping[str, str], recipient: Mapping[str, str],
               payload: Mapping[str, Any] | None = None,
               parent_id: str | None = None) -> LedgerEntry:
        """Append one message and return the persisted entry.

        ``parent_id`` defaults to the ledger's current tip message, which makes
        the conversational thread navigable without inspecting every record.

        Appends are serialised by a re-entrant lock and re-sync the chain tip
        from disk when another writer moved the file, so concurrent workers
        cannot mint a duplicate id or fork the hash chain. The body schema is
        frozen: every field here is covered by the entry digest, so adding or
        renaming one would invalidate ledgers already on disk.
        """
        if action not in ACTIONS:
            raise LedgerError(f"Unknown ledger action {action!r}; expected one of {sorted(ACTIONS)}")
        with self._lock:
            self._resync()
            body = {
                "id": self.next_id(),
                "timestamp": _utc_now(),
                "parent_id": parent_id if parent_id is not None else self.parent_id,
                "sender": dict(sender),
                "recipient": dict(recipient),
                "action": action,
                "payload": dict(payload or {}),
                "prev_hash": self._tip,
            }
            entry_hash = sha256_text(canonical_json(body))
            entry = LedgerEntry(entry_hash=entry_hash, **body)
            # O_APPEND makes each single-line write atomic against concurrent
            # writers; fsync makes it durable, so a crash cannot silently drop a
            # message the caller was told was recorded.
            descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(descriptor, (canonical_json(entry.to_dict()) + "\n").encode("utf-8"))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._count += 1
            self._tip = entry_hash
            self.parent_id = entry.id
            self._written_signature = self._file_signature()
            # The file just changed; the cached parse is stale by definition.
            self._parse_cache = None
            return entry

    def append_many(self, records: Sequence[tuple[str, Mapping[str, str], Mapping[str, str],
                                              Mapping[str, Any] | None]]) -> list[LedgerEntry]:
        """Append several messages under one lock acquisition.

        The chain is inherently sequential, so a batch of related records is
        written in a single critical section. That keeps a burst of concurrent
        worker events from interleaving with each other mid-thread.
        """
        with self._lock:
            return [self.append(action, sender, recipient, payload)
                    for action, sender, recipient, payload in records]

    def note(self, action: str, sender: Mapping[str, str], recipient: Mapping[str, str],
             payload: Mapping[str, Any] | None = None) -> LedgerEntry:
        """Append to the current thread, using the previous message as parent."""
        return self.append(action, sender, recipient, payload)

    def actions_used(self) -> dict[str, int]:
        """Count of each action actually recorded, for telemetry and the paper."""
        counts: dict[str, int] = {}
        for raw in self.read_raw():
            action = str(raw.get("action", ""))
            counts[action] = counts.get(action, 0) + 1
        return counts

    def by_sender(self, agent_id: str) -> list[LedgerEntry]:
        """Every entry a given agent sent, oldest first."""
        return [LedgerEntry(**{**raw, "sender": raw.get("sender", {}),
                               "recipient": raw.get("recipient", {}),
                               "payload": raw.get("payload", {})})
                for raw in self.read_raw()
                if str(raw.get("sender", {}).get("agent_id", "")) == agent_id]

    def by_payload(self, key: str, value: str) -> list[LedgerEntry]:
        """Every entry whose payload carries ``key == value``.

        Coordination records identify their subject in the payload (a task id, a
        message id) rather than in the envelope, because the envelope names the
        *conversation*. Recovery reads the board back through this.
        """
        return [LedgerEntry(**{**raw, "sender": raw.get("sender", {}),
                               "recipient": raw.get("recipient", {}),
                               "payload": raw.get("payload", {})})
                for raw in self.read_raw()
                if str(raw.get("payload", {}).get(key, "")) == value]

    # -- verification -------------------------------------------------------
    def verify(self) -> tuple[bool, str]:
        """Recompute the whole chain. Returns ``(ok, detail)``."""
        tip = _GENESIS
        seen: set[str] = set()
        for index, raw in enumerate(self.read_raw(), start=1):
            expected_id = f"MSG-{index:04d}"
            if raw.get("id") != expected_id:
                return False, f"Entry {index} has id {raw.get('id')!r}, expected {expected_id!r}"
            if raw.get("action") not in ACTIONS:
                return False, f"Entry {raw.get('id')} uses unknown action {raw.get('action')!r}"
            if raw.get("prev_hash") != tip:
                return False, f"Entry {raw.get('id')} breaks the hash chain"
            body = {key: value for key, value in raw.items() if key != "entry_hash"}
            recomputed = sha256_text(canonical_json(body))
            if recomputed != raw.get("entry_hash"):
                return False, f"Entry {raw.get('id')} payload was altered after recording"
            if raw.get("id") in seen:
                return False, f"Duplicate entry id {raw.get('id')}"
            seen.add(str(raw.get("id")))
            tip = str(raw.get("entry_hash"))
        return True, f"{len(seen)} entries verified"

    def by_action(self, action: str) -> list[LedgerEntry]:
        return [LedgerEntry(**{**raw, "sender": raw.get("sender", {}),
                               "recipient": raw.get("recipient", {}),
                               "payload": raw.get("payload", {})})
                for raw in self.read_raw() if raw.get("action") == action]

    def agents(self) -> list[str]:
        """Every distinct agent id that appears as a sender."""
        return sorted({str(raw.get("sender", {}).get("agent_id", ""))
                       for raw in self.read_raw()} - {""})

    def tree(self) -> str:
        """Render the message tree for the paper appendix.

        Roots are entries whose parent is absent from the ledger, so a truncated
        or externally seeded ledger still renders instead of raising.
        """
        raw_entries = self.read_raw()
        by_id = {str(raw.get("id")): raw for raw in raw_entries}
        children: dict[str, list[dict[str, Any]]] = {}
        roots: list[dict[str, Any]] = []
        for raw in raw_entries:
            parent = raw.get("parent_id")
            if parent is None or str(parent) not in by_id:
                roots.append(raw)
            else:
                children.setdefault(str(parent), []).append(raw)
        lines: list[str] = []
        for root in roots:
            self._tree_line(root, children, lines, depth=0, last=True)
        return "\n".join(lines)

    def _tree_line(self, raw: Mapping[str, Any],
                   children: Mapping[str, list[dict[str, Any]]],
                   lines: list[str], depth: int, last: bool) -> None:
        if depth == 0:
            connector = ""
        else:
            connector = "`- " if last else "|- "
        sender = str(raw.get("sender", {}).get("agent_id", "?"))
        lines.append(f"{'  ' * depth}{connector}{raw.get('id')}  {sender}  ->  {raw.get('action')}")
        kids = children.get(str(raw.get("id")), [])
        for index, kid in enumerate(kids):
            self._tree_line(kid, children, lines, depth + 1, last=index == len(kids) - 1)
