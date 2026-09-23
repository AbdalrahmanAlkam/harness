"""SQLite storage and repository for execution traces and verified experience."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from adaptive_harness.models.domain import ExecutionTrace


DEFAULT_DB_PATH = Path("output/experience.db")


class ExperienceRepository:
    """Encapsulates SQLite persistence for execution traces and verified agent feedback."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes database schema and indices."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    task_text TEXT NOT NULL,
                    predicted_strategy TEXT NOT NULL,
                    selected_strategy TEXT NOT NULL,
                    verified_strategy TEXT,
                    harness_success BOOLEAN NOT NULL,
                    recovered BOOLEAN NOT NULL,
                    attempts_count INTEGER NOT NULL,
                    execution_time_ms REAL NOT NULL,
                    confidence REAL NOT NULL,
                    entropy REAL NOT NULL,
                    margin REAL NOT NULL,
                    probabilities_json TEXT NOT NULL,
                    trace_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_task_id ON executions(task_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_verified_strategy ON executions(verified_strategy);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_harness_success ON executions(harness_success);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON executions(created_at);")
            conn.execute("""CREATE TABLE IF NOT EXISTS solution_cache (
                task_key TEXT NOT NULL, workspace TEXT NOT NULL, settings_key TEXT NOT NULL,
                answer TEXT NOT NULL, dependencies_json TEXT NOT NULL,
                original_tokens INTEGER NOT NULL DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (task_key, workspace, settings_key))""")

    @staticmethod
    def _task_key(task: str) -> str:
        words = task.split()
        # Command wording is case-insensitive; Linux filenames are not.
        return "v2:" + (" ".join(word.casefold() for word in words[:-1]) + " " + words[-1] if words else "")

    def lookup_solution(self, task: str, workspace: str, settings_key: str) -> dict | None:
        """Return a verified answer only while all recorded file inputs are unchanged."""
        root = Path(workspace).resolve()
        with self._get_connection() as conn:
            row = conn.execute("SELECT answer, dependencies_json, original_tokens FROM solution_cache "
                               "WHERE task_key=? AND workspace=? AND settings_key=?",
                               (self._task_key(task), str(root), settings_key)).fetchone()
        if row is None:
            return None
        try:
            dependencies = json.loads(row["dependencies_json"])
            if not isinstance(dependencies, dict) or not dependencies:
                return None
            for relative_path, expected_hash in dependencies.items():
                path = (root / relative_path).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    return None
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                    return None
            return {"answer": row["answer"], "original_tokens": row["original_tokens"]}
        except (OSError, ValueError, TypeError):
            return None

    def save_solution(self, task: str, workspace: str, settings_key: str, answer: str,
                      dependencies: dict[str, str], original_tokens: int) -> None:
        if not dependencies or not answer.strip():
            return
        with self._get_connection() as conn:
            conn.execute("""INSERT INTO solution_cache
                (task_key, workspace, settings_key, answer, dependencies_json, original_tokens)
                VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(task_key, workspace, settings_key)
                DO UPDATE SET answer=excluded.answer, dependencies_json=excluded.dependencies_json,
                original_tokens=excluded.original_tokens, created_at=CURRENT_TIMESTAMP""",
                (self._task_key(task), str(Path(workspace).resolve()), settings_key, answer,
                 json.dumps(dependencies), max(0, original_tokens)))

    def record_trace(self, trace: ExecutionTrace) -> int:
        """Persists a complete execution trace to SQLite."""
        # Find which strategy, if any, was successfully verified
        verified_strategy = None
        for attempt in trace.attempts:
            if attempt.success and attempt.verification and attempt.verification.success:
                verified_strategy = attempt.strategy
                break

        probs_json = json.dumps(trace.classifier_probabilities)
        full_trace_json = json.dumps(trace.model_dump(mode="json"))

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO executions (
                    task_id, task_text, predicted_strategy, selected_strategy,
                    verified_strategy, harness_success, recovered, attempts_count,
                    execution_time_ms, confidence, entropy, margin,
                    probabilities_json, trace_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.task_id,
                    trace.task_text,
                    trace.classifier_top1,
                    trace.final_strategy,
                    verified_strategy,
                    1 if trace.success else 0,
                    1 if trace.recovered else 0,
                    len(trace.attempts),
                    trace.total_time_ms,
                    trace.confidence,
                    trace.entropy,
                    trace.margin,
                    probs_json,
                    full_trace_json,
                    trace.created_at.isoformat(),
                ),
            )
            return cursor.lastrowid or -1

    def get_recent_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent execution traces as dictionaries."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, task_id, task_text, predicted_strategy, selected_strategy,
                       verified_strategy, harness_success, recovered, attempts_count,
                       execution_time_ms, confidence, entropy, margin, created_at
                FROM executions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_verified_training_examples(self, exclude_fallback: bool = True) -> List[Tuple[str, str]]:
        """Extracts strictly verified historical tasks and verified strategies for retraining."""
        with self._get_connection() as conn:
            query = """
                SELECT task_text, verified_strategy
                FROM executions
                WHERE harness_success = 1
                  AND verified_strategy IS NOT NULL
            """
            if exclude_fallback:
                query += " AND verified_strategy != 'fallback'"

            rows = conn.execute(query).fetchall()
            return [(row["task_text"], row["verified_strategy"]) for row in rows]

    def get_statistics(self) -> Dict[str, Any]:
        """Calculates aggregated performance statistics over stored history."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) as cnt FROM executions").fetchone()["cnt"]
            if total == 0:
                return {
                    "total_executions": 0,
                    "success_rate": 0.0,
                    "recovery_count": 0,
                    "avg_time_ms": 0.0,
                    "avg_attempts": 0.0,
                }

            row = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(harness_success) as successful,
                    SUM(recovered) as recovered_count,
                    AVG(execution_time_ms) as avg_time,
                    AVG(attempts_count) as avg_attempts,
                    AVG(confidence) as avg_confidence,
                    AVG(entropy) as avg_entropy
                FROM executions
                """
            ).fetchone()

            strategy_counts = conn.execute(
                """
                SELECT selected_strategy, COUNT(*) as count
                FROM executions
                GROUP BY selected_strategy
                """
            ).fetchall()

            return {
                "total_executions": row["total"],
                "success_rate": round(row["successful"] / row["total"], 4) if row["total"] else 0.0,
                "recovery_count": row["recovered_count"],
                "avg_time_ms": round(row["avg_time"] or 0.0, 2),
                "avg_attempts": round(row["avg_attempts"] or 0.0, 2),
                "avg_confidence": round(row["avg_confidence"] or 0.0, 4),
                "avg_entropy": round(row["avg_entropy"] or 0.0, 4),
                "strategy_distribution": {r["selected_strategy"]: r["count"] for r in strategy_counts},
            }

    def clear(self) -> None:
        """Wipes the database tables for testing or resetting."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM executions;")
            conn.execute("DELETE FROM solution_cache;")
