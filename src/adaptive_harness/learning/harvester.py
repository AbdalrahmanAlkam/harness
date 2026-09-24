"""Read verified developer traces from the experience database."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any


def harvest_verified_traces(db_path: str | Path) -> list[dict[str, Any]]:
    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Experience database does not exist: {path}")
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='agent_traces'").fetchone()
        if not exists:
            return []
        rows = conn.execute("""SELECT id, user_prompt, trajectory_json, final_solution
            FROM agent_traces WHERE verified_success=1 ORDER BY id""").fetchall()
    examples = []
    for row in rows:
        try:
            steps = json.loads(row["trajectory_json"])
        except (ValueError, TypeError):
            continue
        if not isinstance(steps, list) or not steps or not row["final_solution"].strip():
            continue
        examples.append({"user_prompt": row["user_prompt"], "trajectory_steps": steps,
                         "final_solution": row["final_solution"]})
    return examples
