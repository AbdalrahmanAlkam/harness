"""Persistent, explicitly selectable developer conversations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sqlite3
from threading import Lock
from uuid import uuid4


@dataclass
class Session:
    id: str
    title: str
    workspace: str
    messages: list[dict]
    model: str | None = None
    skills: list[str] | None = None
    settings: dict[str, str] | None = None


class SessionStore:
    def __init__(self, db_path: Path | str):
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(db_path), check_same_thread=False)
        self.lock = Lock()
        self.connection.execute("""CREATE TABLE IF NOT EXISTS agent_sessions (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, workspace TEXT NOT NULL,
            messages TEXT NOT NULL, model TEXT, skills TEXT NOT NULL,
            settings TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(agent_sessions)")}
        if "settings" not in columns:
            self.connection.execute("ALTER TABLE agent_sessions ADD COLUMN settings TEXT NOT NULL DEFAULT '{}'")
        self.connection.commit()

    def create(self, workspace: str, title: str = "New session") -> Session:
        session = Session(uuid4().hex[:12], title, workspace, [], None, [], {})
        self.save(session)
        return session

    def save(self, session: Session) -> None:
        with self.lock:
            self.connection.execute("""INSERT INTO agent_sessions(id,title,workspace,messages,model,skills,settings,updated_at)
                VALUES (?,?,?,?,?,?,?,CURRENT_TIMESTAMP) ON CONFLICT(id) DO UPDATE SET
                title=excluded.title, workspace=excluded.workspace, messages=excluded.messages,
                model=excluded.model, skills=excluded.skills, settings=excluded.settings,
                updated_at=CURRENT_TIMESTAMP""",
                (session.id, session.title, session.workspace, json.dumps(session.messages),
                 session.model, json.dumps(session.skills or []), json.dumps(session.settings or {})))
            self.connection.commit()

    def load(self, session_id: str) -> Session | None:
        with self.lock:
            row = self.connection.execute("SELECT id,title,workspace,messages,model,skills,settings FROM agent_sessions WHERE id=?", (session_id,)).fetchone()
        return Session(row[0], row[1], row[2], json.loads(row[3]), row[4], json.loads(row[5]), json.loads(row[6])) if row else None

    def list(self, limit: int = 20) -> list[Session]:
        with self.lock:
            rows = self.connection.execute("SELECT id,title,workspace,messages,model,skills,settings FROM agent_sessions ORDER BY updated_at DESC, rowid DESC LIMIT ?", (limit,)).fetchall()
        return [Session(row[0], row[1], row[2], json.loads(row[3]), row[4], json.loads(row[5]), json.loads(row[6])) for row in rows]

    def close(self) -> None:
        with self.lock:
            self.connection.close()
