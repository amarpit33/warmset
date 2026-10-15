"""Evidence ledger: every decision (with all candidates and estimates), action, verdict and outcome.

SQLite, stdlib only. Prompts and outputs are stored once, content-addressed, so the ledger can
explain a run without bloating. Labels (ground truth) can be attached later for calibration.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .types import Usage, Verdict

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, created_at REAL NOT NULL, ended_at REAL, name TEXT, policy TEXT NOT NULL,
    status TEXT NOT NULL, stop_reason TEXT, budget_json TEXT, inputs_json TEXT, result_hash TEXT,
    seconds REAL DEFAULT 0, calls INTEGER DEFAULT 0, tokens INTEGER DEFAULT 0, loads INTEGER DEFAULT 0,
    load_seconds REAL DEFAULT 0, wall_seconds REAL, unverified_steps INTEGER DEFAULT 0, meta_json TEXT
);
CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(id), seq INTEGER NOT NULL,
    name TEXT NOT NULL, input_hash TEXT, status TEXT, accepted_key TEXT, output_hash TEXT,
    started_at REAL NOT NULL, ended_at REAL
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, step_id INTEGER NOT NULL REFERENCES steps(id),
    attempt INTEGER NOT NULL, chosen_key TEXT, temperature REAL, reason TEXT NOT NULL, view_json TEXT NOT NULL,
    ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, step_id INTEGER NOT NULL,
    decision_id INTEGER NOT NULL REFERENCES decisions(id), step_name TEXT NOT NULL, action_key TEXT NOT NULL,
    kind TEXT NOT NULL, model TEXT, temperature REAL, ok INTEGER NOT NULL, passed INTEGER NOT NULL,
    error TEXT, prompt_tokens INTEGER, completion_tokens INTEGER, seconds REAL, load_seconds REAL,
    exec_seconds REAL, calls INTEGER, predicted_p REAL, predicted_seconds REAL, output_hash TEXT,
    context_json TEXT, bucket TEXT, ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS verdicts (
    action_id INTEGER NOT NULL REFERENCES actions(id), verifier TEXT NOT NULL, outcome TEXT NOT NULL,
    evidence TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, ts REAL NOT NULL, type TEXT NOT NULL,
    data_json TEXT
);
CREATE TABLE IF NOT EXISTS blobs (hash TEXT PRIMARY KEY, content TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS labels (
    run_id TEXT PRIMARY KEY REFERENCES runs(id), correct INTEGER NOT NULL, source TEXT, detail TEXT
);
CREATE INDEX IF NOT EXISTS ix_actions_step ON actions(step_name, action_key);
CREATE INDEX IF NOT EXISTS ix_steps_run ON steps(run_id);
CREATE INDEX IF NOT EXISTS ix_decisions_step ON decisions(step_id);
"""


def to_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, default=lambda o: o.model_dump(mode="json") if hasattr(o, "model_dump") else repr(o),
                      sort_keys=True)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]


class Ledger:
    """Thread-safe wrapper around one SQLite connection."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._conn:
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(f"ledger {path} has schema v{version}; this warmset supports v{SCHEMA_VERSION}")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        self._conn.close()

    def _exec(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self._lock, self._conn:
            return self._conn.execute(sql, params)

    def _all(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # ------------------------------------------------------------ writes

    def put_blob(self, content: Any) -> str:
        text = content if isinstance(content, str) else to_json(content)
        h = content_hash(text)
        self._exec("INSERT OR IGNORE INTO blobs VALUES (?, ?)", (h, text))
        return h

    def blob(self, h: str | None) -> str | None:
        if not h:
            return None
        rows = self._all("SELECT content FROM blobs WHERE hash=?", (h,))
        return rows[0]["content"] if rows else None

    def create_run(self, *, policy: str, name: str | None, budget: Any, inputs: Any, meta: Any = None) -> str:
        run_id = uuid.uuid4().hex[:12]
        self._exec("INSERT INTO runs (id, created_at, name, policy, status, budget_json, inputs_json, meta_json) "
                   "VALUES (?, ?, ?, ?, 'running', ?, ?, ?)",
                   (run_id, time.time(), name, policy, to_json(budget), to_json(inputs), to_json(meta)))
        return run_id

    def finish_run(self, run_id: str, *, status: str, stop_reason: str | None, result: Any, seconds: float,
                   calls: int, tokens: int, loads: int, load_seconds: float, wall_seconds: float,
                   unverified_steps: int) -> None:
        rh = self.put_blob(result) if result is not None else None
        self._exec("UPDATE runs SET ended_at=?, status=?, stop_reason=?, result_hash=?, seconds=?, calls=?, tokens=?, "
                   "loads=?, load_seconds=?, wall_seconds=?, unverified_steps=? WHERE id=?",
                   (time.time(), status, stop_reason, rh, seconds, calls, tokens, loads, load_seconds, wall_seconds,
                    unverified_steps, run_id))

    def start_step(self, run_id: str, seq: int, name: str, inputs: Any) -> int:
        cur = self._exec("INSERT INTO steps (run_id, seq, name, input_hash, started_at) VALUES (?, ?, ?, ?, ?)",
                         (run_id, seq, name, self.put_blob(inputs), time.time()))
        return int(cur.lastrowid)

    def finish_step(self, step_id: int, status: str, accepted_key: str | None, output: Any) -> None:
        oh = self.put_blob(output) if output is not None else None
        self._exec("UPDATE steps SET status=?, accepted_key=?, output_hash=?, ended_at=? WHERE id=?",
                   (status, accepted_key, oh, time.time(), step_id))

    def record_decision(self, run_id: str, step_id: int, attempt: int, chosen_key: str | None,
                        temperature: float | None, reason: str, view: dict[str, Any]) -> int:
        cur = self._exec("INSERT INTO decisions (run_id, step_id, attempt, chosen_key, temperature, reason, "
                         "view_json, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (run_id, step_id, attempt, chosen_key, temperature, reason, json.dumps(view), time.time()))
        return int(cur.lastrowid)

    def record_action(self, *, run_id: str, step_id: int, decision_id: int, step_name: str, key: str, kind: str,
                      model: str | None, temperature: float, ok: bool, passed: bool, error: str | None,
                      usage: Usage, predicted_p: float, predicted_seconds: float, output: Any,
                      context: dict[str, Any] | None, verdicts: list[Verdict], bucket: str | None = None) -> int:
        oh = self.put_blob(output) if output is not None else None
        cur = self._exec(
            "INSERT INTO actions (run_id, step_id, decision_id, step_name, action_key, kind, model, temperature, ok, "
            "passed, error, prompt_tokens, completion_tokens, seconds, load_seconds, exec_seconds, calls, predicted_p, "
            "predicted_seconds, output_hash, context_json, bucket, ts) VALUES (" + ",".join("?" * 23) + ")",
            (run_id, step_id, decision_id, step_name, key, kind, model, temperature, int(ok), int(passed), error,
             usage.prompt_tokens, usage.completion_tokens, usage.seconds, usage.load_seconds, usage.exec_seconds,
             usage.calls, predicted_p, predicted_seconds, oh, json.dumps(context) if context else None, bucket,
             time.time()))
        action_id = int(cur.lastrowid)
        with self._lock, self._conn:
            self._conn.executemany("INSERT INTO verdicts VALUES (?, ?, ?, ?)",
                                   [(action_id, v.verifier, v.outcome, v.evidence) for v in verdicts])
        return action_id

    def record_event(self, run_id: str, type_: str, data: dict[str, Any]) -> None:
        self._exec("INSERT INTO events (run_id, ts, type, data_json) VALUES (?, ?, ?, ?)",
                   (run_id, time.time(), type_, to_json(data)))

    def label(self, run_id: str, correct: bool, source: str = "checker", detail: str = "") -> None:
        self._exec("INSERT OR REPLACE INTO labels VALUES (?, ?, ?, ?)", (run_id, int(correct), source, detail))

    # ------------------------------------------------------------ reads

    def runs(self, limit: int = 50, name: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT r.*, l.correct FROM runs r LEFT JOIN labels l ON l.run_id = r.id"
        params: tuple[Any, ...] = ()
        if name:
            q += " WHERE r.name = ?"
            params = (name,)
        return [dict(r) for r in self._all(q + " ORDER BY r.created_at DESC LIMIT ?", (*params, limit))]

    def run(self, run_id: str) -> dict[str, Any] | None:
        rows = self._all("SELECT r.*, l.correct FROM runs r LEFT JOIN labels l ON l.run_id = r.id "
                         "WHERE r.id = ? OR r.id LIKE ?", (run_id, run_id + "%"))
        return dict(rows[0]) if rows else None

    def steps(self, run_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self._all("SELECT * FROM steps WHERE run_id=? ORDER BY seq", (run_id,))]

    def decisions(self, run_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self._all(
            "SELECT d.*, s.name AS step_name, s.seq AS step_seq FROM decisions d JOIN steps s ON s.id = d.step_id "
            "WHERE d.run_id=? ORDER BY d.id", (run_id,))]

    def actions(self, run_id: str) -> list[dict[str, Any]]:
        rows = [dict(r) for r in self._all("SELECT * FROM actions WHERE run_id=? ORDER BY id", (run_id,))]
        for a in rows:
            a["verdicts"] = [dict(v) for v in self._all("SELECT verifier, outcome, evidence FROM verdicts "
                                                        "WHERE action_id=?", (a["id"],))]
        return rows

    def events(self, run_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self._all("SELECT * FROM events WHERE run_id=? ORDER BY id", (run_id,))]

    def outcome_rows(self, exclude_names: tuple[str, ...] = ()) -> list[dict[str, Any]]:
        """One row per executed action, for fitting estimators."""
        q = ("SELECT a.step_name, a.action_key, a.passed, a.exec_seconds, a.predicted_p, a.ok, a.error, a.bucket, "
             "r.name AS run_name "
             "FROM actions a JOIN runs r ON r.id = a.run_id")
        return [dict(r) for r in self._all(q) if r["run_name"] not in exclude_names]
