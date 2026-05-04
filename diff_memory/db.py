import json
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
  id           TEXT PRIMARY KEY,
  type         TEXT NOT NULL CHECK(type IN ('preference','fact','project','constraint','episodic')),
  text         TEXT NOT NULL,
  embedding    BLOB NOT NULL,
  embed_dim    INTEGER NOT NULL,
  importance   REAL NOT NULL DEFAULT 0.5,
  stability    REAL NOT NULL DEFAULT 0.5,
  confidence   REAL NOT NULL DEFAULT 0.5,
  created_at   TEXT NOT NULL,
  last_seen    TEXT NOT NULL,
  source_count INTEGER NOT NULL DEFAULT 1,
  contradicts  TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS source_log (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  memory_id  TEXT NOT NULL,
  text       TEXT NOT NULL,
  timestamp  TEXT NOT NULL,
  action     TEXT NOT NULL CHECK(action IN ('create','update','reinforce')),
  novelty    REAL,
  reason     TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_last_seen ON memories(last_seen);
CREATE INDEX IF NOT EXISTS idx_source_log_memory ON source_log(memory_id);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Memory:
    id: str
    type: str
    text: str
    embedding: np.ndarray
    importance: float
    stability: float
    confidence: float
    created_at: str
    last_seen: str
    source_count: int = 1
    contradicts: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("embedding", None)
        return d

    @staticmethod
    def new(text: str, type_: str, embedding: np.ndarray, importance: float,
            stability: float, confidence: float, ts: Optional[str] = None) -> "Memory":
        ts = ts or now_iso()
        return Memory(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            type=type_,
            text=text,
            embedding=embedding.astype(np.float32),
            importance=float(importance),
            stability=float(stability),
            confidence=float(confidence),
            created_at=ts,
            last_seen=ts,
        )


class MemoryDB:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ---------- write ----------

    def insert(self, m: Memory, source_text: str, novelty: float, reason: str = "") -> None:
        emb = np.asarray(m.embedding, dtype=np.float32)
        self.conn.execute(
            """INSERT INTO memories
               (id, type, text, embedding, embed_dim, importance, stability, confidence,
                created_at, last_seen, source_count, contradicts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (m.id, m.type, m.text, emb.tobytes(), int(emb.shape[0]),
             m.importance, m.stability, m.confidence,
             m.created_at, m.last_seen, m.source_count, json.dumps(m.contradicts)),
        )
        self._log(m.id, source_text, "create", novelty, reason)
        self.conn.commit()

    def update_text(self, mem_id: str, text: str, embedding: np.ndarray,
                    importance: float, stability: float, confidence: float,
                    contradicts_add: list, source_text: str, novelty: float, reason: str = "") -> None:
        cur = self.get(mem_id)
        if cur is None:
            raise KeyError(mem_id)
        new_contradicts = list(set(cur.contradicts + (contradicts_add or [])))
        emb = np.asarray(embedding, dtype=np.float32)
        self.conn.execute(
            """UPDATE memories
               SET text=?, embedding=?, embed_dim=?, importance=?, stability=?, confidence=?,
                   last_seen=?, source_count=source_count+1, contradicts=?
               WHERE id=?""",
            (text, emb.tobytes(), int(emb.shape[0]),
             float(importance), float(stability), float(confidence),
             now_iso(), json.dumps(new_contradicts), mem_id),
        )
        self._log(mem_id, source_text, "update", novelty, reason)
        self.conn.commit()

    def reinforce(self, mem_id: str, source_text: str, novelty: float,
                  confidence_bump: float = 0.05, reason: str = "") -> None:
        cur = self.get(mem_id)
        if cur is None:
            raise KeyError(mem_id)
        new_conf = min(1.0, cur.confidence + confidence_bump)
        self.conn.execute(
            """UPDATE memories
               SET last_seen=?, source_count=source_count+1, confidence=?
               WHERE id=?""",
            (now_iso(), new_conf, mem_id),
        )
        self._log(mem_id, source_text, "reinforce", novelty, reason)
        self.conn.commit()

    def lower_confidence(self, mem_id: str, factor: float = 0.6) -> None:
        cur = self.get(mem_id)
        if cur is None:
            return
        new_conf = max(0.0, cur.confidence * factor)
        self.conn.execute("UPDATE memories SET confidence=? WHERE id=?", (new_conf, mem_id))
        self.conn.commit()

    def set_importance(self, mem_id: str, importance: float) -> None:
        self.conn.execute("UPDATE memories SET importance=? WHERE id=?",
                          (max(0.0, min(1.0, importance)), mem_id))
        self.conn.commit()

    def delete(self, mem_id: str) -> None:
        self.conn.execute("DELETE FROM memories WHERE id=?", (mem_id,))
        self.conn.execute("DELETE FROM source_log WHERE memory_id=?", (mem_id,))
        self.conn.commit()

    def _log(self, mem_id: str, text: str, action: str, novelty: float, reason: str) -> None:
        self.conn.execute(
            """INSERT INTO source_log (memory_id, text, timestamp, action, novelty, reason)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (mem_id, text, now_iso(), action, float(novelty), reason),
        )

    # ---------- read ----------

    def get(self, mem_id: str) -> Optional[Memory]:
        row = self.conn.execute("SELECT * FROM memories WHERE id=?", (mem_id,)).fetchone()
        return self._row_to_memory(row) if row else None

    def all(self) -> list[Memory]:
        rows = self.conn.execute("SELECT * FROM memories").fetchall()
        return [self._row_to_memory(r) for r in rows]

    def history(self, mem_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM source_log WHERE memory_id=? ORDER BY id ASC", (mem_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        emb = np.frombuffer(row["embedding"], dtype=np.float32)
        return Memory(
            id=row["id"],
            type=row["type"],
            text=row["text"],
            embedding=emb,
            importance=row["importance"],
            stability=row["stability"],
            confidence=row["confidence"],
            created_at=row["created_at"],
            last_seen=row["last_seen"],
            source_count=row["source_count"],
            contradicts=json.loads(row["contradicts"]) if row["contradicts"] else [],
        )
