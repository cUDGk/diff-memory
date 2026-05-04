"""Differential memory store: novelty-gated insert, weighted retrieval, time decay."""
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .config import Config
from .db import Memory, MemoryDB, now_iso
from .embed import Embedder, cosine_matrix, make_embedder
from .llm import OllamaJudge, Judgment


@dataclass
class ScoredMemory:
    memory: Memory
    score: float
    similarity: float
    recency: float


@dataclass
class AddResult:
    action: str            # create | update | reinforce | ignore
    memory_id: Optional[str]
    novelty: float
    judgment: Judgment


class MemoryStore:
    def __init__(self, config: Optional[Config] = None,
                 db: Optional[MemoryDB] = None,
                 embedder: Optional[Embedder] = None,
                 judge: Optional[OllamaJudge] = None):
        self.cfg = config or Config()
        self.db = db or MemoryDB(self.cfg.db_path)
        self.embedder = embedder or make_embedder(
            self.cfg.embed_backend, self.cfg.embed_model, self.cfg.ollama_url
        )
        self.judge = judge or OllamaJudge(
            self.cfg.ollama_url, self.cfg.llm_model,
            self.cfg.llm_timeout_sec, self.cfg.llm_keep_alive,
        )

    # ---------- write ----------

    def add(self, text: str, ts: Optional[str] = None) -> AddResult:
        text = text.strip()
        if not text:
            return AddResult("ignore", None, 0.0, Judgment(action="ignore", reason="empty"))

        emb = self.embedder.embed_passage(text)
        all_mems = self.db.all()
        candidates = self._top_similar(emb, all_mems, self.cfg.top_n_candidates)
        max_sim = candidates[0]["sim"] if candidates else 0.0
        novelty = float(1.0 - max_sim)

        if novelty > self.cfg.novelty_threshold or not candidates:
            j = self.judge.judge_new(text)
            return self._apply_new(text, emb, ts, novelty, j)
        else:
            j = self.judge.judge_with_candidates(text, candidates)
            return self._apply_with_candidates(text, emb, ts, novelty, j, candidates)

    def _apply_new(self, text: str, emb: np.ndarray, ts: Optional[str],
                   novelty: float, j: Judgment) -> AddResult:
        if j.action == "ignore":
            return AddResult("ignore", None, novelty, j)
        body = j.text or text
        m = Memory.new(body, j.type, emb, j.importance, j.stability, j.confidence, ts)
        self.db.insert(m, source_text=text, novelty=novelty, reason=j.reason)
        return AddResult("create", m.id, novelty, j)

    def _apply_with_candidates(self, text: str, emb: np.ndarray, ts: Optional[str],
                               novelty: float, j: Judgment, candidates: list[dict]) -> AddResult:
        if j.action == "ignore":
            return AddResult("ignore", None, novelty, j)

        if j.action == "reinforce":
            target = j.target_id or candidates[0]["id"]
            self.db.reinforce(target, source_text=text, novelty=novelty, reason=j.reason)
            return AddResult("reinforce", target, novelty, j)

        if j.action == "update":
            target = j.target_id or candidates[0]["id"]
            for cid in (j.contradicts_ids or []):
                if cid != target:
                    self.db.lower_confidence(cid)
            self.db.update_text(
                target, j.text or text, emb,
                j.importance, j.stability, j.confidence,
                contradicts_add=j.contradicts_ids or [],
                source_text=text, novelty=novelty, reason=j.reason,
            )
            return AddResult("update", target, novelty, j)

        # action == "create"
        return self._apply_new(text, emb, ts, novelty, j)

    # ---------- read ----------

    def query(self, q: str, k: int = 5,
              type_filter: Optional[str] = None,
              min_confidence: float = 0.0) -> list[ScoredMemory]:
        emb = self.embedder.embed_query(q)
        mems = self.db.all()
        if type_filter:
            mems = [m for m in mems if m.type == type_filter]
        if min_confidence > 0:
            mems = [m for m in mems if m.confidence >= min_confidence]
        if not mems:
            return []

        mat = np.stack([m.embedding for m in mems])
        sims = cosine_matrix(emb, mat)
        now = datetime.now(timezone.utc)

        scored: list[ScoredMemory] = []
        for m, sim in zip(mems, sims):
            rec = self._recency(m, now)
            score = (
                self.cfg.w_similarity  * float(sim)
                + self.cfg.w_importance * m.importance
                + self.cfg.w_stability  * m.stability
                + self.cfg.w_recency    * rec
                + self.cfg.w_confidence * m.confidence
            )
            scored.append(ScoredMemory(m, float(score), float(sim), float(rec)))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    # ---------- maintenance ----------

    def decay(self) -> int:
        """type 別 tau で importance を時間減衰。更新件数を返す。"""
        now = datetime.now(timezone.utc)
        n = 0
        for m in self.db.all():
            tau_days = self.cfg.tau_by_type.get(m.type, 90.0)
            # stability で tau を伸ばす（max 3倍）
            tau_eff = tau_days * (1.0 + 2.0 * m.stability)
            dt_days = max(0.0, (now - _parse_iso(m.last_seen)).total_seconds() / 86400.0)
            factor = math.exp(-dt_days / tau_eff)
            new_imp = m.importance * factor
            if abs(new_imp - m.importance) > 1e-4:
                self.db.set_importance(m.id, new_imp)
                n += 1
        return n

    # ---------- prompt injection ----------

    def render_for_prompt(self, q: str, k: int = 5) -> str:
        scored = self.query(q, k=k)
        if not scored:
            return ""
        lines = ["# 関連する記憶"]
        for s in scored:
            m = s.memory
            lines.append(f"- [{m.type}] {m.text}  "
                         f"(score={s.score:.2f} sim={s.similarity:.2f} conf={m.confidence:.2f})")
        return "\n".join(lines)

    # ---------- internals ----------

    def _top_similar(self, emb: np.ndarray, mems: list[Memory], n: int) -> list[dict]:
        if not mems:
            return []
        mat = np.stack([m.embedding for m in mems])
        sims = cosine_matrix(emb, mat)
        idx = np.argsort(-sims)[:n]
        out = []
        for i in idx:
            m = mems[int(i)]
            out.append({
                "id": m.id, "type": m.type, "text": m.text,
                "sim": float(sims[int(i)]), "confidence": m.confidence,
            })
        return out

    @staticmethod
    def _recency(m: Memory, now: datetime) -> float:
        """last_seen からの経過を [0,1] に。30日で 1/e。"""
        dt_days = max(0.0, (now - _parse_iso(m.last_seen)).total_seconds() / 86400.0)
        return math.exp(-dt_days / 30.0)


def _parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
