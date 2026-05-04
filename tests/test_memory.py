"""MemoryStore の振る舞いを LLM/embedder をモックして単体検証する。

Embedder は決定的な簡易ベクトルを返すスタブ、Judge は事前指定したアクションを返すスタブ。
"""
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from diff_memory.config import Config
from diff_memory.db import MemoryDB
from diff_memory.llm import Judgment
from diff_memory.memory import MemoryStore


class StubEmbedder:
    """テキスト→決定的ベクトルの簡易実装。同じテキストは同じベクトル、別テキストは異なる。"""
    dim = 8

    def _v(self, text: str) -> np.ndarray:
        # 文字コードから決定的に生成
        v = np.zeros(self.dim, dtype=np.float32)
        for i, c in enumerate(text):
            v[i % self.dim] += (ord(c) % 31) / 31.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_passage(self, text: str) -> np.ndarray:
        return self._v(text)

    def embed_query(self, text: str) -> np.ndarray:
        return self._v(text)


class ScriptedJudge:
    """add() 順に判定結果を返すキュー型 Judge。"""
    def __init__(self, scripted: list[Judgment]):
        self.queue = list(scripted)
        self.calls_new = 0
        self.calls_with = 0

    def judge_new(self, text: str) -> Judgment:
        self.calls_new += 1
        return self.queue.pop(0)

    def judge_with_candidates(self, text: str, candidates) -> Judgment:
        self.calls_with += 1
        return self.queue.pop(0)


def _make_store(scripted: list[Judgment]) -> tuple[MemoryStore, str]:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    cfg = Config()
    cfg.db_path = tmp.name
    db = MemoryDB(tmp.name)
    store = MemoryStore(config=cfg, db=db, embedder=StubEmbedder(), judge=ScriptedJudge(scripted))
    return store, tmp.name


class TestNoveltyAndCreate(unittest.TestCase):
    def test_first_add_is_always_create_with_novelty_one(self):
        j = Judgment(action="create", type="fact", text="t", importance=0.5, stability=0.5, confidence=0.5)
        store, path = _make_store([j])
        try:
            res = store.add("Pythonでコード書いてる")
            self.assertEqual(res.action, "create")
            self.assertAlmostEqual(res.novelty, 1.0)
            self.assertEqual(len(store.db.all()), 1)
        finally:
            store.db.close()
            os.unlink(path)

    def test_high_novelty_routes_to_judge_new_not_with_candidates(self):
        # 1回目 (新規) と 2回目 (空似でない別話題) は judge_new を呼ぶ想定
        j1 = Judgment(action="create", type="fact", text="t1", importance=0.5, stability=0.5, confidence=0.5)
        j2 = Judgment(action="create", type="fact", text="t2", importance=0.5, stability=0.5, confidence=0.5)
        store, path = _make_store([j1, j2])
        try:
            store.add("xxxx")
            store.add("totally different content yyyy")
            self.assertEqual(store.judge.calls_new, 2)
            self.assertEqual(store.judge.calls_with, 0)
        finally:
            store.db.close()
            os.unlink(path)


class TestReinforce(unittest.TestCase):
    def test_reinforce_increments_source_count_no_duplicate_record(self):
        first = Judgment(action="create", type="preference", text="登山好き",
                         importance=0.5, stability=0.5, confidence=0.5)
        again = Judgment(action="reinforce", target_id=None, type="preference",
                         text="登山好き", importance=0.5, stability=0.5, confidence=0.6)
        store, path = _make_store([first, again])
        try:
            r1 = store.add("登山が好き")
            r2 = store.add("登山が好き")  # 同じテキスト → 同じ embedding → novelty=0
            self.assertEqual(r1.action, "create")
            self.assertEqual(r2.action, "reinforce")
            mems = store.db.all()
            self.assertEqual(len(mems), 1)
            self.assertEqual(mems[0].source_count, 2)
        finally:
            store.db.close()
            os.unlink(path)


class TestUpdate(unittest.TestCase):
    def test_update_overwrites_text_and_records_contradicts(self):
        first = Judgment(action="create", type="preference", text="Node使ってる",
                         importance=0.5, stability=0.5, confidence=0.6)
        # update では target_id を指定する想定だが None でも候補先頭に振られる
        upd = Judgment(action="update", target_id=None, type="preference",
                       text="Deno使ってる", importance=0.7, stability=0.5, confidence=0.9,
                       contradicts_ids=[])
        store, path = _make_store([first, upd])
        try:
            r1 = store.add("Node使ってる")
            r2 = store.add("Node使ってる")  # 同じテキスト → 候補1件 → judge_with_candidates
            self.assertEqual(r1.action, "create")
            self.assertEqual(r2.action, "update")
            mems = store.db.all()
            self.assertEqual(len(mems), 1)
            self.assertEqual(mems[0].text, "Deno使ってる")
        finally:
            store.db.close()
            os.unlink(path)


class TestIgnore(unittest.TestCase):
    def test_ignore_does_not_persist(self):
        j = Judgment(action="ignore", type="fact", text="", importance=0.0, stability=0.0, confidence=0.0)
        store, path = _make_store([j])
        try:
            res = store.add("おはよう")
            self.assertEqual(res.action, "ignore")
            self.assertIsNone(res.memory_id)
            self.assertEqual(store.db.all(), [])
        finally:
            store.db.close()
            os.unlink(path)

    def test_empty_text_is_ignored_without_calling_llm(self):
        store, path = _make_store([])  # スクリプト空 → LLM 呼ばれない事を保証
        try:
            res = store.add("   ")
            self.assertEqual(res.action, "ignore")
            self.assertEqual(store.judge.calls_new, 0)
            self.assertEqual(store.judge.calls_with, 0)
        finally:
            store.db.close()
            os.unlink(path)


class TestQueryScoring(unittest.TestCase):
    def test_query_returns_top_k_in_descending_score(self):
        # 3件 create
        scripted = [
            Judgment(action="create", type="fact", text="A", importance=0.9, stability=0.9, confidence=0.9),
            Judgment(action="create", type="fact", text="B", importance=0.5, stability=0.5, confidence=0.5),
            Judgment(action="create", type="fact", text="C", importance=0.1, stability=0.1, confidence=0.1),
        ]
        store, path = _make_store(scripted)
        try:
            store.add("aaa apple application")
            store.add("bbb banana boxing")
            store.add("ccc cucumber clinic")
            results = store.query("aaa apple application", k=3)
            self.assertEqual(len(results), 3)
            # スコア降順
            for i in range(len(results) - 1):
                self.assertGreaterEqual(results[i].score, results[i+1].score)
        finally:
            store.db.close()
            os.unlink(path)


class TestDecay(unittest.TestCase):
    def test_decay_reduces_importance_over_time(self):
        j = Judgment(action="create", type="episodic", text="x",
                     importance=1.0, stability=0.0, confidence=0.5)
        store, path = _make_store([j])
        try:
            store.add("xxxx")
            mem_id = store.db.all()[0].id
            # last_seen を強制的に過去にする
            past = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
            store.db.conn.execute("UPDATE memories SET last_seen=? WHERE id=?", (past, mem_id))
            store.db.conn.commit()

            store.decay()

            got = store.db.get(mem_id)
            # episodic は tau=14日、stability=0 なので tau_eff=14, 60日経過 → 大幅減衰
            self.assertLess(got.importance, 0.05)
        finally:
            store.db.close()
            os.unlink(path)

    def test_decay_preserves_high_stability_preference(self):
        j = Judgment(action="create", type="preference", text="x",
                     importance=1.0, stability=1.0, confidence=0.5)
        store, path = _make_store([j])
        try:
            store.add("xxxx")
            mem_id = store.db.all()[0].id
            past = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
            store.db.conn.execute("UPDATE memories SET last_seen=? WHERE id=?", (past, mem_id))
            store.db.conn.commit()

            store.decay()

            got = store.db.get(mem_id)
            # preference は tau=365、stability=1 なので tau_eff=1095、60日では殆ど減らない
            self.assertGreater(got.importance, 0.9)
        finally:
            store.db.close()
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
