"""DB レイヤの単体テスト (LLM 不要、SQLite in-memory ではなくテンポラリファイルで動かす)。"""
import json
import os
import tempfile
import unittest

import numpy as np

from diff_memory.db import Memory, MemoryDB, now_iso


def _emb(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestMemoryDB(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = MemoryDB(self.tmp.name)

    def tearDown(self):
        self.db.close()
        os.unlink(self.tmp.name)

    def test_insert_and_get_roundtrip(self):
        emb = _emb(1)
        m = Memory.new("テスト記憶", "fact", emb, 0.6, 0.5, 0.7)
        self.db.insert(m, source_text="テスト発話", novelty=0.9, reason="new")

        got = self.db.get(m.id)
        self.assertEqual(got.text, "テスト記憶")
        self.assertEqual(got.type, "fact")
        self.assertEqual(got.source_count, 1)
        self.assertAlmostEqual(got.importance, 0.6)
        # embedding がバイナリ往復しても一致
        np.testing.assert_allclose(got.embedding, emb, atol=1e-6)
        # source_log に1件残る
        hist = self.db.history(m.id)
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["action"], "create")

    def test_reinforce_increments_count_and_confidence(self):
        m = Memory.new("再言及テスト", "preference", _emb(2), 0.5, 0.5, 0.6)
        self.db.insert(m, source_text="一回目", novelty=0.8, reason="")

        # 同じことの再言及2回
        for _ in range(2):
            self.db.reinforce(m.id, source_text="再言及", novelty=0.05, reason="dup")

        got = self.db.get(m.id)
        self.assertEqual(got.source_count, 3)
        self.assertGreater(got.confidence, 0.6)  # 0.05 ずつ加算 (cap 1.0)
        self.assertLessEqual(got.confidence, 1.0)
        self.assertEqual(len(self.db.history(m.id)), 3)

    def test_update_text_replaces_content_and_appends_contradicts(self):
        m_old = Memory.new("Node使ってる", "preference", _emb(3), 0.5, 0.4, 0.6)
        self.db.insert(m_old, source_text="昔", novelty=0.9, reason="")

        # コンフリクトする別記憶を作って id を contradicts に積む想定
        m_other = Memory.new("過去のメモ", "preference", _emb(4), 0.5, 0.4, 0.6)
        self.db.insert(m_other, source_text="他", novelty=0.9, reason="")

        new_emb = _emb(5)
        self.db.update_text(
            m_old.id, "Deno使ってる", new_emb,
            importance=0.7, stability=0.5, confidence=0.9,
            contradicts_add=[m_other.id],
            source_text="今はDeno", novelty=0.3, reason="swap",
        )

        got = self.db.get(m_old.id)
        self.assertEqual(got.text, "Deno使ってる")
        self.assertEqual(got.source_count, 2)
        self.assertIn(m_other.id, got.contradicts)
        np.testing.assert_allclose(got.embedding, new_emb, atol=1e-6)

    def test_lower_confidence_caps_at_zero(self):
        m = Memory.new("揺らぐ記憶", "fact", _emb(6), 0.5, 0.5, 0.4)
        self.db.insert(m, source_text="", novelty=0.9, reason="")

        for _ in range(20):
            self.db.lower_confidence(m.id, factor=0.5)

        got = self.db.get(m.id)
        self.assertGreaterEqual(got.confidence, 0.0)
        self.assertLess(got.confidence, 1e-3)

    def test_delete_removes_memory_and_history(self):
        m = Memory.new("消す対象", "fact", _emb(7), 0.5, 0.5, 0.5)
        self.db.insert(m, source_text="", novelty=0.9, reason="")
        self.db.reinforce(m.id, source_text="", novelty=0.0, reason="")

        self.db.delete(m.id)

        self.assertIsNone(self.db.get(m.id))
        self.assertEqual(self.db.history(m.id), [])

    def test_invalid_type_rejected_by_check_constraint(self):
        with self.assertRaises(Exception):
            m = Memory.new("不正type", "garbage", _emb(8), 0.5, 0.5, 0.5)
            self.db.insert(m, source_text="", novelty=0.9, reason="")

    def test_now_iso_is_parseable(self):
        from datetime import datetime
        s = now_iso()
        dt = datetime.fromisoformat(s)
        self.assertIsNotNone(dt.tzinfo)


if __name__ == "__main__":
    unittest.main()
