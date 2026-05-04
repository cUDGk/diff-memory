"""Embedding ユーティリティの単体テスト (LLM 不要)。"""
import unittest

import numpy as np

from diff_memory.embed import cosine_matrix, _normalize


class TestCosineMatrix(unittest.TestCase):
    def test_empty_matrix_returns_empty_vector(self):
        q = np.array([1, 0, 0], dtype=np.float32)
        out = cosine_matrix(q, np.empty((0, 3), dtype=np.float32))
        self.assertEqual(out.shape, (0,))

    def test_identical_vectors_score_one(self):
        q = _normalize(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        mat = np.stack([q])
        out = cosine_matrix(q, mat)
        np.testing.assert_allclose(out, [1.0], atol=1e-6)

    def test_orthogonal_vectors_score_zero(self):
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        m = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        out = cosine_matrix(q, m)
        np.testing.assert_allclose(out, [0.0], atol=1e-6)

    def test_ranking_order_preserved(self):
        q = _normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        mat = np.stack([
            _normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32)),  # 完全一致
            _normalize(np.array([0.5, 0.5, 0.0], dtype=np.float32)),  # 部分一致
            _normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32)),  # 直交
        ])
        sims = cosine_matrix(q, mat)
        # 降順で並ぶこと
        self.assertGreater(sims[0], sims[1])
        self.assertGreater(sims[1], sims[2])


class TestNormalize(unittest.TestCase):
    def test_unit_vector_unchanged(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(_normalize(v), v, atol=1e-6)

    def test_arbitrary_vector_becomes_unit(self):
        v = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        n = _normalize(v)
        self.assertAlmostEqual(float(np.linalg.norm(n)), 1.0, places=6)

    def test_zero_vector_returns_zero(self):
        v = np.zeros(3, dtype=np.float32)
        np.testing.assert_array_equal(_normalize(v), v)


if __name__ == "__main__":
    unittest.main()
