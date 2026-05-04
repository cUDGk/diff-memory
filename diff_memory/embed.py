"""Embedding 抽象。

backend を 'sentence-transformers' (ローカル, torch必要) と
'ollama' (Ollama サーバ経由, torch不要) の2種から選択可能。
e5系は 'query:' / 'passage:' prefix を付けると精度が出る。
"""
from typing import Protocol

import numpy as np
import requests


class Embedder(Protocol):
    dim: int
    def embed_passage(self, text: str) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def cosine_matrix(query: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """query: (D,)  mat: (N, D)  両方 L2-normalized 前提 → 単純内積でコサイン。"""
    if mat.size == 0:
        return np.array([], dtype=np.float32)
    return mat @ query


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ---------- sentence-transformers backend ----------

class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import
        self.model_name = model_name
        self._is_e5 = "e5" in model_name.lower()
        self.model = SentenceTransformer(model_name)
        # 新APIフォールバック付き (sentence-transformers 5.x で改名)
        get_dim = (getattr(self.model, "get_embedding_dimension", None)
                   or self.model.get_sentence_embedding_dimension)
        self.dim = int(get_dim())

    def _encode(self, text: str, prefix: str) -> np.ndarray:
        s = f"{prefix}: {text}" if self._is_e5 else text
        v = self.model.encode(s, normalize_embeddings=True, convert_to_numpy=True)
        return v.astype(np.float32)

    def embed_passage(self, text: str) -> np.ndarray:
        return self._encode(text, "passage")

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode(text, "query")


# ---------- Ollama backend ----------

class OllamaEmbedder:
    """Ollama の /api/embed を使う。bge-m3 / nomic-embed-text 等。
    bge-m3 は出力が L2 normalized 済み。それ以外でも安全のため再正規化する。
    """

    def __init__(self, base_url: str, model: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        # 初回 embed で dim 確定。先に1回叩いて取得する。
        v = self._embed("__init__")
        self.dim = int(v.shape[0])

    def _embed(self, text: str) -> np.ndarray:
        r = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": text},
            timeout=self.timeout,
        )
        r.raise_for_status()
        body = r.json()
        embs = body.get("embeddings") or []
        if not embs:
            raise ValueError(f"no embeddings in response: {body}")
        v = np.asarray(embs[0], dtype=np.float32)
        return _normalize(v)

    def embed_passage(self, text: str) -> np.ndarray:
        return self._embed(text)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)


# ---------- factory ----------

def make_embedder(backend: str, model: str, ollama_url: str = "http://localhost:11434") -> Embedder:
    backend = backend.lower()
    if backend in ("st", "sentence-transformers", "local"):
        return SentenceTransformerEmbedder(model)
    if backend in ("ollama",):
        return OllamaEmbedder(ollama_url, model)
    raise ValueError(f"unknown embed backend: {backend!r} (use 'sentence-transformers' or 'ollama')")
