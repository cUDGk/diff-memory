import os
from dataclasses import dataclass


@dataclass
class Config:
    db_path: str = os.environ.get("DIFF_MEM_DB", "memory.db")
    # 既定は minipc (Tailscale) の Ollama。MoE Opus4.7 distill が GPU で warm 稼働。
    # ローカル運用に切替えたい場合: DIFF_MEM_OLLAMA_URL=http://localhost:11434
    ollama_url: str = os.environ.get("DIFF_MEM_OLLAMA_URL", "http://100.83.48.127:11434")
    # 既定は minipc 上の qwen36-claude:fixed (Qwen3.6 35B-A3B Claude4.7 Opus distill)。
    # GPU offload (Vulkan) で warm 16 tok/s, 判定 ~1-2s。
    # ローカル CPU で動かす場合は qwen3:4b (~10s/判定) か qwen36-opus47 (warm ~38s/判定、要22GB free)。
    llm_model: str = os.environ.get("DIFF_MEM_MODEL", "qwen36-claude:fixed")
    # 既定は in-process の sentence-transformers (CLI 終了で解放されるので Ollama 側 RAM を圧迫しない)。
    # 完全 torch 不要にしたい場合: DIFF_MEM_EMBED_BACKEND=ollama DIFF_MEM_EMBED_MODEL=bge-m3
    embed_backend: str = os.environ.get("DIFF_MEM_EMBED_BACKEND", "sentence-transformers")
    embed_model: str = os.environ.get("DIFF_MEM_EMBED_MODEL", "intfloat/multilingual-e5-small")
    novelty_threshold: float = float(os.environ.get("DIFF_MEM_NOVELTY", "0.25"))
    top_n_candidates: int = int(os.environ.get("DIFF_MEM_TOP_N", "5"))
    llm_timeout_sec: int = int(os.environ.get("DIFF_MEM_LLM_TIMEOUT", "600"))
    # minipc 既定だと他のスクリプトと GPU 取り合うため控えめ。専用機なら "6h" 等に。
    llm_keep_alive: str = os.environ.get("DIFF_MEM_KEEP_ALIVE", "5m")

    # スコアリング重み (search 時)
    w_similarity: float = 0.45
    w_importance: float = 0.25
    w_stability: float = 0.15
    w_recency: float = 0.10
    w_confidence: float = 0.05

    # type 別の decay 時定数 (日)。importance *= exp(-Δt / tau)
    # 高 stability ほど tau 大 → 減衰遅い。type で上書き可能。
    tau_by_type: dict = None

    def __post_init__(self):
        if self.tau_by_type is None:
            self.tau_by_type = {
                "preference": 365.0,
                "constraint": 365.0,
                "fact": 180.0,
                "project": 60.0,
                "episodic": 14.0,
            }
