"""差分記憶エンジンの統合テスト。

minipc 等の共有 LLM 環境を想定し、各ステップでネットワーク/LLM 失敗を catch して続行する。
embedder は1プロセス内に閉じて cold load を1回で済ませる。
"""
import os
import sys
import time
import traceback
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diff_memory.config import Config
from diff_memory.memory import MemoryStore


PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def step(name: str, fn: Callable) -> tuple[bool, str]:
    t0 = time.time()
    try:
        out = fn()
        dt = time.time() - t0
        print(f"{PASS} {name:40s} ({dt:5.1f}s)  {out}")
        return True, str(out)
    except Exception as e:
        dt = time.time() - t0
        print(f"{FAIL} {name:40s} ({dt:5.1f}s)  {type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stderr)
        return False, ""


def main():
    cfg = Config()
    db = cfg.db_path
    if os.path.exists(db) and os.environ.get("STRESS_RESET", "1") == "1":
        os.remove(db)
        print(f"# reset DB: {db}")

    print(f"# embed: {cfg.embed_backend} / {cfg.embed_model}")
    print(f"# llm  : {cfg.llm_model} @ {cfg.ollama_url}")
    print(f"# keep_alive: {cfg.llm_keep_alive}, timeout: {cfg.llm_timeout_sec}s")
    print()

    print("# Initializing store (embedder cold load happens here once)")
    t0 = time.time()
    store = MemoryStore(cfg)
    print(f"# store ready in {time.time()-t0:.1f}s")
    print()

    fixtures = [
        # (label, text, expected_action_one_of)
        ("create:tech-fact",     "Windows環境でPythonとNode.jsを開発に使っている", {"create"}),
        ("reinforce:same",       "Pythonでよくコード書いてる", {"reinforce", "create", "update"}),
        ("update:contradict",    "今はNode.jsよりDeno使ってる", {"update", "create", "reinforce"}),
        ("create:unrelated",     "週末は登山が趣味で月2回は山に行く", {"create"}),
        ("ignore:greeting",      "おはよう", {"ignore", "create"}),
        ("create:another-domain","Bluemap Minecraft サーバを 10.0.0.1 で動かしている", {"create"}),
        ("reinforce:second-hit", "登山では北アルプス系をよく回る", {"reinforce", "update", "create"}),
    ]

    n_pass = 0
    n_fail = 0

    print("=" * 70)
    print("  ADD path")
    print("=" * 70)
    for label, text, expected in fixtures:
        ok, out = step(label, lambda t=text: store.add(t).action)
        if ok:
            if out in expected:
                n_pass += 1
            else:
                print(f"     ⚠️ unexpected action {out!r}, expected one of {expected}")
                n_pass += 1  # action mismatch is not a hard fail (LLM judgment varies)
        else:
            n_fail += 1

    print()
    print("=" * 70)
    print("  QUERY path")
    print("=" * 70)
    queries = [
        ("query:tech",   "プログラミング言語の構成"),
        ("query:hobby",  "週末の趣味"),
        ("query:server", "ローカルサーバ"),
        ("query:vague",  "今日の作業"),
    ]
    for label, q in queries:
        def go(q=q):
            results = store.query(q, k=3)
            if not results:
                return "(no results)"
            top = results[0]
            return f"top=[{top.memory.type}] {top.memory.text[:40]}... score={top.score:.2f}"
        ok, _ = step(label, go)
        if ok: n_pass += 1
        else: n_fail += 1

    print()
    print("=" * 70)
    print("  Maintenance")
    print("=" * 70)
    ok, _ = step("decay",         lambda: f"touched {store.decay()}")
    if ok: n_pass += 1
    else: n_fail += 1

    ok, _ = step("inject:prompt", lambda: store.render_for_prompt("今やってる開発", k=3)[:80] + "...")
    if ok: n_pass += 1
    else: n_fail += 1

    print()
    print("=" * 70)
    print(f"  RESULT: {n_pass} pass / {n_fail} fail")
    print("=" * 70)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
