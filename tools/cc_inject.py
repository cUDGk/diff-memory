"""Claude Code UserPromptSubmit フック用: ユーザー発話に類似する記憶を追記する。

settings.json への登録例:
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {"type": "command",
           "command": "C:\\\\Users\\\\user\\\\Desktop\\\\diff-memory\\\\.venv\\\\Scripts\\\\python.exe C:\\\\Users\\\\user\\\\Desktop\\\\diff-memory\\\\tools\\\\cc_inject.py"}
        ]
      }
    ]
  }

stdin に Claude Code が JSON を流す ({"prompt": "...", ...})。
stdout に出した内容が追加コンテキストとしてプロンプトに付く。
"""
import json
import os
import sys
import traceback

# diff_memory パッケージを import path に追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

K = int(os.environ.get("DIFF_MEM_INJECT_K", "5"))
MIN_SCORE = float(os.environ.get("DIFF_MEM_INJECT_MIN_SCORE", "0.30"))


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        # 単独実行で stdin が無い時はエラーを出さず無音で抜ける
        return 0

    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return 0

    try:
        from diff_memory.config import Config
        from diff_memory.memory import MemoryStore

        store = MemoryStore(Config())
        results = store.query(prompt, k=K)
        results = [r for r in results if r.score >= MIN_SCORE]
        if not results:
            return 0

        lines = ["<diff-memory>",
                 "# 関連する過去の記憶 (score 順、score>=%.2f)" % MIN_SCORE]
        for r in results:
            m = r.memory
            lines.append(f"- [{m.type}] {m.text}  "
                         f"(score={r.score:.2f} sim={r.similarity:.2f} "
                         f"conf={m.confidence:.2f} n={m.source_count})")
        lines.append("</diff-memory>")
        sys.stdout.write("\n".join(lines) + "\n")
    except Exception:
        # フック失敗で会話を止めないため stderr に書いて 0 で抜ける
        traceback.print_exc(file=sys.stderr)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
