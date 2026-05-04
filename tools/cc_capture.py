"""Claude Code Stop フック用: 直近の user 発話を記憶に追加する。

stdin に {"transcript_path": "...", ...} が届く。
transcript_path は JSONL (1行1メッセージ) なので末尾を逆走して
直近の user role の text を拾う。LLM 判定は MemoryStore.add() 内部で走る。

settings.json への登録例:
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {"type": "command",
           "command": "C:\\\\Users\\\\user\\\\Desktop\\\\diff-memory\\\\.venv\\\\Scripts\\\\python.exe C:\\\\Users\\\\user\\\\Desktop\\\\diff-memory\\\\tools\\\\cc_capture.py"}
        ]
      }
    ]
  }
"""
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _last_user_text(transcript_path: str) -> str | None:
    if not transcript_path or not os.path.exists(transcript_path):
        return None
    last = None
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Claude Code transcript は {type, message: {role, content}} 構造
            msg = rec.get("message") or rec
            if (msg.get("role") or rec.get("type")) != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                last = content
            elif isinstance(content, list):
                # content blocks → text 結合
                texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                if texts:
                    last = "\n".join(texts)
    return last


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    text = _last_user_text(payload.get("transcript_path", ""))
    if not text or len(text) < 5:
        return 0

    try:
        from diff_memory.config import Config
        from diff_memory.memory import MemoryStore

        store = MemoryStore(Config())
        res = store.add(text)
        # フックは静かに動かす。デバッグしたければ stderr へ。
        sys.stderr.write(f"[diff-memory] action={res.action} mem={res.memory_id} novelty={res.novelty:.2f}\n")
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
