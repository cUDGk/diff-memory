"""Ollama judgment client.

minipc の Ollama を JSON モードで叩き、判定結果を dict で返す。
"""
import json
from dataclasses import dataclass
from typing import Optional

import requests


VALID_TYPES = ("preference", "fact", "project", "constraint", "episodic")
VALID_ACTIONS = ("create", "update", "reinforce", "ignore")


@dataclass
class Judgment:
    action: str            # create | update | reinforce | ignore
    target_id: Optional[str] = None
    type: str = "fact"
    text: str = ""
    importance: float = 0.5
    stability: float = 0.5
    confidence: float = 0.5
    contradicts_ids: list = None
    reason: str = ""

    def __post_init__(self):
        if self.contradicts_ids is None:
            self.contradicts_ids = []


SYSTEM = """あなたは記憶エンジンの判定モジュールです。出力は必ず JSON のみ。説明文・コードブロック・前置きは禁止。"""

NEW_PROMPT = """新しいユーザー発話を分析し、保存する記憶を1件抽出してください（似た既存記憶は無し）。

# 入力
"{text}"

# 出力 (JSON のみ)
{{
  "action": "create" または "ignore",
  "type": "preference|fact|project|constraint|episodic のどれか1つ",
  "text": "簡潔な要約（1文、主語省略しない）",
  "importance": 0.0〜1.0,
  "stability": 0.0〜1.0,
  "confidence": 0.0〜1.0,
  "reason": "短い判断理由"
}}

# 判定ガイド
- ignore: 雑談・挨拶・一時的な気分・保存価値が無い独り言
- preference: ユーザーの好み・スタイル・嫌悪
- fact: 客観的な事実・属性・環境
- project: 進行中のタスク・目標
- constraint: 守るべき制約・ルール・締切
- episodic: 一時的な出来事・体験
- importance: ユーザーがそれを将来覚えておくと役立つ度合い
- stability: その情報がどれくらい長く有効か（一過性=低、属性=高）
- confidence: 発話から読み取れる確信度
"""

UPDATE_PROMPT = """新しいユーザー発話と既存類似記憶を比較し、1つのアクションを選んでください。

# 新発話
"{text}"

# 既存類似記憶（id順）
{candidates}

# 出力 (JSON のみ)
{{
  "action": "create | update | reinforce | ignore",
  "target_id": "update/reinforce の場合のみ既存ID",
  "type": "preference|fact|project|constraint|episodic",
  "text": "保存/更新後の本文（1文）",
  "importance": 0.0〜1.0,
  "stability": 0.0〜1.0,
  "confidence": 0.0〜1.0,
  "contradicts_ids": ["矛盾する古い記憶のid（あれば）"],
  "reason": "短い判断理由"
}}

# アクションの使い分け
- create: 既存と関連はあるが別の新情報（同じ話題でも別側面）
- update: 既存の記憶を上書き／改変。内容が変わった・矛盾する場合は古いidを contradicts_ids に列挙
- reinforce: 同じ内容の繰り返し・追認。本文は既存をそのまま採用してよい
- ignore: 既存で十分・保存価値が無い
"""


class OllamaJudge:
    def __init__(self, base_url: str, model: str, timeout: int = 120, keep_alive: str = "5m"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.keep_alive = keep_alive

    def ping(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def _call(self, prompt: str) -> dict:
        r = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "system": SYSTEM,
                "format": "json",
                "stream": False,
                "think": False,  # Qwen3等のthinkingモデルでJSONがthinkingに吸われるのを防ぐ
                "keep_alive": self.keep_alive,  # MoE再ロードの頻発を防ぐ
                "options": {"temperature": 0.1},
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        body = r.json()
        # think: False が無視されるサーバ向けに thinking を保険として読む
        raw = body.get("response") or body.get("thinking") or "{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # 余分テキスト混入時は最初の { から最後の } まで切り出す
            i, j = raw.find("{"), raw.rfind("}")
            if i >= 0 and j > i:
                return json.loads(raw[i:j + 1])
            raise ValueError(f"LLM returned non-JSON: {raw[:200]!r}")

    @staticmethod
    def _normalize(d: dict) -> Judgment:
        action = (d.get("action") or "ignore").lower().strip()
        if action not in VALID_ACTIONS:
            action = "ignore"
        type_ = (d.get("type") or "fact").lower().strip()
        if type_ not in VALID_TYPES:
            type_ = "fact"
        return Judgment(
            action=action,
            target_id=d.get("target_id"),
            type=type_,
            text=(d.get("text") or "").strip(),
            importance=_clip01(d.get("importance", 0.5)),
            stability=_clip01(d.get("stability", 0.5)),
            confidence=_clip01(d.get("confidence", 0.5)),
            contradicts_ids=d.get("contradicts_ids") or [],
            reason=(d.get("reason") or "").strip(),
        )

    def judge_new(self, text: str) -> Judgment:
        d = self._call(NEW_PROMPT.format(text=text))
        return self._normalize(d)

    def judge_with_candidates(self, text: str, candidates: list[dict]) -> Judgment:
        cand_str = "\n".join(
            f"- id={c['id']} type={c['type']} sim={c['sim']:.3f} conf={c['confidence']:.2f}: {c['text']}"
            for c in candidates
        )
        d = self._call(UPDATE_PROMPT.format(text=text, candidates=cand_str))
        return self._normalize(d)


def _clip01(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, v))
