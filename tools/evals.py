"""判定品質の評価セットとランナー。

各ケースは「先行発話を順に流した後、target_text を投入したときに何が起きるべきか」を expected で表現する。
- expected_action: 'create' | 'update' | 'reinforce' | 'ignore' (複数許容セットで定義)
- expected_type:   'preference' | 'fact' | 'project' | 'constraint' | 'episodic' (None で type判定はスキップ)

minipc 経由を想定して per-step に少しスリープ、各エラーは捕捉して続行する。
"""
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diff_memory.config import Config
from diff_memory.memory import MemoryStore


@dataclass
class Case:
    name: str                       # 識別ラベル
    setup: list[str]                # この case 開始前に流す先行発話
    target: str                     # 評価対象の発話
    expected_action: set[str]       # 許容する action (LLM 揺れ吸収のため複数可)
    expected_type: Optional[set[str]] = None
    notes: str = ""


CASES: list[Case] = [
    # ===== 単発 create =====
    Case("c01_tech_fact",          [], "Windows環境でPythonとNode.jsを開発に使っている",
         {"create"}, {"fact", "preference"}),
    Case("c02_hobby",              [], "週末は登山が趣味で月2回は山に行く",
         {"create"}, {"preference"}),
    Case("c03_project",            [], "差分記憶エンジンをローカルLLMで実装中",
         {"create"}, {"project"}),
    Case("c04_constraint",         [], "10時から会議があるので作業は9時半までに切り上げる",
         {"create"}, {"constraint", "episodic"}),
    Case("c05_episodic",           [], "今日は雨で散歩に行けなかった",
         {"create"}, {"episodic"}),

    # ===== ignore (雑談・無意味) =====
    Case("i01_greeting",           [], "おはよう",
         {"ignore"}, None, "挨拶単独"),
    Case("i02_filler",             [], "うーん",
         {"ignore"}, None, "つなぎ語"),
    Case("i03_reaction",           [], "ありがとう",
         {"ignore"}, None, "応答ワード"),
    Case("i04_one_word_question",  [], "なに？",
         {"ignore"}, None, "短い疑問"),
    Case("i05_emoji_only",         [], "😂",
         {"ignore"}, None, "絵文字単独"),
    Case("i06_test_input",         [], "test",
         {"ignore"}, None, "テストワード"),

    # ===== reinforce (同内容を別の言い方で) =====
    Case("r01_same_hobby",
         ["週末は登山が趣味で月2回は山に行く"],
         "登山が好きで週末よく山行く",
         {"reinforce"}, {"preference"}),
    Case("r02_same_tech",
         ["Pythonでよくコード書いてる"],
         "Pythonでプログラミングすることが多い",
         {"reinforce"}, {"preference", "fact"}),
    Case("r03_same_project",
         ["差分記憶エンジンを実装中"],
         "差分メモリエンジンの実装を進めてる",
         {"reinforce"}, {"project"}),

    # ===== update (矛盾・置き換え) =====
    Case("u01_tech_swap",
         ["Node.jsをメインで使ってる"],
         "今はNode.jsよりDeno使ってる",
         {"update"}, {"preference", "fact"}),
    Case("u02_hobby_change",
         ["週1で焼肉食べに行く"],
         "最近は焼肉から寿司に好みが変わった",
         {"update"}, {"preference"}),
    Case("u03_status_change",
         ["差分記憶エンジン実装中"],
         "差分記憶エンジンの実装が完了した",
         {"update", "create"}, {"project"}),
    Case("u04_value_correction",
         ["RAMは32GB"],
         "RAMは48GBの間違いだった",
         {"update"}, {"fact"}),

    # ===== create (関連はあるが別側面) =====
    Case("c06_related_diff_facet",
         ["Pythonで開発してる"],
         "TypeScript もたまに使う",
         {"create", "update", "reinforce"}, {"fact", "preference"},
         "関連あるが言語追加で別情報"),
    Case("c07_subdomain",
         ["Minecraft サーバを家で動かしてる"],
         "Bluemap で 3D マップを公開してる",
         {"create", "reinforce"}, {"project", "fact"}),

    # ===== type 判別が問われる =====
    Case("t01_preference_clear",
         [],
         "コーヒーは深煎り派、浅煎りは苦手",
         {"create"}, {"preference"}),
    Case("t02_fact_clear",
         [],
         "今のメインPCはRyzen 9 7940HS搭載、48GB RAM",
         {"create"}, {"fact"}),
    Case("t03_project_clear",
         [],
         "Minecraftで自動農場 MOD を Java で書いてる",
         {"create"}, {"project"}),
    Case("t04_constraint_clear",
         [],
         "金曜の17時までにレビューを返す必要がある",
         {"create"}, {"constraint"}),
    Case("t05_episodic_clear",
         [],
         "昨日デンタルクリニックで定期検診を受けた",
         {"create"}, {"episodic"}),

    # ===== 紛らわしい境界 =====
    Case("b01_question_not_fact",
         [],
         "Pythonって遅いよね？",
         {"ignore", "create"}, None,
         "問いかけ・主張ぼんやり"),
    Case("b02_negation",
         [],
         "実はNode.jsはあまり使わない",
         {"create"}, {"preference", "fact"}),
    Case("b03_temporal_state",
         [],
         "今エディタで Ctrl+S が効かなくて困ってる",
         {"create"}, {"episodic", "project", "fact"},
         "一時的な状態だが課題情報。fact も許容 (LLM が現状事実として捉える妥当解釈)"),
    Case("b04_meta",
         [],
         "さっき言ったの忘れて",
         {"ignore"}, None,
         "メタ指示、自分自身は覚える対象でない"),
    Case("b05_third_party",
         [],
         "母が最近スマホ買い替えた",
         {"create"}, {"fact", "episodic"},
         "他人の話、保存可"),

    # ===== 長文・複数情報 (現状は1件抽出を期待) =====
    Case("m01_multi_info",
         [],
         "WindowsでPython使ってて、趣味は登山。来週から有給とる予定。",
         {"create"}, None,
         "複数情報が混在、最低1件 create 期待"),

    # ===== 数値の上書き =====
    Case("u05_count_change",
         ["山に月2回行く"],
         "最近は山に月4回くらい行ってる",
         {"update", "reinforce"}, {"preference"}),

    # ===== タイプ移行を許す =====
    Case("u06_episodic_to_pref",
         ["先週ラーメン食べに行った"],
         "ラーメンが好きで週1で食べに行ってる",
         {"create", "update"}, {"preference"}),
]


@dataclass
class Result:
    case: Case
    actual_action: str = ""
    actual_type: str = ""
    actual_text: str = ""
    error: str = ""
    duration_sec: float = 0.0
    action_pass: bool = False
    type_pass: bool = False


def run_one(store: MemoryStore, case: Case) -> Result:
    # 各ケースは独立した DB で実行 (setup の影響を他に漏らさない)
    # 順序重要: 既存接続を閉じてから unlink、その後 reopen
    from diff_memory.db import MemoryDB
    store.db.close()
    if os.path.exists(store.cfg.db_path):
        os.remove(store.cfg.db_path)
    store.db = MemoryDB(store.cfg.db_path)

    try:
        for s in case.setup:
            store.add(s)
            time.sleep(0.2)
        t0 = time.time()
        res = store.add(case.target)
        dt = time.time() - t0
    except Exception as e:
        return Result(case=case, error=f"{type(e).__name__}: {e}", duration_sec=0)

    actual_action = res.action
    actual_type = res.judgment.type if res.action != "ignore" else "-"
    actual_text = res.judgment.text

    action_pass = actual_action in case.expected_action
    type_pass = (case.expected_type is None) or (actual_type in case.expected_type) or (actual_action == "ignore")

    return Result(
        case=case,
        actual_action=actual_action, actual_type=actual_type, actual_text=actual_text,
        duration_sec=dt,
        action_pass=action_pass, type_pass=type_pass,
    )


def main():
    cfg = Config()
    print(f"# embed: {cfg.embed_backend} / {cfg.embed_model}")
    print(f"# llm  : {cfg.llm_model} @ {cfg.ollama_url}")
    print(f"# cases: {len(CASES)}")
    print()

    print("# Initializing store (embedder cold load happens here once)")
    t0 = time.time()
    # 評価専用の DB に分離
    cfg.db_path = "evals.db"
    store = MemoryStore(cfg)
    print(f"# store ready in {time.time()-t0:.1f}s")
    print()

    results: list[Result] = []
    print(f"{'#':>2} {'name':30s} {'expect':30s} {'actual':25s} {'A':3s} {'T':3s} {'sec':>6s}")
    print("-" * 100)
    for i, case in enumerate(CASES, 1):
        r = run_one(store, case)
        results.append(r)
        ax = ",".join(sorted(case.expected_action))
        tx = ",".join(sorted(case.expected_type)) if case.expected_type else "-"
        actual = f"{r.actual_action}/{r.actual_type}"
        a_mark = "OK" if r.action_pass else "NG"
        t_mark = "OK" if r.type_pass else "NG"
        print(f"{i:>2} {case.name:30s} {ax + ' / ' + tx:30s} {actual:25s} {a_mark:3s} {t_mark:3s} {r.duration_sec:>6.1f}")
        if r.error:
            print(f"    ERROR: {r.error}")

    print()
    n = len(results)
    a_pass = sum(1 for r in results if r.action_pass)
    t_pass = sum(1 for r in results if r.type_pass)
    err = sum(1 for r in results if r.error)
    total_sec = sum(r.duration_sec for r in results)
    print("=" * 70)
    print(f"  action  : {a_pass}/{n} = {a_pass/n*100:.1f}%")
    print(f"  type    : {t_pass}/{n} = {t_pass/n*100:.1f}%")
    print(f"  errors  : {err}")
    print(f"  total   : {total_sec:.1f}s ({total_sec/n:.1f}s/case avg)")
    print("=" * 70)

    # 失敗詳細
    print("\n## 失敗ケース")
    for r in results:
        if not r.action_pass:
            print(f"- {r.case.name}: expected action={r.case.expected_action}, "
                  f"got={r.actual_action!r} text={r.actual_text!r}")
        elif not r.type_pass:
            print(f"- {r.case.name}: expected type={r.case.expected_type}, "
                  f"got={r.actual_type!r} text={r.actual_text!r}")

    # JSON でも保存
    out_path = os.environ.get("EVALS_OUT", "evals_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([{
            "name": r.case.name,
            "target": r.case.target,
            "setup": r.case.setup,
            "expected_action": sorted(r.case.expected_action),
            "expected_type": sorted(r.case.expected_type) if r.case.expected_type else None,
            "actual_action": r.actual_action,
            "actual_type": r.actual_type,
            "actual_text": r.actual_text,
            "action_pass": r.action_pass,
            "type_pass": r.type_pass,
            "error": r.error,
            "duration_sec": r.duration_sec,
        } for r in results], f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
