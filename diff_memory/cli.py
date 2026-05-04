"""CLI entrypoint: python -m diff_memory <subcommand>"""
import json
import sys

import typer

from .config import Config
from .memory import MemoryStore

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="差分記憶エンジン: 新規性のある情報だけ保存するメモリ層")


def _store() -> MemoryStore:
    return MemoryStore(Config())


@app.command()
def add(text: str = typer.Argument(..., help="保存候補のテキスト")):
    """発話を入れる → LLM が create / update / reinforce / ignore を判定。"""
    s = _store()
    res = s.add(text)
    typer.echo(json.dumps({
        "action": res.action,
        "memory_id": res.memory_id,
        "novelty": round(res.novelty, 3),
        "type": res.judgment.type,
        "text": res.judgment.text,
        "importance": res.judgment.importance,
        "stability": res.judgment.stability,
        "confidence": res.judgment.confidence,
        "reason": res.judgment.reason,
    }, ensure_ascii=False, indent=2))


@app.command()
def query(
    q: str = typer.Argument(..., help="検索クエリ"),
    k: int = typer.Option(5, "-k", help="返す件数"),
    type_: str = typer.Option(None, "--type", help="type で絞り込み"),
    min_conf: float = typer.Option(0.0, "--min-conf", help="最低confidence"),
):
    """類似+重み付きスコアで上位K件を取得。"""
    s = _store()
    results = s.query(q, k=k, type_filter=type_, min_confidence=min_conf)
    out = [{
        "id": r.memory.id,
        "type": r.memory.type,
        "text": r.memory.text,
        "score": round(r.score, 3),
        "sim": round(r.similarity, 3),
        "recency": round(r.recency, 3),
        "importance": round(r.memory.importance, 3),
        "stability": round(r.memory.stability, 3),
        "confidence": round(r.memory.confidence, 3),
        "source_count": r.memory.source_count,
    } for r in results]
    typer.echo(json.dumps(out, ensure_ascii=False, indent=2))


@app.command()
def inject(
    q: str = typer.Argument(..., help="検索クエリ"),
    k: int = typer.Option(5, "-k", help="返す件数"),
):
    """プロンプト注入用のプレーンテキストを出力。"""
    s = _store()
    typer.echo(s.render_for_prompt(q, k=k))


@app.command(name="list")
def list_(
    type_: str = typer.Option(None, "--type", help="type で絞り込み"),
    limit: int = typer.Option(50, "-n", help="表示件数"),
):
    """全記憶を簡易リスト表示。"""
    s = _store()
    mems = s.db.all()
    if type_:
        mems = [m for m in mems if m.type == type_]
    mems.sort(key=lambda m: m.last_seen, reverse=True)
    out = [{
        "id": m.id, "type": m.type, "text": m.text,
        "imp": round(m.importance, 2), "stab": round(m.stability, 2),
        "conf": round(m.confidence, 2), "n": m.source_count,
        "last_seen": m.last_seen,
    } for m in mems[:limit]]
    typer.echo(json.dumps(out, ensure_ascii=False, indent=2))


@app.command()
def show(mem_id: str):
    """1件の詳細 + source_log を出力。"""
    s = _store()
    m = s.db.get(mem_id)
    if not m:
        typer.echo(f"not found: {mem_id}", err=True)
        raise typer.Exit(1)
    typer.echo(json.dumps({
        "memory": m.to_dict(),
        "history": s.db.history(mem_id),
    }, ensure_ascii=False, indent=2))


@app.command()
def decay():
    """type 別 tau で importance を時間減衰。"""
    s = _store()
    n = s.decay()
    typer.echo(f"decayed {n} memories")


@app.command()
def delete(mem_id: str, yes: bool = typer.Option(False, "--yes")):
    """1件削除。--yes が無いと確認。"""
    s = _store()
    if not yes:
        typer.confirm(f"delete {mem_id}?", abort=True)
    s.db.delete(mem_id)
    typer.echo(f"deleted {mem_id}")


@app.command()
def doctor():
    """設定と接続性を確認。"""
    cfg = Config()
    typer.echo(f"db_path        : {cfg.db_path}")
    typer.echo(f"embed_backend  : {cfg.embed_backend}")
    typer.echo(f"embed_model    : {cfg.embed_model}")
    typer.echo(f"ollama_url     : {cfg.ollama_url}")
    typer.echo(f"llm_model      : {cfg.llm_model}")
    typer.echo(f"keep_alive     : {cfg.llm_keep_alive}")
    typer.echo(f"novelty_th     : {cfg.novelty_threshold}")

    from .llm import OllamaJudge
    j = OllamaJudge(cfg.ollama_url, cfg.llm_model, cfg.llm_timeout_sec)
    ok = j.ping()
    typer.echo(f"ollama ping : {'OK' if ok else 'NG'}")
    if not ok:
        sys.exit(2)


def main():
    app()


if __name__ == "__main__":
    main()
