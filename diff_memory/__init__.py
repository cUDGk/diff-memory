"""差分記憶エンジン。重い依存（sentence-transformers）を遅延読み込みするため、
個別モジュール (.memory, .config, .db) から直接 import すること。"""
