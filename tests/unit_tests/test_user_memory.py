"""单元测试：长期记忆按 user_id 隔离 + 旧表平滑迁移。"""

import sqlite3

from opendetect_ai import user_memory


def _point_db(monkeypatch, tmp_path):
    db = str(tmp_path / "mem.db")
    monkeypatch.setattr(user_memory, "_get_db_path", lambda: db)
    return db


def test_profiles_are_isolated_by_user(monkeypatch, tmp_path) -> None:
    _point_db(monkeypatch, tmp_path)
    user_memory.save_user_profile("alice", {"research_interests": ["computer vision"]})
    user_memory.save_user_profile("bob",   {"research_interests": ["nlp"]})

    assert user_memory.load_user_profile("alice") == {"research_interests": ["computer vision"]}
    assert user_memory.load_user_profile("bob")   == {"research_interests": ["nlp"]}
    assert user_memory.load_user_profile("carol") == {}   # 未知用户，互不串画像


def test_migration_from_legacy_schema(monkeypatch, tmp_path) -> None:
    """旧表（key 作主键、无 user_id）应被平滑迁移，历史画像归到 'default'。"""
    db = _point_db(monkeypatch, tmp_path)
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE user_profile (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)")
    conn.execute("INSERT INTO user_profile VALUES ('research_interests', '[\"detection\"]', '2020-01-01')")
    conn.commit()
    conn.close()

    # 首次读取触发迁移
    prof = user_memory.load_user_profile("default")
    assert prof == {"research_interests": ["detection"]}

    # 迁移后表结构应含 user_id 列
    conn = sqlite3.connect(db)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(user_profile)").fetchall()]
    conn.close()
    assert "user_id" in cols
