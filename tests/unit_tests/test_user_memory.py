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


def test_memory_can_be_disabled_and_reenabled(monkeypatch, tmp_path) -> None:
    _point_db(monkeypatch, tmp_path)
    user_memory.save_user_profile("alice", {"research_interests": ["vision"]})
    user_memory.set_memory_settings("alice", enabled=False)
    assert user_memory.load_user_profile("alice") == {}

    # 关闭期间的新提取不应写入；重新开启后旧数据仍由用户自行决定是否删除。
    user_memory.save_user_profile("alice", {"research_interests": ["nlp"]})
    user_memory.set_memory_settings("alice", enabled=True)
    assert user_memory.load_user_profile("alice") == {"research_interests": ["vision"]}


def test_memory_ttl_and_metadata(monkeypatch, tmp_path) -> None:
    db = _point_db(monkeypatch, tmp_path)
    user_memory.save_user_profile(
        "alice", {"research_interests": ["vision"]}, source="explicit"
    )
    user_memory.set_memory_settings("alice", ttl_days=1)
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE user_profile SET updated_at='2020-01-01T00:00:00+00:00' WHERE user_id='alice'"
    )
    conn.commit()
    conn.close()

    assert user_memory.load_user_profile("alice") == {}
    entries = user_memory.list_memory_entries("alice")
    assert entries[0]["source"] == "explicit"
    assert entries[0]["updated_at"].startswith("2020-")
