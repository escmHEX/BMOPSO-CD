from __future__ import annotations

import pytest

from binary_mopso_cd.services.ppdb import PPDBSQLiteIndex, resolve_config_path
from binary_mopso_cd.services.turbulence import WordNetPPDBProvider


def write_ppdb(path, rows):
    path.write_text("\n".join(f"[X] ||| {left} ||| {right} ||| features" for left, right in rows), encoding="utf-8")


def test_ppdb_sqlite_builds_full_index_without_per_key_limit(tmp_path):
    source = tmp_path / "ppdb.txt"
    rows = [("help", f"aid{i}") for i in range(40)]
    write_ppdb(source, rows)
    index_path = tmp_path / "ppdb.sqlite"

    index = PPDBSQLiteIndex(index_path, source)

    assert index.lookup("help") == [f"aid{i}" for i in range(40)]
    assert index.lookup("aid3") == ["help"]
    index.close()


def test_ppdb_sqlite_does_not_rebuild_existing_index(tmp_path):
    source = tmp_path / "ppdb.txt"
    write_ppdb(source, [("help", "aid")])
    index_path = tmp_path / "ppdb.sqlite"
    first = PPDBSQLiteIndex(index_path, source)
    first.close()
    write_ppdb(source, [("help", "changed")])

    second = PPDBSQLiteIndex(index_path, source)

    assert second.lookup("help") == ["aid"]
    second.close()


def test_ppdb_sqlite_logs_first_time_index_build(tmp_path, capsys):
    source = tmp_path / "ppdb.txt"
    write_ppdb(source, [("help", "aid")])
    index_path = tmp_path / "ppdb.sqlite"

    index = PPDBSQLiteIndex(index_path, source)
    index.close()

    captured = capsys.readouterr()
    assert "Building PPDB SQLite index" in captured.out
    assert str(source) in captured.out
    assert str(index_path) in captured.out
    assert "PPDB SQLite index ready" in captured.out


def test_ppdb_sqlite_logs_existing_index_reuse(tmp_path, capsys):
    source = tmp_path / "ppdb.txt"
    write_ppdb(source, [("help", "aid")])
    index_path = tmp_path / "ppdb.sqlite"
    first = PPDBSQLiteIndex(index_path, source)
    first.close()
    capsys.readouterr()

    second = PPDBSQLiteIndex(index_path, source)
    second.close()

    captured = capsys.readouterr()
    assert "Using existing PPDB SQLite index" in captured.out
    assert str(index_path) in captured.out
    assert "Building PPDB SQLite index" not in captured.out


def test_ppdb_sqlite_fails_when_index_and_source_are_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="models.ppdb.source_path"):
        PPDBSQLiteIndex(tmp_path / "missing.sqlite", tmp_path / "missing.ppdb")


def test_ppdb_venv_placeholder_resolves_against_active_environment(tmp_path, monkeypatch):
    from binary_mopso_cd.services import ppdb

    monkeypatch.setattr(ppdb.sys, "prefix", str(tmp_path / "env"))

    assert resolve_config_path("{venv}/var/binary_mopso_cd/ppdb.sqlite") == (
        tmp_path / "env" / "var" / "binary_mopso_cd" / "ppdb.sqlite"
    )


def test_wordnet_ppdb_operator_keeps_word_replacements_only(tmp_path):
    source = tmp_path / "ppdb.txt"
    write_ppdb(source, [("help", "first aid"), ("help", "support")])
    index = PPDBSQLiteIndex(tmp_path / "ppdb.sqlite", source)
    provider = WordNetPPDBProvider(index, use_wordnet=False)

    assert provider.candidates("help", target_index=0, max_variants=5, use_ppdb=True) == ["support"]
    index.close()
