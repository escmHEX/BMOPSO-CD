from __future__ import annotations

import gzip
import os
import sqlite3
import sys
from collections.abc import Iterable
from pathlib import Path

from binary_mopso_cd.utils import canonical_text


SCHEMA_VERSION = 1
DEFAULT_BATCH_SIZE = 10_000


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def environment_root() -> Path:
    return Path(sys.prefix).resolve()


def resolve_config_path(value: str | None, base_dir: Path | None = None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    root = base_dir or project_root()
    text = str(value).replace("{venv}", str(environment_root()))
    path = Path(text)
    return path if path.is_absolute() else (root / path).resolve()


def open_ppdb_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def parse_ppdb_line(line: str) -> tuple[str, str] | None:
    parts = [part.strip() for part in line.split("|||")]
    if len(parts) >= 3:
        return parts[1], parts[2]
    tab_parts = [part.strip() for part in line.split("\t")]
    if len(tab_parts) >= 2:
        return tab_parts[0], tab_parts[1]
    return None


class PPDBSQLiteIndex:
    def __init__(
        self,
        index_path: Path | None,
        source_path: Path | None = None,
        *,
        enabled: bool = True,
        auto_build: bool = True,
    ):
        self.index_path = index_path
        self.source_path = source_path
        self.enabled = enabled
        self._connection: sqlite3.Connection | None = None
        if not enabled:
            return
        if index_path is None:
            raise ValueError("models.ppdb.index_path must be configured when PPDB is enabled")
        if not index_path.exists():
            if not auto_build:
                raise FileNotFoundError(f"PPDB index not found: {index_path}")
            if source_path is None or not source_path.exists():
                raise FileNotFoundError(
                    "PPDB index does not exist and source PPDB file was not found. "
                    f"Configure models.ppdb.source_path or create the index at {index_path}."
                )
            build_sqlite_index(source_path, index_path)

    @property
    def connection(self) -> sqlite3.Connection:
        if self.index_path is None:
            raise RuntimeError("PPDB index is not configured")
        if self._connection is None:
            self._connection = sqlite3.connect(self.index_path)
        return self._connection

    def lookup(self, word: str) -> list[str]:
        if not self.enabled:
            return []
        key = canonical_text(word)
        if not key:
            return []
        rows = self.connection.execute(
            "SELECT value FROM ppdb_entries WHERE key = ? ORDER BY source_order",
            (key,),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None


def build_sqlite_index(source_path: Path, index_path: Path, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = index_path.with_suffix(index_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    connection = sqlite3.connect(temporary_path)
    try:
        configure_build_connection(connection)
        create_schema(connection)
        entry_count = insert_ppdb_entries(connection, source_path, batch_size)
        connection.execute(
            "INSERT INTO ppdb_metadata(key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        connection.execute(
            "INSERT INTO ppdb_metadata(key, value) VALUES (?, ?)",
            ("source_path", str(source_path)),
        )
        connection.execute(
            "INSERT INTO ppdb_metadata(key, value) VALUES (?, ?)",
            ("entry_count", str(entry_count)),
        )
        connection.commit()
    except Exception:
        connection.close()
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    connection.close()
    os.replace(temporary_path, index_path)


def configure_build_connection(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.execute("PRAGMA locking_mode = EXCLUSIVE")


def create_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE ppdb_entries (
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            source_order INTEGER NOT NULL,
            PRIMARY KEY (key, value)
        ) WITHOUT ROWID
        """
    )
    connection.execute(
        """
        CREATE TABLE ppdb_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )


def insert_ppdb_entries(connection: sqlite3.Connection, source_path: Path, batch_size: int) -> int:
    total = 0
    batch: list[tuple[str, str, int]] = []
    with open_ppdb_text(source_path) as handle:
        for source_order, line in enumerate(handle):
            parsed = parse_ppdb_line(line)
            if parsed is None:
                continue
            left, right = parsed
            batch.extend(entry_rows(left, right, source_order))
            if len(batch) >= batch_size:
                total += insert_batch(connection, batch)
                batch.clear()
        if batch:
            total += insert_batch(connection, batch)
    return total


def entry_rows(left: str, right: str, source_order: int) -> list[tuple[str, str, int]]:
    left_key = canonical_text(left)
    right_key = canonical_text(right)
    rows: list[tuple[str, str, int]] = []
    if left_key and right_key and left_key != right_key:
        rows.append((left_key, right.strip(), source_order))
        rows.append((right_key, left.strip(), source_order))
    return rows


def insert_batch(connection: sqlite3.Connection, rows: Iterable[tuple[str, str, int]]) -> int:
    before = connection.total_changes
    connection.executemany(
        "INSERT OR IGNORE INTO ppdb_entries(key, value, source_order) VALUES (?, ?, ?)",
        rows,
    )
    connection.commit()
    return connection.total_changes - before
