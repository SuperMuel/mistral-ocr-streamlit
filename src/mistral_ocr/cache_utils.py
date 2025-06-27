from __future__ import annotations

"""Utilities for caching OCR results locally."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import hashlib
import sqlite3

from platformdirs import user_cache_dir

CACHE_DIR_NAME = "mistral-ocr"
DB_NAME = "cache.db"


@dataclass(slots=True)
class CacheEntry:
    """Represents a single cached OCR result."""

    pdf_hash: str
    filename: str
    source_path: str
    size_bytes: int
    markdown_content: str
    created_at: str
    last_accessed: str
    mistral_model: str


class Cache:
    """SQLite-backed cache for OCR results."""

    def __init__(self, enabled: bool = True, cache_dir: Path | None = None) -> None:
        self.enabled = enabled
        if not enabled:
            self.db_path = None
            return
        self.cache_dir = cache_dir or Path(user_cache_dir(CACHE_DIR_NAME))
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.enabled = False
            self.db_path = None
            return
        self.db_path = self.cache_dir / DB_NAME
        self._initialize_db()

    def _initialize_db(self) -> None:
        if not self.enabled or self.db_path is None:
            return
        with sqlite3.connect(self.db_path, timeout=5) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    pdf_hash TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    markdown_content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    mistral_model TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_filename ON cache_entries(filename);"
            )

    def _connect(self) -> sqlite3.Connection:
        assert self.db_path is not None
        return sqlite3.connect(self.db_path, timeout=5)

    def get(self, pdf_hash: str) -> CacheEntry | None:
        if not self.enabled or self.db_path is None:
            return None
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM cache_entries WHERE pdf_hash=?",
                (pdf_hash,),
            )
            row = cur.fetchone()
            if not row:
                return None
            entry = CacheEntry(*row)
            conn.execute(
                "UPDATE cache_entries SET last_accessed=? WHERE pdf_hash=?",
                (datetime.now(UTC).isoformat(), pdf_hash),
            )
            return entry

    def set(self, entry: CacheEntry) -> None:
        if not self.enabled or self.db_path is None:
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache_entries (
                    pdf_hash, filename, source_path, size_bytes,
                    markdown_content, created_at, last_accessed, mistral_model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.pdf_hash,
                    entry.filename,
                    entry.source_path,
                    entry.size_bytes,
                    entry.markdown_content,
                    entry.created_at,
                    entry.last_accessed,
                    entry.mistral_model,
                ),
            )

    def clear(self) -> None:
        if not self.enabled or self.db_path is None:
            return
        if self.db_path.exists():
            self.db_path.unlink()
        self._initialize_db()

    def stats(self) -> dict[str, int | str]:
        if not self.enabled or self.db_path is None:
            return {}
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT COUNT(*), SUM(LENGTH(markdown_content)) FROM cache_entries"
            )
            total_entries, total_size = cur.fetchone()
            cur = conn.execute(
                "SELECT MIN(created_at), MAX(created_at) FROM cache_entries"
            )
            oldest, newest = cur.fetchone()
        return {
            "total_entries": total_entries or 0,
            "total_size": total_size or 0,
            "oldest": oldest or "",
            "newest": newest or "",
        }


def compute_pdf_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a PDF file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
