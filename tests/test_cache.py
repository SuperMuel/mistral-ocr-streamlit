from pathlib import Path
from datetime import UTC, datetime

from mistral_ocr.cache_utils import Cache, CacheEntry, compute_pdf_hash
from tests.test_utils import create_single_test_pdf


def test_compute_pdf_hash_consistency(tmp_path: Path) -> None:
    pdf = tmp_path / "test.pdf"
    create_single_test_pdf(pdf, "hash test")
    h1 = compute_pdf_hash(pdf)
    h2 = compute_pdf_hash(pdf)
    assert h1 == h2


def test_cache_store_and_retrieve(tmp_path: Path) -> None:
    cache = Cache(enabled=True, cache_dir=tmp_path)
    entry = CacheEntry(
        pdf_hash="abc",
        filename="a.pdf",
        source_path="/tmp/a.pdf",
        size_bytes=10,
        markdown_content="content",
        created_at=datetime.now(UTC).isoformat(),
        last_accessed=datetime.now(UTC).isoformat(),
        mistral_model="test",
    )
    cache.set(entry)
    loaded = cache.get("abc")
    assert loaded is not None
    assert loaded.markdown_content == "content"


def test_cache_clear(tmp_path: Path) -> None:
    cache = Cache(enabled=True, cache_dir=tmp_path)
    entry = CacheEntry(
        pdf_hash="abc",
        filename="a.pdf",
        source_path="/tmp/a.pdf",
        size_bytes=10,
        markdown_content="content",
        created_at=datetime.now(UTC).isoformat(),
        last_accessed=datetime.now(UTC).isoformat(),
        mistral_model="test",
    )
    cache.set(entry)
    cache.clear()
    stats = cache.stats()
    assert stats.get("total_entries", 0) == 0
