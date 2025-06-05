from pathlib import Path
from typing import Any

import pytest

from mistral_ocr import (
    find_pdf_files,
    determine_output_directory,
    create_conversion_plan,
    process_and_save_pdf,
)


class DummyClient:
    pass


class DummyResponse:
    pass


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("doc.pdf", 1),
        ("DOC.PDF", 1),
        ("not_pdf.txt", 0),
    ],
)
def test_find_pdf_files_file(tmp_path: Path, filename: str, expected: int) -> None:
    file_path = tmp_path / filename
    file_path.write_text("content")
    result = find_pdf_files(file_path)
    assert len(result) == expected


def test_find_pdf_files_directory(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    files = [
        tmp_path / "a.pdf",
        tmp_path / "sub" / "b.PDF",
        tmp_path / "c.txt",
    ]
    for f in files:
        f.write_text("data")
    result = find_pdf_files(tmp_path)
    assert set(p.name for p in result) == {"a.pdf", "b.PDF"}


def test_find_pdf_files_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    result = find_pdf_files(missing)
    assert result == []


@pytest.mark.parametrize(
    "input_path,output_dir,expected",
    [
        ("dir", "explicit", "explicit"),
        ("dir", None, "dir"),
    ],
)
def test_determine_output_directory_directory(tmp_path: Path, input_path: str, output_dir: str | None, expected: str) -> None:
    base = tmp_path / "dir"
    base.mkdir()
    arg_output = tmp_path / output_dir if output_dir else None
    result = determine_output_directory(base, arg_output)
    assert result == (tmp_path / expected)


def test_determine_output_directory_file(tmp_path: Path) -> None:
    file_path = tmp_path / "f.pdf"
    file_path.write_text("x")
    result = determine_output_directory(file_path, None)
    assert result == tmp_path


def test_create_conversion_plan_output_paths(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_text("x")
    plan = create_conversion_plan(pdf, None, False, False)
    assert plan.output_dir == tmp_path
    assert len(plan.files) == 1
    assert plan.files[0].output_path == tmp_path / "doc.md"
    assert plan.files[0].action == "convert"


def test_create_conversion_plan_skip_existing(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_text("x")
    md = tmp_path / "doc.md"
    md.write_text("y")
    plan = create_conversion_plan(tmp_path, None, False, False)
    [action] = plan.files
    assert action.action == "skip"
    assert action.skip_reason == "already exists"


def test_process_and_save_pdf_custom_funcs(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_text("x")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    called: dict[str, Any] = {}

    def fake_process(client: DummyClient, path: Path) -> DummyResponse:
        called["process"] = path
        return DummyResponse()

    def fake_extract(resp: DummyResponse) -> str:
        called["extract"] = resp
        return "markdown"

    def fake_save(content: str, path: Path) -> None:
        called["save"] = (content, path)
        path.write_text(content)

    success, msg, doc = process_and_save_pdf(
        DummyClient(),
        pdf,
        out_dir,
        process_func=fake_process,
        extract_func=fake_extract,
        save_func=fake_save,
    )

    assert success
    assert doc
    assert doc.output_path.read_text() == "markdown"
    assert called["process"] == pdf
    assert called["save"][1] == out_dir / "doc.md"
