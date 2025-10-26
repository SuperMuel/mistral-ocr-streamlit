from pathlib import Path

from mistral_ocr import (
    create_conversion_plan_pure,
    find_supported_documents_pure,
    is_supported_document,
)


def test_is_supported_document_accepts_pdf_and_image(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"")
    image = tmp_path / "image.png"
    image.write_bytes(b"")

    assert is_supported_document(pdf) is True
    assert is_supported_document(image) is True


def test_find_supported_documents_in_directory(tmp_path: Path) -> None:
    pdf = tmp_path / "docs" / "a.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"")
    image = tmp_path / "docs" / "b.jpeg"
    image.write_bytes(b"")
    other = tmp_path / "docs" / "notes.txt"
    other.write_text("ignore")

    documents, warnings = find_supported_documents_pure(tmp_path / "docs")

    assert set(documents) == {pdf, image}
    assert warnings == []


def test_find_supported_documents_warns_on_single_file(tmp_path: Path) -> None:
    unsupported = tmp_path / "report.txt"
    unsupported.write_text("unsupported")

    documents, warnings = find_supported_documents_pure(unsupported)

    assert documents == []
    assert warnings == [f"Warning: {unsupported} is not a supported document"]


def test_create_conversion_plan_pure_uses_documents(tmp_path: Path) -> None:
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"")
    image = tmp_path / "b.png"
    image.write_bytes(b"")
    output_dir = tmp_path / "out"

    plan = create_conversion_plan_pure(
        document_paths=[pdf, image],
        input_path=tmp_path,
        output_dir=output_dir,
        force=False,
        clipboard=False,
    )

    assert [action.input_path for action in plan.files] == [pdf, image]
