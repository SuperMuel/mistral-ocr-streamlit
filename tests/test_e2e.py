"""End-to-end tests for the mistral-ocr CLI tool."""

import os
import tempfile
import subprocess
from pathlib import Path
from typing import Generator
import pytest
from dotenv import load_dotenv

from tests.test_utils import create_test_pdfs, create_single_test_pdf


load_dotenv()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def api_key() -> str:
    """Get API key from environment."""
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        pytest.skip("MISTRAL_API_KEY not found in environment")
    return key


class TestSingleFileProcessing:
    """Test processing single PDF files."""

    def test_single_file_conversion(self, temp_dir: Path, api_key: str) -> None:
        """Test converting a single PDF file."""
        pdf_path = temp_dir / "test_document.pdf"
        create_single_test_pdf(pdf_path, "This is a test document for OCR processing.")

        result = subprocess.run(
            ["uv", "run", "mistral-ocr", str(pdf_path), "--api-key", api_key],
            capture_output=True,
            text=True,
            cwd=temp_dir.parent,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        expected_md_path = temp_dir / "test_document.md"
        assert expected_md_path.exists(), f"Output file not created: {expected_md_path}"

        content = expected_md_path.read_text()
        assert "test document" in content.lower()

    def test_single_file_with_output_path(self, temp_dir: Path, api_key: str) -> None:
        """Test converting a single PDF with custom output path."""
        pdf_path = temp_dir / "input.pdf"
        output_path = temp_dir / "custom_output.md"
        create_single_test_pdf(pdf_path, "Custom output path test document.")

        result = subprocess.run(
            [
                "uv",
                "run",
                "mistral-ocr",
                str(pdf_path),
                "--output-dir",
                str(output_path.parent),
                "--api-key",
                api_key,
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        expected_output = temp_dir / "input.md"
        assert expected_output.exists(), f"Output file not created: {expected_output}"

        content = expected_output.read_text()
        assert "custom output" in content.lower()


class TestDirectoryProcessing:
    """Test processing directories with multiple PDF files."""

    def test_directory_conversion(self, temp_dir: Path, api_key: str) -> None:
        """Test converting all PDFs in a directory structure."""
        expected_content = create_test_pdfs(temp_dir)

        result = subprocess.run(
            ["uv", "run", "mistral-ocr", str(temp_dir), "--api-key", api_key],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # The CLI flattens directory structures, so check for files in the root output directory
        for pdf_path, expected_text in expected_content.items():
            filename = Path(pdf_path).name.replace(".pdf", ".md")
            md_path = temp_dir / filename
            assert md_path.exists(), f"Output file not created: {md_path}"

            content = md_path.read_text()
            assert any(
                word in content.lower() for word in expected_text.lower().split()[:3]
            )

    def test_directory_with_output_path(self, temp_dir: Path, api_key: str) -> None:
        """Test converting directory with custom output directory."""
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"

        expected_content = create_test_pdfs(input_dir)

        result = subprocess.run(
            [
                "uv",
                "run",
                "mistral-ocr",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--api-key",
                api_key,
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        for pdf_path in expected_content.keys():
            filename = Path(pdf_path).name.replace(".pdf", ".md")
            md_path = output_dir / filename
            assert md_path.exists(), f"Output file not created: {md_path}"


class TestDryRunMode:
    """Test dry run functionality."""

    def test_dry_run_single_file(self, temp_dir: Path, api_key: str) -> None:
        """Test dry run mode with single file."""
        pdf_path = temp_dir / "dry_run_test.pdf"
        create_single_test_pdf(pdf_path, "Dry run test document.")

        result = subprocess.run(
            [
                "uv",
                "run",
                "mistral-ocr",
                str(pdf_path),
                "--dry-run",
                "--api-key",
                api_key,
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert (
            "dry run" in result.stdout.lower()
            or "would process" in result.stdout.lower()
        )

        md_path = temp_dir / "dry_run_test.md"
        assert not md_path.exists(), "File was created during dry run"

    def test_dry_run_directory(self, temp_dir: Path, api_key: str) -> None:
        """Test dry run mode with directory."""
        expected_content = create_test_pdfs(temp_dir)

        result = subprocess.run(
            [
                "uv",
                "run",
                "mistral-ocr",
                str(temp_dir),
                "--dry-run",
                "--api-key",
                api_key,
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert (
            "dry run" in result.stdout.lower()
            or "would process" in result.stdout.lower()
        )

        for pdf_path in expected_content.keys():
            md_path = temp_dir / pdf_path.replace(".pdf", ".md")
            assert not md_path.exists(), f"File was created during dry run: {md_path}"


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_nonexistent_file(self, temp_dir: Path, api_key: str) -> None:
        """Test handling of nonexistent input file."""
        nonexistent_path = temp_dir / "nonexistent.pdf"

        result = subprocess.run(
            ["uv", "run", "mistral-ocr", str(nonexistent_path), "--api-key", api_key],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0, "CLI should fail with nonexistent file"
        assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()


class TestClipboardFeature:
    """Test clipboard functionality."""

    def test_clipboard_single_file(self, temp_dir: Path, api_key: str) -> None:
        """Test copying single file result to clipboard."""
        pdf_path = temp_dir / "clipboard_test.pdf"
        create_single_test_pdf(pdf_path, "Clipboard test document.")

        result = subprocess.run(
            [
                "uv",
                "run",
                "mistral-ocr",
                str(pdf_path),
                "--clipboard",
                "--api-key",
                api_key,
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert "clipboard" in result.stdout.lower()

    def test_clipboard_multiple_files(self, temp_dir: Path, api_key: str) -> None:
        """Test copying multiple files to clipboard with headers."""
        input_files = {
            "doc1.pdf": "First document content.",
            "doc2.pdf": "Second document content.",
        }

        for filename, content in input_files.items():
            pdf_path = temp_dir / filename
            create_single_test_pdf(pdf_path, content)

        result = subprocess.run(
            [
                "uv",
                "run",
                "mistral-ocr",
                str(temp_dir),
                "--clipboard",
                "--api-key",
                api_key,
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert "clipboard" in result.stdout.lower()
