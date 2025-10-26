"""Utility functions for OCR processing and file operations."""

import base64
import os
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse
from datetime import UTC, datetime
from typing import Final
from pydantic import BaseModel

from .cache_utils import Cache, CacheEntry, compute_file_hash

import pyperclip  # type: ignore[import-untyped]
from dotenv import load_dotenv
from mistralai import Mistral, OCRResponse

load_dotenv()


class ProcessedDocument(BaseModel):
    """Represents a successfully processed document."""

    filename: str
    content: str
    output_path: Path
    from_cache: bool = False


class DocumentKind(str, Enum):
    """Supported document types for OCR."""

    PDF = "pdf"
    IMAGE = "image"


IMAGE_MIME_TYPES: Final[dict[str, str]] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

SUPPORTED_SUFFIXES: Final[dict[str, DocumentKind]] = {
    ".pdf": DocumentKind.PDF,
    **{suffix: DocumentKind.IMAGE for suffix in IMAGE_MIME_TYPES},
}


def detect_document_kind_from_suffix(suffix: str) -> DocumentKind | None:
    """Return the document kind for a given file suffix."""

    return SUPPORTED_SUFFIXES.get(suffix.lower())


def detect_document_kind_from_path(path: Path) -> DocumentKind | None:
    """Determine document kind from a filesystem path."""

    return detect_document_kind_from_suffix(path.suffix)


def detect_document_kind_from_url(url: str) -> DocumentKind | None:
    """Determine document kind from a URL."""

    parsed = urlparse(url)
    return detect_document_kind_from_suffix(Path(parsed.path).suffix)


def initialize_mistral_client(api_key: str | None = None) -> Mistral | None:
    """Initialize and return Mistral client with given API key or from environment.

    Args:
        api_key: Optional API key. If None, will try to get from environment.

    Returns:
        Initialized Mistral client or None if initialization fails.
    """
    if not api_key:
        api_key = os.environ.get("MISTRAL_API_KEY")

    if not api_key:
        return None

    return Mistral(api_key=api_key)


def build_document_payload_for_url(url: str) -> dict[str, str]:
    """Build the document payload for a remote resource."""

    kind = detect_document_kind_from_url(url)
    if kind is DocumentKind.IMAGE:
        return {"type": "image_url", "image_url": url}
    return {"type": "document_url", "document_url": url}


def process_document_url(
    client: Mistral, url: str, include_image_base64: bool = False
) -> OCRResponse:
    """Process a remote document (PDF or image) and return OCR results."""

    return client.ocr.process(
        model="mistral-ocr-latest",
        document=build_document_payload_for_url(url),
        include_image_base64=include_image_base64,
    )


def _process_pdf_file(
    client: Mistral, file_path: Path, include_image_base64: bool
) -> OCRResponse:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as f:
        file_content = f.read()

    return _process_pdf_bytes(client, file_content, file_path.name, include_image_base64)


def _process_pdf_bytes(
    client: Mistral, file_content: bytes, file_name: str, include_image_base64: bool
) -> OCRResponse:
    uploaded_pdf = client.files.upload(
        file={
            "file_name": file_name,
            "content": file_content,
        },
        purpose="ocr",
    )

    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

    return client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        },
        include_image_base64=include_image_base64,
    )


def _process_image_file(
    client: Mistral, file_path: Path, include_image_base64: bool
) -> OCRResponse:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    mime_type = IMAGE_MIME_TYPES.get(file_path.suffix.lower())
    if not mime_type:
        raise ValueError(f"Unsupported image format: {file_path.suffix}")

    with open(file_path, "rb") as f:
        file_content = f.read()

    return _process_image_bytes(client, file_content, file_path.suffix, include_image_base64)


def _process_image_bytes(
    client: Mistral,
    file_content: bytes,
    suffix: str,
    include_image_base64: bool,
) -> OCRResponse:
    mime_type = IMAGE_MIME_TYPES.get(suffix.lower())
    if not mime_type:
        raise ValueError(f"Unsupported image format: {suffix}")

    base64_content = base64.b64encode(file_content).decode("utf-8")
    data_url = f"data:{mime_type};base64,{base64_content}"

    return client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "image_url",
            "image_url": data_url,
        },
        include_image_base64=include_image_base64,
    )


def process_document_file(
    client: Mistral,
    file_path: Path,
    include_image_base64: bool = False,
) -> OCRResponse:
    """Process a local PDF or image file and return OCR results."""

    kind = detect_document_kind_from_path(file_path)
    if kind is None:
        raise ValueError(f"Unsupported file type for OCR: {file_path.suffix}")

    if kind is DocumentKind.PDF:
        return _process_pdf_file(client, file_path, include_image_base64)
    return _process_image_file(client, file_path, include_image_base64)


def process_document_bytes(
    client: Mistral,
    file_content: bytes,
    file_name: str,
    include_image_base64: bool = False,
) -> OCRResponse:
    """Process in-memory document bytes and return OCR results."""

    suffix = Path(file_name).suffix
    kind = detect_document_kind_from_suffix(suffix)
    if kind is None:
        raise ValueError(f"Unsupported file type for OCR: {suffix}")

    if kind is DocumentKind.PDF:
        return _process_pdf_bytes(client, file_content, file_name, include_image_base64)
    return _process_image_bytes(client, file_content, suffix, include_image_base64)


def extract_markdown_from_response(ocr_response: OCRResponse) -> str:
    """Extract and combine markdown content from OCR response.

    Args:
        ocr_response: Response from Mistral OCR API

    Returns:
        Combined markdown content from all pages
    """
    return "\n\n".join(page.markdown for page in ocr_response.pages)


def save_markdown_to_file(content: str, output_path: Path) -> None:
    """Save markdown content to a file.

    Args:
        content: Markdown content to save
        output_path: Path where to save the file

    Raises:
        OSError: If file cannot be written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def process_and_save_document(
    client: Mistral,
    input_path: Path,
    output_dir: Path,
    force: bool = False,
    cache: Cache | None = None,
) -> tuple[bool, str, ProcessedDocument | None]:
    """Process a single document and save its markdown output."""

    try:
        output_path = output_dir / f"{input_path.stem}.md"

        if output_path.exists() and not force:
            return (
                False,
                f"Output file {output_path} already exists. Use --force to overwrite.",
                None,
            )

        document_hash = compute_file_hash(input_path)
        if cache:
            cached = cache.get(document_hash)
            if cached:
                save_markdown_to_file(cached.markdown_content, output_path)
                processed_doc = ProcessedDocument(
                    filename=input_path.name,
                    content=cached.markdown_content,
                    output_path=output_path,
                    from_cache=True,
                )
                return (
                    True,
                    f"{input_path} (cached)",
                    processed_doc,
                )

        ocr_response = process_document_file(client, input_path)
        markdown_content = extract_markdown_from_response(ocr_response)
        save_markdown_to_file(markdown_content, output_path)

        processed_doc = ProcessedDocument(
            filename=input_path.name,
            content=markdown_content,
            output_path=output_path,
            from_cache=False,
        )

        if cache:
            entry = CacheEntry(
                pdf_hash=document_hash,
                filename=input_path.name,
                source_path=str(input_path),
                size_bytes=input_path.stat().st_size,
                markdown_content=markdown_content,
                created_at=datetime.now(UTC).isoformat(),
                last_accessed=datetime.now(UTC).isoformat(),
                mistral_model="mistral-ocr-latest",
            )
            cache.set(entry)

        return (
            True,
            f"Successfully processed {input_path} -> {output_path}",
            processed_doc,
        )

    except Exception as e:
        return (False, f"Error processing {input_path}: {str(e)}", None)


def format_documents_for_clipboard(documents: list[ProcessedDocument]) -> str:
    """Format processed documents for clipboard copying.

    Args:
        documents: List of processed documents

    Returns:
        Formatted string ready for clipboard
    """
    if not documents:
        return ""

    if len(documents) == 1:
        return documents[0].content

    # Multiple documents - concatenate with headers and separators
    formatted_parts: list[str] = []
    for doc in documents:
        formatted_parts.append(f"# {doc.filename}\n\n{doc.content}")

    return "\n\n---\n\n".join(formatted_parts)


def copy_to_clipboard(content: str) -> tuple[bool, str]:
    """Copy content to system clipboard.

    Args:
        content: Content to copy to clipboard

    Returns:
        Tuple of (success, message)
    """
    try:
        pyperclip.copy(content)  # type: ignore[no-untyped-call]
        return True, "Successfully copied to clipboard"
    except Exception as e:
        return False, f"Failed to copy to clipboard: {str(e)}"


def handle_clipboard_operation(documents: list[ProcessedDocument]) -> str | None:
    """Handle clipboard operation for processed documents.

    Args:
        documents: List of successfully processed documents

    Returns:
        Status message or None if no documents to copy
    """
    if not documents:
        return None

    formatted_content = format_documents_for_clipboard(documents)
    success, message = copy_to_clipboard(formatted_content)

    if success:
        doc_count = len(documents)
        if doc_count == 1:
            return "Copied converted document to clipboard"
        else:
            return f"Copied {doc_count} converted documents to clipboard"
    else:
        return f"Warning: {message}"
