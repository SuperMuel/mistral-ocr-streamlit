"""Utility functions for OCR processing and file operations."""

import os
from pathlib import Path
from datetime import UTC, datetime
from pydantic import BaseModel

from .cache_utils import Cache, CacheEntry, compute_pdf_hash

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


def process_pdf_url(
    client: Mistral, url: str, include_image_base64: bool = False
) -> OCRResponse:
    """Process PDF from URL and return OCR results.

    Args:
        client: Initialized Mistral client
        url: URL of the PDF to process
        include_image_base64: Whether to include base64 encoded images

    Returns:
        OCR response from Mistral API

    Raises:
        Exception: If processing fails
    """
    return client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "document_url", "document_url": url},
        include_image_base64=include_image_base64,
    )


def process_pdf_file(
    client: Mistral,
    file_path: Path,
    include_image_base64: bool = False,
) -> OCRResponse:
    """Process PDF file and return OCR results.

    Args:
        client: Initialized Mistral client
        file_path: Path to the PDF file
        include_image_base64: Whether to include base64 encoded images

    Returns:
        OCR response from Mistral API

    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If processing fails
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read file content
    with open(file_path, "rb") as f:
        file_content = f.read()

    # Upload the file
    uploaded_pdf = client.files.upload(
        file={
            "file_name": file_path.name,
            "content": file_content,
        },
        purpose="ocr",
    )

    # Get signed URL
    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

    # Process OCR
    return client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        },
        include_image_base64=include_image_base64,
    )


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


def process_and_save_pdf(
    client: Mistral,
    input_path: Path,
    output_dir: Path,
    force: bool = False,
    cache: Cache | None = None,
) -> tuple[bool, str, ProcessedDocument | None]:
    """Process a single PDF file and save its markdown output.

    Args:
        client: Initialized Mistral client
        input_path: Path to the PDF file
        output_dir: Directory to save the markdown file
        force: Whether to overwrite existing files

    Returns:
        Tuple of (success, message, processed_document)
        processed_document is None if processing failed
    """
    try:
        output_path = output_dir / f"{input_path.stem}.md"

        if output_path.exists() and not force:
            return (
                False,
                f"Output file {output_path} already exists. Use --force to overwrite.",
                None,
            )

        pdf_hash = compute_pdf_hash(input_path)
        if cache:
            cached = cache.get(pdf_hash)
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

        ocr_response = process_pdf_file(client, input_path)
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
                pdf_hash=pdf_hash,
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
