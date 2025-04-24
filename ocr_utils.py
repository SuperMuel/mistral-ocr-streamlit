from pathlib import Path
from mistralai import Mistral, OCRResponse
import os
from dotenv import load_dotenv

load_dotenv()


def initialize_mistral_client(api_key: str | None = None) -> Mistral | None:
    """Initialize and return Mistral client with given API key or from environment"""
    if not api_key and not (api_key := os.environ.get("MISTRAL_API_KEY")):
        raise ValueError(
            "Mistral API key not manually provided and not found in environment"
        )

    return Mistral(api_key=api_key)


def process_pdf_url(
    client: Mistral, url: str, include_image_base64: bool = False
) -> OCRResponse:
    """Process PDF from URL and return OCR results"""
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "document_url", "document_url": url},
        include_image_base64=include_image_base64,
    )
    return ocr_response


def process_pdf_file(
    client: Mistral,
    file_path: str | Path,
    include_image_base64: bool = False,
) -> OCRResponse:
    """Process PDF file and return OCR results"""
    file_path = Path(file_path)
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
        purpose="ocr",  # type: ignore
    )

    # Get signed URL
    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

    # Process OCR
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        },
        include_image_base64=include_image_base64,
    )
    return ocr_response


def save_markdown(markdown: str, output_path: Path) -> None:
    """Save markdown content to a file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown)


def process_and_save_pdf(
    client: Mistral, input_path: Path, output_dir: Path, force: bool = False
) -> tuple[bool, str]:
    """
    Process a single PDF file and save its markdown output.
    Returns (success, message) tuple.
    """
    try:
        output_path = output_dir / f"{input_path.stem}.md"

        if output_path.exists() and not force:
            return (
                False,
                f"Output file {output_path} already exists. Use --force to overwrite.",
            )

        ocr_response = process_pdf_file(client, input_path)

        # Combine markdown from all pages
        markdown = "\n\n".join(page.markdown for page in ocr_response.pages)

        # Save the markdown
        save_markdown(markdown, output_path)
        return True, f"Successfully processed {input_path} -> {output_path}"

    except Exception as e:
        return False, f"Error processing {input_path}: {str(e)}"
