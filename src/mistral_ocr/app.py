"""Streamlit web interface for Mistral OCR document to Markdown converter."""

import os

import streamlit as st
from dotenv import load_dotenv
from mistralai import Mistral, OCRResponse

from mistral_ocr.ocr_utils import (
    extract_markdown_from_response,
    initialize_mistral_client,
    process_document_bytes,
    process_document_url,
)

load_dotenv()


def process_uploaded_document(
    client: Mistral,
    file_content: bytes,
    file_name: str,
    include_image_base64: bool = False,
) -> OCRResponse | None:
    """Process an uploaded PDF or image and return OCR results."""

    try:
        return process_document_bytes(
            client,
            file_content,
            file_name,
            include_image_base64=include_image_base64,
        )
    except ValueError as exc:
        st.error(str(exc))
        return None
    except Exception as exc:
        st.error(f"Error processing uploaded file: {exc}")
        return None


def display_ocr_results(ocr_response: OCRResponse) -> None:
    """Display OCR results in the Streamlit app.

    Args:
        ocr_response: OCR response from Mistral API
    """
    if not ocr_response or not ocr_response.pages:
        st.info("No text extracted or an error occurred.")
        return

    markdown_content = extract_markdown_from_response(ocr_response)
    st.subheader("Extracted Markdown")
    st.markdown(f"```markdown\n{markdown_content}\n```")


def get_api_key() -> str | None:
    """Get API key from sidebar input or environment.

    Returns:
        API key or None if not available
    """
    env_api_key = os.environ.get("MISTRAL_API_KEY")

    return st.sidebar.text_input(
        "Mistral API Key",
        type="password",
        key="mistral_api_key_input",
        value=env_api_key or "",
        help="Required. Get yours from Mistral AI.",
    )


def initialize_client_with_key(api_key: str) -> Mistral | None:
    """Initialize Mistral client with the provided API key.

    Args:
        api_key: API key for Mistral client

    Returns:
        Initialized client or None if initialization failed
    """
    if not api_key:
        return None
    return initialize_mistral_client(api_key)


def handle_file_upload(client: Mistral) -> OCRResponse | None:
    """Handle document upload and processing."""

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
        label_visibility="collapsed",
    )

    if not uploaded_file:
        return None

    if st.button(
        "Convert Uploaded File", use_container_width=True, key="convert_upload"
    ):
        with st.spinner("Processing uploaded document..."):
            file_content = uploaded_file.getvalue()
            return process_uploaded_document(client, file_content, uploaded_file.name)

    return None


def handle_url_input(client: Mistral) -> OCRResponse | None:
    """Handle URL input and processing for PDFs or images."""

    document_url = st.text_input(
        "Enter document URL", label_visibility="collapsed"
    )

    if not document_url:
        return None

    if st.button("Convert URL", use_container_width=True, key="convert_url"):
        with st.spinner("Processing document from URL..."):
            try:
                return process_document_url(client, document_url)
            except Exception as exc:
                st.error(f"Error processing URL: {exc}")
                return None

    return None


def show_placeholder_message() -> None:
    """Show placeholder message when no processing has been initiated."""
    if (
        "convert_upload" not in st.session_state
        and "convert_url" not in st.session_state
    ):
        st.info("Upload a document or enter a URL and click Convert.")


def run_app() -> None:
    """Run the Streamlit application."""
    st.set_page_config(layout="wide")
    st.title("📄 Mistral Document to Markdown Converter")

    # Configuration sidebar
    st.sidebar.header("Configuration")
    api_key = get_api_key()
    if not api_key:
        st.warning("Please enter a valid Mistral API key in the sidebar to proceed.")
        st.stop()

    assert api_key is not None

    # Initialize client
    client = initialize_client_with_key(api_key)
    if not client:
        st.warning("Please enter a valid Mistral API key in the sidebar to proceed.")
        st.stop()

    assert client is not None

    # Main interface
    col1, col2 = st.columns(2)
    ocr_result = None

    with col1:
        st.subheader("⬆️ Upload Document")
        ocr_result = handle_file_upload(client) or ocr_result

    with col2:
        st.subheader("🔗 Enter Document URL")
        ocr_result = handle_url_input(client) or ocr_result

    # Display results
    st.divider()
    if ocr_result:
        display_ocr_results(ocr_result)
    else:
        show_placeholder_message()


if __name__ == "__main__":
    run_app()
