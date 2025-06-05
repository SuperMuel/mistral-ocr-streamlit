"""Streamlit web interface for Mistral OCR PDF to Markdown converter."""

import os

import streamlit as st
from dotenv import load_dotenv
from mistralai import Mistral, OCRResponse

from mistral_ocr.ocr_utils import (
    initialize_mistral_client,
    process_pdf_url,
    extract_markdown_from_response,
)

load_dotenv()


def process_uploaded_pdf(
    client: Mistral,
    file_content: bytes,
    file_name: str = "uploaded.pdf",
    include_image_base64: bool = False,
) -> OCRResponse | None:
    """Process uploaded PDF file and return OCR results.

    Args:
        client: Initialized Mistral client
        file_content: PDF file content as bytes
        file_name: Name of the uploaded file
        include_image_base64: Whether to include base64 encoded images

    Returns:
        OCR response or None if processing failed
    """
    try:
        # Upload the file
        uploaded_pdf = client.files.upload(
            file={
                "file_name": file_name,
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
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
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
    """Handle PDF file upload and processing.

    Args:
        client: Initialized Mistral client

    Returns:
        OCR response or None if no file uploaded or processing failed
    """
    uploaded_file = st.file_uploader(
        "Choose a PDF file", type="pdf", label_visibility="collapsed"
    )

    if not uploaded_file:
        return None

    if st.button(
        "Convert Uploaded File", use_container_width=True, key="convert_upload"
    ):
        with st.spinner("Processing uploaded PDF..."):
            file_content = uploaded_file.getvalue()
            return process_uploaded_pdf(client, file_content, uploaded_file.name)

    return None


def handle_url_input(client: Mistral) -> OCRResponse | None:
    """Handle PDF URL input and processing.

    Args:
        client: Initialized Mistral client

    Returns:
        OCR response or None if no URL provided or processing failed
    """
    pdf_url = st.text_input("Enter PDF URL", label_visibility="collapsed")

    if not pdf_url:
        return None

    if st.button("Convert URL", use_container_width=True, key="convert_url"):
        with st.spinner("Processing PDF from URL..."):
            try:
                return process_pdf_url(client, pdf_url)
            except Exception as e:
                st.error(f"Error processing URL: {e}")
                return None

    return None


def show_placeholder_message() -> None:
    """Show placeholder message when no processing has been initiated."""
    if (
        "convert_upload" not in st.session_state
        and "convert_url" not in st.session_state
    ):
        st.info("Upload a PDF or enter a URL and click Convert.")


def run_app() -> None:
    """Run the Streamlit application."""
    st.set_page_config(layout="wide")
    st.title("📄 Mistral PDF to Markdown Converter")

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
        st.subheader("⬆️ Upload PDF File")
        ocr_result = handle_file_upload(client) or ocr_result

    with col2:
        st.subheader("🔗 Enter PDF URL")
        ocr_result = handle_url_input(client) or ocr_result

    # Display results
    st.divider()
    if ocr_result:
        display_ocr_results(ocr_result)
    else:
        show_placeholder_message()


if __name__ == "__main__":
    run_app()
