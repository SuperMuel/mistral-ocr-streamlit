import os
import streamlit as st
from mistralai import Mistral, OCRResponse
from dotenv import load_dotenv
from ocr_utils import initialize_mistral_client, process_pdf_url, process_pdf_file
from pathlib import Path
import tempfile

load_dotenv()


@st.cache_resource
def _initialize_mistral_client(api_key: str | None = None) -> Mistral | None:
    return initialize_mistral_client(api_key)


def process_uploaded_pdf(
    client: Mistral,
    file_content: bytes,
    include_image_base64: bool = False,
) -> OCRResponse:
    """Process uploaded PDF file and return OCR results"""
    # Upload the file
    uploaded_pdf = client.files.upload(
        file={
            "file_name": "temp.pdf",
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


def display_results(ocr_response) -> None:
    """Display OCR results in the app"""
    text = "\n\n".join(page.markdown for page in ocr_response.pages)
    st.subheader("Extracted Markdown")
    st.code(text)


def main():
    st.title("PDF to Markdown Converter")
    st.write("Convert PDFs to markdown using Mistral OCR API")

    # API Key handling
    if not (api_key := os.environ.get("MISTRAL_API_KEY")):
        api_key = st.sidebar.text_input(
            "Enter Mistral API Key", type="password", key="mistral_api_key"
        )
        if not api_key:
            st.warning("Please enter a valid API key to proceed")
            return

    print(api_key)
    client = initialize_mistral_client(api_key)
    if not client:
        return

    col1, col2 = st.columns(2)

    ocr_result = None  # Store the OCR result to display after columns

    with col1:
        st.subheader("🔗 Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            if st.button("Convert File", use_container_width=True):
                with st.spinner("Processing PDF..."):
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = Path(tmp_file.name)

                    try:
                        ocr_result = process_pdf_file(client, tmp_path)
                    finally:
                        # Clean up temporary file
                        tmp_path.unlink()

    with col2:
        st.subheader("📄 Enter PDF URL")
        pdf_url = st.text_input("Enter PDF URL")
        if pdf_url and st.button("Convert URL", use_container_width=True):
            with st.spinner("Processing PDF..."):
                ocr_result = process_pdf_url(client, pdf_url)

    # Display results after both columns if we have any
    if ocr_result:
        display_results(ocr_result)


if __name__ == "__main__":
    main()
