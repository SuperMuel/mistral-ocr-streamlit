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
    api_key = st.sidebar.text_input("Enter Mistral API Key", type="password")
    if not api_key:
        st.warning("Please enter a valid API key to proceed")
        return

    client = initialize_mistral_client(api_key)
    if not client:
        return

    # Input method selection
    input_method = st.radio("Choose input method:", ("Upload PDF", "Enter PDF URL"))

    if input_method == "Upload PDF":
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            if st.button("Convert"):
                with st.spinner("Processing PDF..."):
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = Path(tmp_file.name)

                    try:
                        ocr_response = process_pdf_file(client, tmp_path)
                        display_results(ocr_response)
                    finally:
                        # Clean up temporary file
                        tmp_path.unlink()

    else:  # Enter PDF URL
        pdf_url = st.text_input("Enter PDF URL")
        if pdf_url and st.button("Convert"):
            with st.spinner("Processing PDF..."):
                ocr_response = process_pdf_url(client, pdf_url)
                display_results(ocr_response)

    # Instructions
    st.sidebar.header("Instructions")
    st.sidebar.write("""
    1. Enter your Mistral API key
    2. Choose to either:
       - Upload a PDF file, or
       - Enter a PDF URL (e.g., https://arxiv.org/pdf/2201.04234)
    3. Click 'Convert' to process the PDF
    4. View the extracted markdown
    """)


if __name__ == "__main__":
    main()
