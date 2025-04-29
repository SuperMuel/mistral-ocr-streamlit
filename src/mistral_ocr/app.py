import os
from mistralai import Mistral, OCRResponse
import streamlit as st

from dotenv import load_dotenv

from ocr_utils import initialize_mistral_client, process_pdf_url

load_dotenv()


# Removed @st.cache_resource as client initialization might depend on sidebar input
def _initialize_mistral_client(
    api_key: str | None = None,
) -> Mistral | None:  # Forward reference Mistral type hint
    # Use the shared utility function
    return initialize_mistral_client(api_key)


# Keep process_uploaded_pdf as it handles the upload/signed URL logic specifically for Streamlit
def process_uploaded_pdf(
    client: "Mistral",  # Forward reference Mistral type hint
    file_content: bytes,
    file_name: str = "uploaded.pdf",  # Use actual filename if available
    include_image_base64: bool = False,
) -> (
    "OCRResponse | None"
):  # Forward reference OCRResponse type hint, return None on error
    """Process uploaded PDF file and return OCR results"""
    try:
        # Upload the file
        uploaded_pdf = client.files.upload(
            file={
                "file_name": file_name,
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
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
        return None


def display_results(ocr_response) -> None:
    """Display OCR results in the app"""
    if ocr_response and ocr_response.pages:
        text = "\n\n".join(page.markdown for page in ocr_response.pages)
        st.subheader("Extracted Markdown")
        # Use st.markdown for better rendering, or st.code for raw block
        st.markdown(f"```markdown\n{text}\n```")
        # Or: st.code(text, language='markdown')
    else:
        st.info("No text extracted or an error occurred.")


def run_app():  # Renamed from main to avoid confusion with __init__.py's main
    st.set_page_config(layout="wide")  # Optional: Use wider layout
    st.title("📄 Mistral PDF to Markdown Converter")
    # st.write("Convert PDFs to markdown using Mistral OCR API")

    # --- API Key Handling ---
    st.sidebar.header("Configuration")
    # Try getting key from environment first
    env_api_key = os.environ.get("MISTRAL_API_KEY")
    api_key_input = st.sidebar.text_input(
        "Mistral API Key",
        type="password",
        key="mistral_api_key_input",
        value=env_api_key or "",  # Pre-fill if found in env
        help="Required. Get yours from Mistral AI.",
    )

    client = None
    if api_key_input:
        client = _initialize_mistral_client(api_key_input)
    elif env_api_key:
        # Attempt init with env key if input is cleared but env exists
        client = _initialize_mistral_client(env_api_key)

    if not client:
        st.warning("Please enter a valid Mistral API key in the sidebar to proceed.")
        st.stop()  # Stop execution if no valid client

    # --- Main UI ---
    col1, col2 = st.columns(2)
    ocr_result = None  # Store the OCR result to display after columns

    with col1:
        st.subheader("⬆️ Upload PDF File")
        uploaded_file = st.file_uploader(
            "Choose a PDF file", type="pdf", label_visibility="collapsed"
        )
        if uploaded_file is not None:
            if st.button(
                "Convert Uploaded File", use_container_width=True, key="convert_upload"
            ):
                with st.spinner("Processing uploaded PDF..."):
                    file_content = uploaded_file.getvalue()
                    ocr_result = process_uploaded_pdf(
                        client, file_content, uploaded_file.name
                    )

    with col2:
        st.subheader("🔗 Enter PDF URL")
        pdf_url = st.text_input("Enter PDF URL", label_visibility="collapsed")
        if pdf_url:
            if st.button("Convert URL", use_container_width=True, key="convert_url"):
                with st.spinner("Processing PDF from URL..."):
                    try:
                        ocr_result = process_pdf_url(client, pdf_url)
                    except Exception as e:
                        st.error(f"Error processing URL: {e}")

    # --- Display Results ---
    st.divider()
    if ocr_result:
        display_results(ocr_result)
    else:
        # Show placeholder if no button has been clicked yet
        if (
            "convert_upload" not in st.session_state
            and "convert_url" not in st.session_state
        ):
            st.info("Upload a PDF or enter a URL and click Convert.")


# This allows running the Streamlit app via `python -m mistral_ocr.app`
if __name__ == "__main__":
    run_app()
