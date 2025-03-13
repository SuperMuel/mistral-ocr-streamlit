# Mistral OCR PDF to Markdown Converter

This project provides both a web interface and a CLI tool for converting PDF documents to Markdown format using the Mistral OCR API. It can process single files or entire directories of PDFs.

## Features

- Web interface for single file conversion
- CLI tool for bulk processing
- Support for both local PDF files and PDF URLs
- Progress tracking for bulk operations
- Markdown output with preserved formatting
- Overwrite protection with force option

## Installation
1. [Install UV](https://docs.astral.sh/uv/getting-started/installation/) 
2. Clone this repository
3. Install dependencies:
   ```bash
   uv sync
   ```

3. Set up your Mistral API key:
   - Create a `.env` file in the project root
   - Add your API key: `MISTRAL_API_KEY=your_api_key_here`
   - Or provide it via command line/web interface

## Usage

### Web Interface

Run the Streamlit app:
```bash
uv run streamlit run app.py
```

Then:
1. Enter your Mistral API key (if not in .env)
2. Choose to either:
   - Upload a PDF file
   - Enter a PDF URL
3. Click 'Convert' to process
4. View the extracted markdown

### CLI Tool

The CLI tool supports both single files and directories:

```bash
# Convert a single PDF file
uv run cli.py convert input.pdf

# Convert a directory of PDFs
uv run cli.py convert path/to/pdfs/

# Specify output directory
uv run cli.py convert input.pdf --output-dir path/to/output

# Force overwrite existing files
uv run cli.py convert input.pdf --force

# Get help
uv run cli.py convert --help
```

## Project Structure

- `app.py` - Streamlit web interface
- `cli.py` - Command line interface
- `ocr_utils.py` - Shared OCR functionality
