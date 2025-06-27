# Mistral OCR PDF to Markdown Converter

This project provides both a CLI tool and a Streamlit web interface for converting PDF documents to Markdown format using the Mistral OCR API. It can process single files or entire directories of PDFs.

## Features

- **CLI Tool:**
    - Process single PDF files or directories.
    - Specify output directory.
    - Option to force overwrite existing markdown files.
    - Dry run mode to preview which files will be converted or overwritten.
    - Progress bar for directory processing.
    - Local cache to avoid re-processing identical PDFs (disable with `--no-cache`).
    - Read API key from `.env`, environment variable (`MISTRAL_API_KEY`), or command-line option.
    - Copy extracted markdown to clipboard
- **Web Interface (Streamlit):**
    - Upload local PDF files.
    - Process PDFs from URLs.
    - Enter API key directly in the interface (also reads from `.env`/environment).
    - View extracted markdown.

## Installation

1.  **Install `uv`** (if you haven't already):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make sure uv's bin directory is in your PATH
    # source $HOME/.cargo/env  (or follow uv's post-install instructions)
    ```

2.  **Clone this repository:**
    ```bash
    git clone <your-repo-url> mistral-ocr
    cd mistral-ocr
    ```

3.  **Set up your Mistral API key:**
    *   Create a `.env` file in the project root (`mistral-ocr/.env`) OR set an environment variable:
      ```
      # Option 1: .env file
      MISTRAL_API_KEY=your_api_key_here

      # Option 2: Environment variable
      export MISTRAL_API_KEY=your_api_key_here
      ```
    *   The CLI tool also accepts the key via `--api-key`.
    *   The Streamlit app allows entering the key in the sidebar.

4.  **(Optional) Install the CLI tool globally:**
    To run the `mistral-ocr` command from anywhere without activating the virtual environment:
    ```bash
    uv tool install . --force # Use --force to overwrite if previously installed
    # Ensure uv's tool directory is in your PATH
    uv tool update-shell
    # Restart your shell or source your profile (e.g., source ~/.zshrc)
    ```
    *Note: Using `-e` (editable) install with `uv tool install` might be less common for tools compared to libraries, but possible if you want changes reflected immediately without reinstalling.*

## Usage

### CLI Tool

*Make sure the tool is installed (`uv tool install .`) OR you have activated the virtual environment (`source .venv/bin/activate` or use `uv run`).*

```bash
# If installed globally:
mistral-ocr --help

# If using uv run (without global install):
uv run mistral-ocr --help

# --- Examples ---

# Convert a single PDF file (output to same directory)
mistral-ocr path/to/your/document.pdf

# Convert a directory of PDFs (output to same directory)
mistral-ocr path/to/pdf_folder/

# Specify output directory
mistral-ocr path/to/your/document.pdf -o path/to/output

# Force overwrite existing markdown files
mistral-ocr path/to/your/document.pdf -f

# Preview files without converting
mistral-ocr path/to/your/document.pdf --dry-run

# Provide API key via command line (overrides .env/env var)
mistral-ocr path/to/document.pdf --api-key sk-yourkeyhere


```

### Streamlit Web Interface

Start the Streamlit app:

```bash
uv run streamlit run src/mistral_ocr/app.py
```

Navigate to the URL shown in the terminal (usually `http://localhost:8501`).

## Development

### Setup Development Environment

1. **Clone and install with development dependencies:**
   ```bash
   git clone https://github.com/SuperMuel/mistral-ocr-streamlit mistral-ocr
   cd mistral-ocr
   uv sync
   ```

### Testing

The project uses `pytest` with two categories of tests:

- **Unit Tests**: Fast tests that don't require external API calls
- **End-to-End (E2E) Tests**: Integration tests that require a valid Mistral API key

#### Running Tests

```bash
# Run only unit tests (default behavior)
uv run pytest

# Run only e2e tests
uv run pytest -m e2e
```

### Code Quality

The project uses `ruff` for linting and formatting:

```bash
# Check code quality
uv run ruff check

# Auto-fix issues
uv run ruff check --fix

# Format code
uv run ruff format
```