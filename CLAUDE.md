# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development
- `uv run mistral-ocr --help` - Run CLI tool without global install
- `uv tool install . --force` - Install CLI tool globally
- `uv run streamlit run src/mistral_ocr/app.py` - Run Streamlit web interface

### Testing and Quality
- `uv run pytest` - Run tests
- `uv run ruff check` - Lint code
- `uv run ruff check --fix` - Fix linting issues
- `uv run coverage run -m pytest && coverage report` - Run tests with coverage

## Architecture

This is a dual-interface OCR tool that converts PDFs to Markdown using the Mistral API:

### Core Components
- **CLI Interface** (`src/mistral_ocr/__init__.py`): Typer-based command-line tool with file/directory processing, progress bars, dry-run mode, and clipboard integration
- **Streamlit Web App** (`src/mistral_ocr/app.py`): Web interface for uploading files or processing URLs
- **OCR Utilities** (`src/mistral_ocr/ocr_utils.py`): Shared functions for Mistral API interaction, file processing, and clipboard operations

### Key Patterns
- All PDF processing goes through Mistral's file upload → signed URL → OCR API flow
- CLI supports both single files and directory recursion with `**/*.pdf` globbing
- Clipboard functionality formats single documents as raw content, multiple documents with headers and separators
- Error handling preserves partial results and provides detailed failure messages
- Output path resolution: single files go to parent directory, directories preserve subfolder structure

### Configuration
- API key sources (priority order): `--api-key` flag, `MISTRAL_API_KEY` env var, `.env` file
- Uses `uv` for dependency management and packaging
- Entry point: `mistral-ocr = "mistral_ocr:main"` in pyproject.toml

### Rules
Rules and best practices to work on this project are writtin in `AGENTS.MD`
