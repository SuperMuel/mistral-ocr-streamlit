"""CLI interface for Mistral OCR PDF to Markdown converter."""

from pathlib import Path
from typing import Annotated, List, Optional, Tuple

import typer
from rich.console import Console
from tqdm import tqdm

from .ocr_utils import (
    initialize_mistral_client,
    process_and_save_pdf,
    handle_clipboard_operation,
    ProcessedDocument,
)

app = typer.Typer(help="Convert PDF files to Markdown using Mistral OCR")
console = Console()


def find_pdf_files(input_path: Path) -> List[Path]:
    """Find all PDF files in the given path.

    Args:
        input_path: Path to search for PDF files

    Returns:
        List of PDF file paths
    """
    if input_path.is_file():
        if input_path.suffix.lower() == ".pdf":
            return [input_path]
        else:
            console.print(f"[yellow]Warning: {input_path} is not a PDF file[/yellow]")
            return []

    if input_path.is_dir():
        return list(input_path.glob("**/*.pdf"))

    return []


def process_pdf_files(
    client,
    pdf_files: List[Path],
    output_dir: Path,
    force: bool,
    show_progress: bool = True,
) -> List[Tuple[bool, str, ProcessedDocument]]:
    """Process multiple PDF files.

    Args:
        client: Initialized Mistral client
        pdf_files: List of PDF files to process
        output_dir: Output directory for markdown files
        force: Whether to overwrite existing files
        show_progress: Whether to show progress bar

    Returns:
        List of processing results
    """
    if not pdf_files:
        console.print("[yellow]No PDF files found to process[/yellow]")
        return []

    results = []
    iterator = pdf_files

    if show_progress and len(pdf_files) > 1:
        iterator = tqdm(pdf_files, desc="Processing PDFs")

    for pdf_file in iterator:
        if show_progress and len(pdf_files) > 1:
            if isinstance(iterator, tqdm):
                iterator.set_description(f"Processing {pdf_file.name}")  # type: ignore

        result = process_and_save_pdf(client, pdf_file, output_dir, force)
        results.append(result)

    return results


def print_processing_summary(
    results: List[Tuple[bool, str, ProcessedDocument]],
) -> None:
    """Print summary of processing results.

    Args:
        results: List of processing results
    """
    success_count = sum(1 for success, _, _ in results if success)
    fail_count = len(results) - success_count

    console.print("\n[bold green]Processing Complete[/bold green]")
    console.print(f"  Successfully processed: [green]{success_count}[/green]")

    if fail_count > 0:
        console.print(f"  Failed / Skipped: [yellow]{fail_count}[/yellow]")
        console.print("\n[bold yellow]Details:[/bold yellow]")

        for success, message, _ in results:
            if not success:
                if "Skipping" in message or "already exists" in message:
                    console.print(f"[yellow]• {message}[/yellow]")
                else:
                    console.print(f"[red]• {message}[/red]")


def determine_output_directory(
    input_path: Path, output_dir: Optional[Path] = None
) -> Path:
    """Determine the output directory for processed files.

    Args:
        input_path: Input path (file or directory)
        output_dir: Explicitly specified output directory

    Returns:
        Resolved output directory path
    """
    if output_dir is not None:
        return output_dir
    elif input_path.is_dir():
        return input_path
    else:  # input_path is a file
        return input_path.parent


@app.command()
def convert(
    input_path: Annotated[
        Path, typer.Argument(help="Path to PDF file or directory containing PDFs")
    ],
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to save markdown files. Defaults to input directory or file's parent.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing markdown files"),
    ] = False,
    clipboard: Annotated[
        bool,
        typer.Option("--clipboard", "-c", help="Copy converted content to clipboard"),
    ] = False,
    api_key: Annotated[
        Optional[str],
        typer.Option(
            "--api-key",
            help="Mistral API Key (overrides .env)",
            envvar="MISTRAL_API_KEY",
        ),
    ] = None,
):
    """Convert PDF files to Markdown using Mistral OCR."""
    # Validate input path
    if not input_path.exists():
        console.print(f"[red]Error: Input path '{input_path}' does not exist.[/red]")
        raise typer.Exit(code=1)

    # Initialize Mistral client
    client = initialize_mistral_client(api_key)
    if not client:
        console.print(
            "[red]Error: Mistral API key is required. "
            "Provide via --api-key option, MISTRAL_API_KEY environment variable, or .env file.[/red]"
        )
        raise typer.Exit(code=1)

    # Determine output directory and create if needed
    resolved_output_dir = determine_output_directory(input_path, output_dir)
    try:
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        console.print(
            f"[red]Error creating output directory '{resolved_output_dir}': {e}[/red]"
        )
        raise typer.Exit(code=1)

    # Print configuration
    console.print(f"Input: {input_path.resolve()}")
    console.print(f"Output Directory: {resolved_output_dir.resolve()}")
    if force:
        console.print(
            "[yellow]Force mode enabled: Existing files will be overwritten.[/yellow]"
        )
    if clipboard:
        console.print(
            "[blue]Clipboard mode enabled: Content will be copied to clipboard.[/blue]"
        )

    try:
        # Find and process PDF files
        pdf_files = find_pdf_files(input_path)
        if not pdf_files:
            console.print(f"[yellow]No PDF files found in {input_path}[/yellow]")
            raise typer.Exit(code=0)

        results = process_pdf_files(client, pdf_files, resolved_output_dir, force)

        # Print processing summary
        print_processing_summary(results)

        # Handle clipboard operation if requested
        if clipboard:
            successful_documents = [
                doc for success, _, doc in results if success and doc
            ]
            clipboard_message = handle_clipboard_operation(successful_documents)
            if clipboard_message:
                if clipboard_message.startswith("Warning:"):
                    console.print(f"[yellow]{clipboard_message}[/yellow]")
                else:
                    console.print(f"[green]{clipboard_message}[/green]")

    except Exception as e:
        console.print(f"\n[bold red]An unexpected error occurred:[/bold red] {str(e)}")
        raise typer.Exit(code=1)


def main():
    """Entry point for the CLI application."""
    app()
