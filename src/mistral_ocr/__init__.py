from pathlib import Path
from typing import Annotated
import typer
from rich.console import Console
from tqdm import tqdm
from mistralai import Mistral

# Use relative import for modules within the same package
from .ocr_utils import initialize_mistral_client, process_and_save_pdf

app = typer.Typer(help="Convert PDF files to Markdown using Mistral OCR")
console = Console()


def process_path(
    client: Mistral,
    input_path: Path,
    output_dir: Path,
    force: bool,
    show_progress: bool = True,
) -> list[tuple[bool, str]]:
    """Process a single file or directory of PDFs"""
    # Client initialization is now handled before calling this function

    results: list[tuple[bool, str]] = []

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            console.print(f"[yellow]Skipping non-PDF file: {input_path}[/yellow]")
            return results
        results.append(process_and_save_pdf(client, input_path, output_dir, force))

    elif input_path.is_dir():
        pdf_files = list(input_path.glob("**/*.pdf"))

        if not pdf_files:
            console.print(f"[yellow]No PDF files found in {input_path}[/yellow]")
            return results

        iterator = pdf_files
        if show_progress:
            iterator = tqdm(pdf_files)

        for pdf_file in iterator:
            if show_progress:
                # Type check to satisfy mypy/pylint if tqdm is used
                if isinstance(iterator, tqdm):
                    iterator.set_description(f"Processing {pdf_file.name}")
            results.append(process_and_save_pdf(client, pdf_file, output_dir, force))

    return results


@app.command()
def convert(
    input_path: Annotated[
        Path, typer.Argument(help="Path to PDF file or directory containing PDFs")
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",  # Added short option -o
            help="Directory to save markdown files. Defaults to input directory or file's parent.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force", "-f", help="Overwrite existing markdown files"
        ),  # Added short option -f
    ] = False,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            help="Mistral API Key (overrides .env)",
            envvar="MISTRAL_API_KEY",  # Allow reading from env var too
        ),
    ] = None,
):
    """Convert PDF files to Markdown using Mistral OCR"""
    if not input_path.exists():
        console.print(f"[red]Error: Input path '{input_path}' does not exist.[/red]")
        raise typer.Exit(code=1)

    # Initialize Mistral client here, checking for API key
    client = initialize_mistral_client(api_key)
    if not client:
        console.print(
            "[red]Error: Mistral API key is required. Provide via --api-key option or MISTRAL_API_KEY environment variable or .env file.[/red]"
        )
        raise typer.Exit(code=1)

    # Determine output directory
    resolved_output_dir: Path
    if output_dir is not None:
        resolved_output_dir = output_dir
    elif input_path.is_dir():
        resolved_output_dir = input_path
    else:  # input_path is a file
        resolved_output_dir = input_path.parent

    # Ensure output directory exists
    try:
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        console.print(
            f"[red]Error creating output directory '{resolved_output_dir}': {e}[/red]"
        )
        raise typer.Exit(code=1)

    console.print(f"Input: {input_path.resolve()}")
    console.print(f"Output Directory: {resolved_output_dir.resolve()}")
    if force:
        console.print(
            "[yellow]Force mode enabled: Existing files will be overwritten.[/yellow]"
        )

    try:
        results = process_path(client, input_path, resolved_output_dir, force)

        # Print summary
        success_count = sum(1 for success, _ in results if success)
        fail_count = len(results) - success_count

        console.print("\n[bold green]Processing Complete[/bold green]")
        console.print(f"  Successfully processed: [green]{success_count}[/green]")
        if fail_count > 0:
            console.print(f"  Failed / Skipped: [yellow]{fail_count}[/yellow]")

        # Print detailed messages for failures/skips
        if fail_count > 0:
            console.print("\n[bold yellow]Details:[/bold yellow]")
            for success, message in results:
                if not success:
                    # Check if it was a skip message or an error message
                    if "Skipping" in message or "already exists" in message:
                        console.print(f"[yellow]• {message}[/yellow]")
                    else:
                        console.print(f"[red]• {message}[/red]")

    except Exception as e:
        console.print(f"\n[bold red]An unexpected error occurred:[/bold red] {str(e)}")
        raise typer.Exit(code=1)


# This main function is the entry point defined in pyproject.toml
def main():
    app()


# No need for if __name__ == "__main__": here
