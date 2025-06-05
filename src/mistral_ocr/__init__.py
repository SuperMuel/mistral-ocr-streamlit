"""CLI interface for Mistral OCR PDF to Markdown converter."""

from pathlib import Path
from typing import Annotated, Literal

from mistralai import Mistral
from pydantic import BaseModel

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


class FileAction(BaseModel):
    """Represents a planned action for a single PDF file.

    Attributes:
        input_path: The PDF file to process
        output_path: Where the markdown output will be saved
        action: Whether to convert the file or skip it
        will_overwrite: True if converting will overwrite an existing file
        skip_reason: Human-readable reason for skipping (only when action="skip")
    """

    input_path: Path
    output_path: Path
    action: Literal["convert", "skip"]
    will_overwrite: bool = False
    skip_reason: str | None = None

    def __str__(self) -> str:
        """Human-readable description of this action."""
        if self.action == "convert":
            suffix = " (will overwrite)" if self.will_overwrite else ""
            return f"Convert: {self.input_path.name} → {self.output_path.name}{suffix}"
        else:
            return f"Skip: {self.input_path.name} ({self.skip_reason})"

    @property
    def is_converting(self) -> bool:
        """True if this action will convert a file."""
        return self.action == "convert"

    @property
    def is_skipping(self) -> bool:
        """True if this action will skip a file."""
        return self.action == "skip"


class ConversionPlan(BaseModel):
    """Plan describing how files should be processed."""

    files: list[FileAction]
    output_dir: Path
    clipboard: bool = False
    force: bool = False


def find_pdf_files(input_path: Path) -> list[Path]:
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
    client: Mistral,
    pdf_files: list[Path],
    output_dir: Path,
    force: bool,
    show_progress: bool = True,
) -> list[tuple[bool, str, ProcessedDocument]]:
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
    results: list[tuple[bool, str, ProcessedDocument]],
) -> None:
    """Print summary of processing results.

    Args:
        results: List of processing results
    """
    success_count = sum(1 for success, _, _ in results if success)
    fail_count = len(results) - success_count

    console.print("\n[bold green]Processing Complete[/bold green]")
    if success_count > 1:
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
    input_path: Path, output_dir: Path | None = None
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


def create_conversion_plan(
    input_path: Path,
    output_dir: Path | None,
    force: bool,
    clipboard: bool,
) -> ConversionPlan:
    """Create a plan describing what actions will be taken."""

    resolved_output_dir = determine_output_directory(input_path, output_dir)
    pdf_files = find_pdf_files(input_path)

    actions: list[FileAction] = []
    for pdf_file in pdf_files:
        output_path = resolved_output_dir / f"{pdf_file.stem}.md"
        if output_path.exists():
            if force:
                actions.append(
                    FileAction(
                        input_path=pdf_file,
                        output_path=output_path,
                        action="convert",
                        will_overwrite=True,
                    )
                )
            else:
                actions.append(
                    FileAction(
                        input_path=pdf_file,
                        output_path=output_path,
                        action="skip",
                        will_overwrite=False,
                        skip_reason="already exists",
                    )
                )
        else:
            actions.append(
                FileAction(
                    input_path=pdf_file,
                    output_path=output_path,
                    action="convert",
                    will_overwrite=False,
                )
            )

    return ConversionPlan(
        files=actions,
        output_dir=resolved_output_dir,
        clipboard=clipboard,
        force=force,
    )


@app.command()
def convert(
    input_path: Annotated[
        Path, typer.Argument(help="Path to PDF file or directory containing PDFs")
    ],
    output_dir: Annotated[
        Path | None,
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
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            "-n",
            help="Show files that would be converted without processing",
        ),
    ] = False,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            help="Mistral API Key (overrides .env)",
            envvar="MISTRAL_API_KEY",
        ),
    ] = None,
) -> None:
    """Convert PDF files to Markdown using Mistral OCR."""
    # Validate input path
    if not input_path.exists():
        console.print(f"[red]Error: Input path '{input_path}' does not exist.[/red]")
        raise typer.Exit(code=1)

    plan = create_conversion_plan(input_path, output_dir, force, clipboard)
    resolved_output_dir = plan.output_dir
    try:
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        console.print(
            f"[red]Error creating output directory '{resolved_output_dir}': {e}[/red]"
        )
        raise typer.Exit(code=1)

    if not plan.files:
        console.print(f"[yellow]No PDF files found in {input_path}[/yellow]")
        raise typer.Exit(code=0)

    if dry_run:
        console.print(
            "[bold blue]Dry run mode - no files will be converted.[/bold blue]"
        )
        console.print(f"Input: {input_path.resolve()}")
        console.print(f"Output Directory: {resolved_output_dir.resolve()}")
        if plan.force:
            console.print(
                "[yellow]Force mode enabled: Existing files will be overwritten.[/yellow]"
            )
        if plan.clipboard:
            console.print(
                "[blue]Clipboard mode enabled: Content will be copied to clipboard.[/blue]"
            )

        to_convert = [
            a for a in plan.files if a.action == "convert" and not a.will_overwrite
        ]
        to_overwrite = [
            a for a in plan.files if a.action == "convert" and a.will_overwrite
        ]
        to_skip = [a for a in plan.files if a.action == "skip"]

        total = len(plan.files)
        console.print(
            f"\n[bold]Summary:[/bold] {len(to_convert)} to convert, {len(to_overwrite)} to overwrite, {len(to_skip)} to skip (total: {total})\n"
        )

        if to_convert:
            console.print("[bold green]Files to convert:[/bold green]")
            for action in to_convert:
                console.print(
                    f"  [green]{action.input_path}[/green] → [cyan]{action.output_path}[/cyan]"
                )
            console.print()
        if to_overwrite:
            console.print("[bold yellow]Files to overwrite:[/bold yellow]")
            for action in to_overwrite:
                console.print(
                    f"  [yellow]{action.input_path}[/yellow] → [cyan]{action.output_path}[/cyan] (will overwrite)"
                )
            console.print()
        if to_skip:
            console.print("[bold]Files to skip:[/bold]")
            for action in to_skip:
                console.print(
                    f"  [dim]{action.input_path}[/dim] → [dim]{action.output_path}[/dim] ([yellow]{action.skip_reason}[/yellow])"
                )
            console.print()

        console.print("[bold green]Dry run complete.[/bold green]")
        raise typer.Exit(code=0)

    # Initialize Mistral client
    client = initialize_mistral_client(api_key)
    if not client:
        console.print(
            "[red]Error: Mistral API key is required. "
            "Provide via --api-key option, MISTRAL_API_KEY environment variable, or .env file.[/red]"
        )
        raise typer.Exit(code=1)

    # Print configuration
    console.print(f"Input: {input_path.resolve()}")
    console.print(f"Output Directory: {resolved_output_dir.resolve()}")
    if plan.force:
        console.print(
            "[yellow]Force mode enabled: Existing files will be overwritten.[/yellow]"
        )
    if plan.clipboard:
        console.print(
            "[blue]Clipboard mode enabled: Content will be copied to clipboard.[/blue]"
        )

    try:
        pdf_files_to_process = [a.input_path for a in plan.files if a.is_converting]
        results = process_pdf_files(
            client, pdf_files_to_process, resolved_output_dir, plan.force
        )

        # Print processing summary
        print_processing_summary(results)

        # Handle clipboard operation if requested
        if plan.clipboard:
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


def main() -> None:
    """Entry point for the CLI application."""
    app()
