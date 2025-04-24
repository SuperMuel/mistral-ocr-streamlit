from pathlib import Path
from typing import Annotated
import typer
from rich.console import Console
from tqdm import tqdm
from ocr_utils import initialize_mistral_client, process_and_save_pdf

app = typer.Typer(help="Convert PDF files to Markdown using Mistral OCR")
console = Console()


def process_path(
    input_path: Path, output_dir: Path, force: bool, show_progress: bool = True
) -> list[tuple[bool, str]]:
    """Process a single file or directory of PDFs"""
    if not (client := initialize_mistral_client()):
        raise typer.Exit(code=1)

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

        if show_progress:
            iterator = tqdm(pdf_files)
            for pdf_file in iterator:
                iterator.set_description(f"Processing {pdf_file.name}")
                results.append(
                    process_and_save_pdf(client, pdf_file, output_dir, force)
                )
        else:
            for pdf_file in pdf_files:
                results.append(
                    process_and_save_pdf(client, pdf_file, output_dir, force)
                )

    return results


@app.command()
def convert(
    input_path: Annotated[
        Path, typer.Argument(help="Path to PDF file or directory containing PDFs")
    ],
    output_dir: Annotated[
        Path, typer.Option(help="Directory to save markdown files")
    ] = Path("output"),
    force: Annotated[
        bool, typer.Option(help="Overwrite existing markdown files")
    ] = False,
):
    """Convert PDF files to Markdown using Mistral OCR"""
    if not input_path.exists():
        console.print(f"[red]Error: {input_path} does not exist[/red]")
        raise typer.Exit(code=1)

    try:
        results = process_path(input_path, output_dir, force)

        # Print summary
        success_count = sum(1 for success, _ in results if success)
        fail_count = len(results) - success_count

        console.print("\n[bold]Summary:[/bold]")
        console.print(f"Successfully processed: [green]{success_count}[/green]")
        if fail_count > 0:
            console.print(f"Failed: [red]{fail_count}[/red]")

        # Print detailed messages
        if fail_count > 0:
            console.print("\n[bold red]Failures:[/bold red]")
            for success, message in results:
                if not success:
                    console.print(f"[red]• {message}[/red]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
