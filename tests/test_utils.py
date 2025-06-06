"""Utilities for generating test PDFs with complex directory structures."""

from pathlib import Path
from fpdf import FPDF


def create_test_pdfs(base_dir: Path) -> dict[str, str]:
    """
    Create test PDFs with unique content in a complex directory structure.

    Returns a dict mapping relative file paths to their expected content.
    """

    test_files = {
        "simple.pdf": "This is a simple test document.",
        "reports/quarterly_report.pdf": "Quarterly sales report shows strong growth in Q3.",
        "reports/financial/budget_2024.pdf": "Annual budget allocation for department operations.",
        "reports/financial/expenses.pdf": "Monthly expense tracking and cost analysis.",
        "documents/contracts/vendor_agreement.pdf": "Service agreement between company and external vendor.",
        "documents/policies/security_policy.pdf": "Information security guidelines and procedures.",
        "archive/2023/meeting_notes.pdf": "Board meeting minutes from December 2023.",
        "archive/2023/projects/project_alpha.pdf": "Project Alpha final deliverables and outcomes.",
        "archive/2024/annual_review.pdf": "Company annual performance review and goals.",
    }

    expected_content: dict[str, str] = {}

    for relative_path, content in test_files.items():
        file_path = base_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, content, new_x="LMARGIN", new_y="NEXT")
        pdf.output(str(file_path))

        expected_content[relative_path] = content

    return expected_content


def create_single_test_pdf(file_path: Path, content: str) -> None:
    """Create a single test PDF with the given content."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, content, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(file_path))
