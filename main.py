import argparse
import sys
from pathlib import Path

import PyPDF2
import pandas as pd
import tabula


def extract_text(pdf_path: str) -> str:
    """Extract raw text from a PDF file."""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_tables(pdf_path: str) -> list[pd.DataFrame]:
    """Extract tables from a PDF file."""
    return tabula.read_pdf(pdf_path, pages="all", multiple_tables=True)


def parse_pricing(text: str, tables: list[pd.DataFrame]) -> dict:
    """Parse pricing details from extracted text and tables."""
    # TODO: implement pricing extraction logic
    return {}


def parse_scope(text: str) -> dict:
    """Parse scope and Salesforce features from extracted text."""
    # TODO: implement scope/feature extraction logic
    return {}


BENCHMARKS = {
    "Marketing Cloud": 10_000,
    "Sales Cloud": 8_000,
    "Service Cloud": 8_000,
    "Commerce Cloud": 15_000,
    "Data Cloud": 12_000,
}


def score_proposal(pricing: dict, scope: dict) -> dict:
    """Score proposal pricing against benchmarks."""
    # TODO: implement scoring logic
    return {}


def generate_report(pricing: dict, scope: dict, scores: dict) -> str:
    """Generate a Markdown report with analysis and suggestions."""
    lines = [
        "# Proposal Analysis Report",
        "",
        "## Pricing Summary",
        "<!-- TODO: populate pricing details -->",
        "",
        "## Scope & Salesforce Features",
        "<!-- TODO: populate scope details -->",
        "",
        "## Benchmark Scoring",
        "<!-- TODO: populate scores -->",
        "",
        "## Suggestions",
        "<!-- TODO: populate suggestions -->",
        "",
    ]
    return "\n".join(lines)


def analyze(pdf_path: str, output_path: str) -> None:
    """Run the full analysis pipeline."""
    if not Path(pdf_path).exists():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    text = extract_text(pdf_path)
    tables = extract_tables(pdf_path)
    pricing = parse_pricing(text, tables)
    scope = parse_scope(text)
    scores = score_proposal(pricing, scope)
    report = generate_report(pricing, scope, scores)

    Path(output_path).write_text(report)
    print(f"Report written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Salesforce proposals from PDF files."
    )
    subparsers = parser.add_subparsers(dest="command")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a proposal PDF")
    analyze_parser.add_argument("input", help="Path to the proposal PDF")
    analyze_parser.add_argument(
        "--output", "-o", default="report.md", help="Output Markdown report path"
    )

    args = parser.parse_args()

    if args.command == "analyze":
        analyze(args.input, args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
