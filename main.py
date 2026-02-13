import argparse
import re
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


PRICE_PATTERN = re.compile(
    r"\$\s?([\d,]+(?:\.\d{2})?)\s*(?:/\s*(mo|month|yr|year|user|license))?"
)

PRICING_KEYWORDS = [
    "total", "subtotal", "annual", "monthly", "license", "subscription",
    "implementation", "setup", "onboarding", "support", "discount",
    "one-time", "recurring",
]


def _extract_prices_from_text(text: str) -> list[dict]:
    """Find all dollar amounts in text with surrounding context."""
    items = []
    for line in text.splitlines():
        for match in PRICE_PATTERN.finditer(line):
            amount_str = match.group(1).replace(",", "")
            amount = float(amount_str)
            period = match.group(2) or ""
            label = _classify_line_item(line)
            items.append({
                "label": label,
                "amount": amount,
                "period": period,
                "context": line.strip(),
            })
    return items


def _classify_line_item(line: str) -> str:
    """Classify a pricing line item based on keyword matching."""
    lower = line.lower()
    for keyword in PRICING_KEYWORDS:
        if keyword in lower:
            return keyword
    return "other"


def _extract_prices_from_tables(tables: list[pd.DataFrame]) -> list[dict]:
    """Extract pricing data from tabular structures."""
    items = []
    for df in tables:
        price_cols = [
            col for col in df.columns
            if any(k in str(col).lower() for k in ["price", "cost", "amount", "fee", "total", "$"])
        ]
        label_cols = [
            col for col in df.columns
            if any(k in str(col).lower() for k in ["item", "description", "product", "service", "name", "component"])
        ]
        if not price_cols:
            # scan all cells for dollar amounts as fallback
            for _, row in df.iterrows():
                for val in row:
                    val_str = str(val)
                    for match in PRICE_PATTERN.finditer(val_str):
                        amount_str = match.group(1).replace(",", "")
                        items.append({
                            "label": "other",
                            "amount": float(amount_str),
                            "period": match.group(2) or "",
                            "context": " | ".join(str(v) for v in row),
                        })
            continue

        label_col = label_cols[0] if label_cols else None
        for _, row in df.iterrows():
            for pc in price_cols:
                val_str = str(row[pc])
                for match in PRICE_PATTERN.finditer(val_str):
                    amount_str = match.group(1).replace(",", "")
                    label = str(row[label_col]).strip() if label_col else _classify_line_item(val_str)
                    items.append({
                        "label": label,
                        "amount": float(amount_str),
                        "period": match.group(2) or "",
                        "context": " | ".join(str(v) for v in row),
                    })
    return items


def parse_pricing(text: str, tables: list[pd.DataFrame]) -> dict:
    """Parse pricing details from extracted text and tables."""
    text_items = _extract_prices_from_text(text)
    table_items = _extract_prices_from_tables(tables)

    # deduplicate by (amount, context) — prefer table-sourced items
    seen = set()
    all_items = []
    for item in table_items + text_items:
        key = (item["amount"], item["context"])
        if key not in seen:
            seen.add(key)
            all_items.append(item)

    total = sum(
        item["amount"] for item in all_items
        if item["label"] in ("total", "subtotal")
    )
    if not total:
        total = sum(item["amount"] for item in all_items)

    return {
        "line_items": all_items,
        "total": total,
    }


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
