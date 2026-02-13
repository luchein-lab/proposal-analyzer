from __future__ import annotations

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
    try:
        return tabula.read_pdf(pdf_path, pages="all", multiple_tables=True)
    except Exception as e:
        print(f"Warning: table extraction failed ({e}). Continuing with text only.", file=sys.stderr)
        return []


PRICE_PATTERN = re.compile(
    r"\$\s?([\d,]+(?:\.\d{2})?)\s*(?:/\s*(mo|month|yr|year|user|license))?"
)

# (label, keywords) — first match wins, so order matters
PRICING_CATEGORIES = [
    # Totals
    ("total", ["total", "subtotal", "gran total"]),
    # Payment milestones
    ("milestone", [
        "anticipo", "advance", "pago final", "final payment",
        "segundo pago", "second payment", "tercer pago", "third payment",
        "hito", "milestone", "firma", "signing", "go-live", "kick-off",
    ]),
    # Rates
    ("rate", [
        "por hora", "per hour", "/hora", "/hour", "blended",
        "tarifa", "rate", "t&m", "time & material",
    ]),
    # Implementation / services
    ("implementation", [
        "implementación", "implementacion", "implementation",
        "servicios", "services", "configuración", "configuracion",
        "setup", "onboarding", "arranque",
    ]),
    # Licensing
    ("license", [
        "licencia", "license", "suscripción", "suscripcion",
        "subscription", "usuario", "user", "seat",
    ]),
    # Support
    ("support", [
        "soporte", "support", "mantenimiento", "maintenance",
        "hypercare", "estabilización", "estabilizacion",
    ]),
    # Discounts
    ("discount", [
        "descuento", "discount", "bonificación", "bonificacion",
    ]),
    # Recurring
    ("recurring", [
        "mensual", "monthly", "anual", "annual", "yearly",
        "recurrente", "recurring",
    ]),
]


def _normalize_pdf_text(text: str) -> str:
    """Collapse excessive whitespace from PDF extraction artifacts."""
    # Replace newlines preceded/followed by a word char with a space
    # (fixes "Marketing\nCloud" -> "Marketing Cloud")
    text = re.sub(r"(\w)\s*\n\s*(\w)", r"\1 \2", text)
    # Collapse runs of spaces (common in PDF text extraction)
    text = re.sub(r" {2,}", " ", text)
    # Fix PDF artifact where uppercase letter is separated from rest of word
    # e.g. "M arketing" -> "Marketing", "D ata" -> "Data"
    text = re.sub(r"\b([A-Z]) ([a-z])", r"\1\2", text)
    return text


def _get_context_window(lines: list[str], idx: int, window: int = 5) -> str:
    """Get surrounding non-empty lines as context for a price match."""
    start = max(0, idx - window)
    end = min(len(lines), idx + window + 1)
    context_lines = []
    for i in range(start, end):
        stripped = lines[i].strip()
        if stripped:
            context_lines.append(stripped)
    return " ".join(context_lines)


def _classify_line_item(text: str) -> str:
    """Classify a pricing line item based on keyword matching."""
    lower = text.lower()
    for label, keywords in PRICING_CATEGORIES:
        for keyword in keywords:
            if keyword in lower:
                return label
    return "other"


def _extract_prices_from_text(text: str) -> list[dict]:
    """Find all dollar amounts in text with surrounding context."""
    normalized = _normalize_pdf_text(text)
    lines = normalized.splitlines()
    items = []
    for idx, line in enumerate(lines):
        for match in PRICE_PATTERN.finditer(line):
            amount_str = match.group(1).replace(",", "")
            amount = float(amount_str)
            period = match.group(2) or ""
            context = _get_context_window(lines, idx)
            label = _classify_line_item(context)
            items.append({
                "label": label,
                "amount": amount,
                "period": period,
                "context": context,
            })
    return items


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

    # Find the total: use the max "total"-labeled item (usually the grand total),
    # then fall back to summing "implementation" items, then sum everything.
    total_items = [item["amount"] for item in all_items if item["label"] == "total"]
    if total_items:
        total = max(total_items)
    else:
        impl_items = [item["amount"] for item in all_items if item["label"] == "implementation"]
        total = max(impl_items) if impl_items else sum(item["amount"] for item in all_items)

    return {
        "line_items": all_items,
        "total": total,
    }


SALESFORCE_PRODUCTS = {
    "Marketing Cloud": [
        "journey builder", "email studio", "advertising studio",
        "mobile studio", "social studio", "audience builder",
        "marketing cloud connect", "marketing cloud intelligence",
        "marketing cloud personalization",
    ],
    "Sales Cloud": [
        "lead management", "opportunity management", "forecasting",
        "pipeline management", "cpq", "revenue cloud", "sales engagement",
        "territory management", "einstein activity capture",
    ],
    "Service Cloud": [
        "case management", "knowledge base", "omni-channel",
        "field service", "visual remote assistant", "service console",
        "chatbot", "live agent", "einstein bots",
    ],
    "Commerce Cloud": [
        "b2c commerce", "b2b commerce", "order management",
        "commerce storefront", "headless commerce", "payments",
    ],
    "Data Cloud": [
        "data cloud", "cdp", "customer data platform", "data streams",
        "identity resolution", "calculated insights", "segmentation",
        "data graphs",
    ],
    "Experience Cloud": [
        "experience cloud", "community cloud", "customer portal",
        "partner portal", "experience builder",
    ],
    "Tableau": [
        "tableau", "crm analytics", "analytics cloud",
        "einstein analytics", "dashboard", "reporting",
    ],
    "MuleSoft": [
        "mulesoft", "anypoint", "api management", "integration",
    ],
    "Platform": [
        "lightning", "apex", "visualforce", "flow", "process builder",
        "custom objects", "app builder", "platform events",
        "shield", "event monitoring",
    ],
}

SCOPE_SECTION_PATTERNS = [
    re.compile(r"(?i)(?:scope\s+of\s+work|project\s+scope|deliverables|phases?|alcance)\s*[:\n]"),
    re.compile(r"(?i)(?:out\s+of\s+scope|exclusions|exclusiones|fuera\s+del?\s+alcance)\s*[:\n]"),
]


def _detect_products(text: str) -> dict[str, list[str]]:
    """Detect Salesforce products and specific features mentioned."""
    normalized = _normalize_pdf_text(text)
    lower = normalized.lower()
    detected = {}
    for product, features in SALESFORCE_PRODUCTS.items():
        matched_features = [f for f in features if f in lower]
        if matched_features or product.lower() in lower:
            detected[product] = matched_features
    return detected


def _extract_scope_sections(text: str) -> dict[str, str]:
    """Extract in-scope and out-of-scope sections from text."""
    normalized = _normalize_pdf_text(text)
    sections = {}
    lines = normalized.splitlines()
    for i, line in enumerate(lines):
        for pattern in SCOPE_SECTION_PATTERNS:
            if pattern.search(line):
                key = "out_of_scope" if "out" in line.lower() or "exclusion" in line.lower() else "in_scope"
                block = []
                for subsequent in lines[i + 1:]:
                    # stop at the next heading-like line or empty gap
                    if subsequent.strip() == "":
                        if block:
                            break
                        continue
                    if re.match(r"^[A-Z][A-Za-z\s]{2,30}[:\n]", subsequent):
                        break
                    block.append(subsequent.strip())
                if block:
                    sections.setdefault(key, []).extend(block)
    return {k: "\n".join(v) for k, v in sections.items()}


def _estimate_user_count(text: str) -> int | None:
    """Try to find user/license count from text."""
    normalized = _normalize_pdf_text(text)
    patterns = [
        re.compile(r"(\d+)\s*(?:users?|licenses?|seats?|usuarios?|licencias?)", re.IGNORECASE),
        re.compile(r"(?:users?|licenses?|seats?|usuarios?|licencias?)\s*[:\-]\s*(\d+)", re.IGNORECASE),
    ]
    for pattern in patterns:
        match = pattern.search(normalized)
        if match:
            count = int(match.group(1))
            if count >= 2:  # avoid false positives like "1 conexión"
                return count
    return None


def parse_scope(text: str) -> dict:
    """Parse scope and Salesforce features from extracted text."""
    products = _detect_products(text)
    sections = _extract_scope_sections(text)
    user_count = _estimate_user_count(text)

    return {
        "products": products,
        "sections": sections,
        "user_count": user_count,
    }


BENCHMARKS = {
    "Marketing Cloud": 10_000,
    "Sales Cloud": 8_000,
    "Service Cloud": 8_000,
    "Commerce Cloud": 15_000,
    "Data Cloud": 12_000,
}


def _match_line_items_to_products(
    line_items: list[dict], products: dict[str, list[str]]
) -> dict[str, float]:
    """Map extracted line items to Salesforce products by keyword matching."""
    product_totals: dict[str, float] = {}
    for item in line_items:
        context_lower = item["context"].lower()
        for product in products:
            if product.lower() in context_lower:
                product_totals[product] = product_totals.get(product, 0) + item["amount"]
    return product_totals


def score_proposal(pricing: dict, scope: dict) -> dict:
    """Score proposal pricing against benchmarks."""
    products = scope.get("products", {})
    line_items = pricing.get("line_items", [])
    total = pricing.get("total", 0)

    product_costs = _match_line_items_to_products(line_items, products)

    results = {}
    for product in products:
        benchmark = BENCHMARKS.get(product)
        cost = product_costs.get(product, 0)

        if benchmark and cost:
            ratio = cost / benchmark
            if ratio <= 0.8:
                rating = "good"
            elif ratio <= 1.2:
                rating = "fair"
            else:
                rating = "high"
            results[product] = {
                "cost": cost,
                "benchmark": benchmark,
                "ratio": round(ratio, 2),
                "rating": rating,
            }
        else:
            results[product] = {
                "cost": cost,
                "benchmark": benchmark,
                "ratio": None,
                "rating": "unpriced" if not cost else "no_benchmark",
            }

    # overall score: 0-100 based on how many products rate well
    rated = [r for r in results.values() if r["ratio"] is not None]
    if rated:
        avg_ratio = sum(r["ratio"] for r in rated) / len(rated)
        overall = max(0, min(100, round(100 * (2 - avg_ratio) / 2)))
    else:
        overall = 0

    return {
        "products": results,
        "overall_score": overall,
        "total_proposed": total,
    }


RATING_LABELS = {
    "good": "Below benchmark",
    "fair": "Near benchmark",
    "high": "Above benchmark",
    "unpriced": "No pricing found",
    "no_benchmark": "No benchmark available",
}


def _generate_suggestions(scores: dict, scope: dict) -> list[str]:
    """Generate actionable suggestions based on scores and scope."""
    suggestions = []
    product_scores = scores.get("products", {})

    for product, info in product_scores.items():
        rating = info["rating"]
        if rating == "high":
            ratio = info["ratio"]
            suggestions.append(
                f"**{product}** is priced {ratio}x the benchmark "
                f"(${info['cost']:,.0f} vs ${info['benchmark']:,.0f}). "
                f"Negotiate or request itemized justification."
            )
        elif rating == "unpriced":
            suggestions.append(
                f"**{product}** is referenced in scope but has no "
                f"associated pricing. Request a detailed cost breakdown."
            )

    detected = scope.get("products", {})
    for product, features in detected.items():
        if product in BENCHMARKS and product not in product_scores:
            suggestions.append(
                f"**{product}** features detected ({', '.join(features[:3])}) "
                f"but not scored. Confirm whether it's included in the proposal."
            )

    if not scope.get("sections", {}).get("out_of_scope"):
        suggestions.append(
            "No out-of-scope section detected. Request explicit exclusions "
            "to avoid scope creep."
        )

    if not scope.get("user_count"):
        suggestions.append(
            "No user/license count found. Clarify the number of seats "
            "to validate per-user pricing."
        )

    if not suggestions:
        suggestions.append("Proposal looks well-structured. No major concerns.")

    return suggestions


def generate_report(pricing: dict, scope: dict, scores: dict) -> str:
    """Generate a Markdown report with analysis and suggestions."""
    lines = ["# Proposal Analysis Report", ""]

    # --- Pricing Summary ---
    lines.append("## Pricing Summary")
    lines.append("")
    total = pricing.get("total", 0)
    lines.append(f"**Total proposed cost:** ${total:,.2f}")
    lines.append("")
    line_items = pricing.get("line_items", [])
    if line_items:
        lines.append("| Label | Amount | Period | Context |")
        lines.append("|-------|--------|--------|---------|")
        for item in line_items:
            period = item["period"] or "—"
            context = item["context"][:60] + "..." if len(item["context"]) > 60 else item["context"]
            lines.append(
                f"| {item['label']} | ${item['amount']:,.2f} | {period} | {context} |"
            )
        lines.append("")
    else:
        lines.append("_No line items extracted._")
        lines.append("")

    # --- Scope & Features ---
    lines.append("## Scope & Salesforce Features")
    lines.append("")
    products = scope.get("products", {})
    if products:
        for product, features in products.items():
            feature_str = ", ".join(features) if features else "_product name only_"
            lines.append(f"- **{product}**: {feature_str}")
        lines.append("")
    else:
        lines.append("_No Salesforce products detected._")
        lines.append("")

    user_count = scope.get("user_count")
    if user_count:
        lines.append(f"**Estimated users/licenses:** {user_count}")
        lines.append("")

    sections = scope.get("sections", {})
    if sections.get("in_scope"):
        lines.append("### In Scope")
        lines.append("")
        lines.append(sections["in_scope"])
        lines.append("")
    if sections.get("out_of_scope"):
        lines.append("### Out of Scope")
        lines.append("")
        lines.append(sections["out_of_scope"])
        lines.append("")

    # --- Benchmark Scoring ---
    lines.append("## Benchmark Scoring")
    lines.append("")
    overall = scores.get("overall_score", 0)
    lines.append(f"**Overall score:** {overall}/100")
    lines.append("")
    product_scores = scores.get("products", {})
    if product_scores:
        lines.append("| Product | Cost | Benchmark | Ratio | Rating |")
        lines.append("|---------|------|-----------|-------|--------|")
        for product, info in product_scores.items():
            cost = f"${info['cost']:,.0f}" if info["cost"] else "—"
            benchmark = f"${info['benchmark']:,.0f}" if info["benchmark"] else "—"
            ratio = f"{info['ratio']}x" if info["ratio"] is not None else "—"
            rating = RATING_LABELS.get(info["rating"], info["rating"])
            lines.append(f"| {product} | {cost} | {benchmark} | {ratio} | {rating} |")
        lines.append("")
    else:
        lines.append("_No products scored._")
        lines.append("")

    # --- Suggestions ---
    lines.append("## Suggestions")
    lines.append("")
    suggestions = _generate_suggestions(scores, scope)
    for s in suggestions:
        lines.append(f"- {s}")
    lines.append("")

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
