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
    "Marketing Cloud": 15_000,
    "Sales Cloud": 8_000,
    "Service Cloud": 8_000,
    "Commerce Cloud": 15_000,
    "Data Cloud": 12_000,
}

# Products that are commonly expected in modern Salesforce implementations
NOTABLE_MISSING_PRODUCTS = {
    "Agentforce": ["agentforce", "agent force", "autonomous agent"],
    "Einstein AI": ["einstein ai", "einstein gpt", "predictive ai", "generative ai"],
    "Shield": ["shield", "platform encryption", "event monitoring", "field audit"],
    "Revenue Cloud": ["revenue cloud", "billing", "cpq"],
}

# --- Integration Validation ---

INTEGRATION_PATTERNS = {
    "system": re.compile(
        r"(?i)(?:integra\w+|conector\w*|connector|conexi[oó]n|api)\s+"
        r"(?:con|with|to|a|de|from)?\s*([A-Z][\w\s]{2,30}?)(?:\s*[:\-\(,\.]|$)"
    ),
    "api_type": re.compile(
        r"(?i)(rest\s*api|soap\s*api|graphql|webhook|bulk\s*api|streaming\s*api)", re.IGNORECASE
    ),
    "frequency": re.compile(
        r"(?i)(real[- ]?time|batch\s+\w+|near[- ]?real[- ]?time|diario|daily|hourly|"
        r"cada\s+\d+\s+\w+|every\s+\d+\s+\w+|streaming|event[- ]?driven)"
    ),
    "direction": re.compile(
        r"(?i)(bi[- ]?direccional|bidirectional|ingesta|ingest|export|push|pull|"
        r"read|write|sync|unidirectional|unidireccional)"
    ),
}

INTEGRATION_READINESS_CHECKS = [
    ("api_documented", [
        "api documentada", "documented api", "documentación técnica",
        "technical documentation", "swagger", "openapi",
    ]),
    ("sandbox_access", [
        "sandbox", "ambiente de prueba", "test environment",
        "staging", "dev environment", "ambiente de desarrollo",
    ]),
    ("auth_defined", [
        "oauth", "api key", "token", "credenciales", "credentials",
        "autenticación", "authentication", "llaves de acceso",
    ]),
    ("error_handling", [
        "error handling", "manejo de errores", "retry", "reintento",
        "fallback", "dead letter", "monitoring", "monitoreo", "alertas",
    ]),
    ("data_volume", [
        "volumen", "volume", "registros", "records", "rows",
        "filas", "capacidad", "capacity", "límite", "limit",
    ]),
]


def _extract_integrations(text: str) -> list[dict]:
    """Extract integration details from proposal text."""
    normalized = _normalize_pdf_text(text)
    lower = normalized.lower()

    # Find named systems/platforms being integrated
    known_systems = {}
    system_patterns = [
        (r"(?i)(PMS|property management)", "PMS"),
        (r"(?i)(lealtad|loyalty|fiesta rewards|rewards)", "Loyalty/Rewards"),
        (r"(?i)(concierge\s+digital|concierge)", "Concierge Digital"),
        (r"(?i)(marketing\s+cloud)", "Marketing Cloud"),
        (r"(?i)(data\s+cloud|data\s+360)", "Data Cloud"),
        (r"(?i)(mulesoft|anypoint)", "MuleSoft"),
        (r"(?i)(sitio\s+web|website|web\s+portal)", "Website"),
        (r"(?i)(erp|sap|oracle|netsuite)", "ERP"),
        (r"(?i)(flip\.?to|referidos|referral)", "Referral Platform"),
    ]

    for pattern_str, system_name in system_patterns:
        if re.search(pattern_str, normalized):
            known_systems[system_name] = {"mentioned": True}

    # Enrich each system with integration details
    integrations = []
    for system_name, info in known_systems.items():
        # Find the context around this system's mentions
        system_lower = system_name.lower().split("/")[0]  # use first name
        contexts = []
        for line in normalized.splitlines():
            if system_lower in line.lower():
                contexts.append(line.strip())
        full_context = " ".join(contexts)
        context_lower = full_context.lower()

        # Detect API type
        api_match = INTEGRATION_PATTERNS["api_type"].search(full_context)
        api_type = api_match.group(1) if api_match else None

        # Detect frequency
        freq_match = INTEGRATION_PATTERNS["frequency"].search(full_context)
        frequency = freq_match.group(1) if freq_match else None

        # Detect direction
        dir_match = INTEGRATION_PATTERNS["direction"].search(full_context)
        direction = dir_match.group(1) if dir_match else None

        # Detect limits/caps
        cap_match = re.search(
            r"(?i)(?:hasta|up to|limit|cap|máximo|max)\s+(\d+)\s+(entidades|entities|conexi|connect|records|registros)",
            full_context,
        )
        cap = cap_match.group(0) if cap_match else None

        # Check readiness signals
        readiness = {}
        for check_name, keywords in INTEGRATION_READINESS_CHECKS:
            readiness[check_name] = any(kw in context_lower or kw in lower for kw in keywords)

        # Determine if in-scope or out-of-scope
        exclusion_context = ""
        for line in normalized.splitlines():
            if system_lower in line.lower() and any(
                kw in line.lower() for kw in ["exclu", "fuera", "out of scope", "no incluye", "not included"]
            ):
                exclusion_context = line.strip()
        in_scope = not bool(exclusion_context)

        integrations.append({
            "system": system_name,
            "in_scope": in_scope,
            "exclusion_note": exclusion_context,
            "api_type": api_type,
            "frequency": frequency,
            "direction": direction,
            "cap": cap,
            "readiness": readiness,
        })

    return integrations


def _validate_integrations(integrations: list[dict]) -> list[dict]:
    """Validate each integration and flag risks."""
    for integ in integrations:
        risks = []
        if integ["in_scope"]:
            if not integ["api_type"]:
                risks.append("API type not specified (REST, SOAP, etc.)")
            if not integ["frequency"]:
                risks.append("Data sync frequency not defined")
            if not integ["direction"]:
                risks.append("Data flow direction unclear")
            readiness = integ["readiness"]
            if not readiness.get("api_documented"):
                risks.append("No mention of API documentation")
            if not readiness.get("sandbox_access"):
                risks.append("No sandbox/test environment mentioned")
            if not readiness.get("auth_defined"):
                risks.append("Authentication method not specified")
            if not readiness.get("error_handling"):
                risks.append("No error handling or monitoring strategy")
        integ["risks"] = risks
        integ["risk_level"] = (
            "high" if len(risks) >= 4
            else "medium" if len(risks) >= 2
            else "low"
        )
    return integrations


def parse_integrations(text: str) -> list[dict]:
    """Extract and validate integrations from proposal text."""
    integrations = _extract_integrations(text)
    return _validate_integrations(integrations)


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


def score_proposal(pricing: dict, scope: dict, integrations: list[dict]) -> dict:
    """Score proposal pricing against benchmarks and compute overall score."""
    products = scope.get("products", {})
    line_items = pricing.get("line_items", [])
    total = pricing.get("total", 0)

    product_costs = _match_line_items_to_products(line_items, products)

    product_results = {}
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
            product_results[product] = {
                "cost": cost,
                "benchmark": benchmark,
                "ratio": round(ratio, 2),
                "rating": rating,
            }
        else:
            product_results[product] = {
                "cost": cost,
                "benchmark": benchmark,
                "ratio": None,
                "rating": "unpriced" if not cost else "no_benchmark",
            }

    # --- Pricing score (0-10): how competitive is the pricing? ---
    # Sum individual benchmarks, then apply a bundled-project multiplier (1.5x)
    # since multi-product implementations have integration overhead, PM, etc.
    total_benchmark = sum(BENCHMARKS.get(p, 0) for p in products)
    bundled_benchmark = total_benchmark * 1.5
    if bundled_benchmark and total:
        price_ratio = total / bundled_benchmark
        if price_ratio <= 0.7:
            pricing_score = 10.0
            pricing_label = "Very Competitive"
        elif price_ratio <= 1.0:
            pricing_score = 8.0
            pricing_label = "Competitive"
        elif price_ratio <= 1.3:
            pricing_score = 6.0
            pricing_label = "Fair"
        elif price_ratio <= 1.6:
            pricing_score = 4.0
            pricing_label = "Above Market"
        else:
            pricing_score = 2.0
            pricing_label = "Expensive"
    else:
        pricing_score = 5.0
        pricing_label = "Unverifiable"

    # --- Scope score (0-10): coverage breadth and completeness ---
    detected_count = len(products)
    has_in_scope = bool(scope.get("sections", {}).get("in_scope"))
    has_out_scope = bool(scope.get("sections", {}).get("out_of_scope"))

    scope_score = min(10.0, detected_count * 2.0)
    if has_in_scope:
        scope_score = min(10.0, scope_score + 1.0)
    if has_out_scope:
        scope_score = min(10.0, scope_score + 1.0)

    # Detect notable missing products
    normalized_text = _normalize_pdf_text(
        " ".join(f for feats in products.values() for f in feats)
        + " " + " ".join(products.keys())
    ).lower()
    missing_products = []
    for product_name, keywords in NOTABLE_MISSING_PRODUCTS.items():
        if not any(kw in normalized_text for kw in keywords):
            missing_products.append(product_name)

    # --- Integration score (0-10) ---
    in_scope_integrations = [i for i in integrations if i["in_scope"]]
    if in_scope_integrations:
        risk_scores = {"low": 10, "medium": 6, "high": 3}
        integ_score = sum(
            risk_scores.get(i["risk_level"], 5) for i in in_scope_integrations
        ) / len(in_scope_integrations)
    else:
        integ_score = 5.0

    integ_score = round(integ_score, 1)

    # --- Overall score (1-10 scale) ---
    overall = round(
        pricing_score * 0.35 + scope_score * 0.35 + integ_score * 0.30, 1
    )

    return {
        "products": product_results,
        "overall_score": overall,
        "pricing_score": pricing_score,
        "pricing_label": pricing_label,
        "scope_score": scope_score,
        "integration_score": integ_score,
        "missing_products": missing_products,
        "total_proposed": total,
        "total_benchmark": total_benchmark,
    }


RATING_LABELS = {
    "good": "Below benchmark",
    "fair": "Near benchmark",
    "high": "Above benchmark",
    "unpriced": "No pricing found",
    "no_benchmark": "No benchmark available",
}


def _generate_suggestions(scores: dict, scope: dict, integrations: list[dict]) -> list[str]:
    """Generate actionable suggestions based on scores, scope, and integrations."""
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

    # Missing products
    for product_name in scores.get("missing_products", []):
        suggestions.append(
            f"**{product_name}** is not included in this proposal. "
            f"Evaluate whether it should be part of the implementation roadmap."
        )

    # Integration risks
    high_risk = [i for i in integrations if i["risk_level"] == "high" and i["in_scope"]]
    for integ in high_risk:
        top_risks = ", ".join(integ["risks"][:2])
        suggestions.append(
            f"Integration with **{integ['system']}** has high risk: {top_risks}. "
            f"Address before project kickoff."
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


def _generate_exec_summary(pricing: dict, scope: dict, scores: dict, integrations: list[dict]) -> list[str]:
    """Generate executive summary section."""
    lines = []
    lines.append("## Executive Summary")
    lines.append("")

    overall = scores.get("overall_score", 0)
    lines.append(f"**Proposal Score: {overall}/10**")
    lines.append("")

    # Pricing assessment
    total = pricing.get("total", 0)
    pricing_label = scores.get("pricing_label", "Unknown")
    total_benchmark = scores.get("total_benchmark", 0)
    detected_products = list(scope.get("products", {}).keys())

    benchmark_detail = ""
    if detected_products and total_benchmark:
        product_benchmarks = []
        for p in detected_products:
            b = BENCHMARKS.get(p)
            if b:
                product_benchmarks.append(f"{p} ~${b:,}")
        if product_benchmarks:
            benchmark_detail = f" [{', '.join(product_benchmarks)} benchmarks]"

    lines.append(
        f"**Pricing:** {pricing_label} — ${total:,.0f} USD total"
        f"{benchmark_detail}"
    )

    # Scope assessment
    covered = ", ".join(detected_products) if detected_products else "None detected"
    missing = scores.get("missing_products", [])
    missing_str = f" — misses {', '.join(missing)}" if missing else ""
    lines.append(f"**Scope:** Covers {covered}{missing_str}")

    # Integration health
    in_scope_integrations = [i for i in integrations if i["in_scope"]]
    high_risk_count = sum(1 for i in in_scope_integrations if i["risk_level"] == "high")
    med_risk_count = sum(1 for i in in_scope_integrations if i["risk_level"] == "medium")
    if in_scope_integrations:
        integ_summary = f"{len(in_scope_integrations)} integrations"
        risk_parts = []
        if high_risk_count:
            risk_parts.append(f"{high_risk_count} high-risk")
        if med_risk_count:
            risk_parts.append(f"{med_risk_count} medium-risk")
        if risk_parts:
            integ_summary += f" ({', '.join(risk_parts)})"
        else:
            integ_summary += " (all low-risk)"
    else:
        integ_summary = "No integrations detected"
    lines.append(f"**Integrations:** {integ_summary}")
    lines.append("")

    # Top suggestions (max 3)
    suggestions = _generate_suggestions(scores, scope, integrations)
    lines.append("**Key Recommendations:**")
    for s in suggestions[:3]:
        lines.append(f"1. {s}")
    lines.append("")

    return lines


def _generate_integration_section(integrations: list[dict]) -> list[str]:
    """Generate the integration validation section."""
    lines = []
    lines.append("## Integration Validation")
    lines.append("")

    if not integrations:
        lines.append("_No integrations detected._")
        lines.append("")
        return lines

    in_scope = [i for i in integrations if i["in_scope"]]
    out_scope = [i for i in integrations if not i["in_scope"]]

    if in_scope:
        lines.append("### In-Scope Integrations")
        lines.append("")
        lines.append("| System | API Type | Frequency | Direction | Cap | Risk |")
        lines.append("|--------|----------|-----------|-----------|-----|------|")
        for integ in in_scope:
            api = integ["api_type"] or "—"
            freq = integ["frequency"] or "—"
            direction = integ["direction"] or "—"
            cap = integ["cap"] or "—"
            risk = integ["risk_level"].upper()
            lines.append(f"| {integ['system']} | {api} | {freq} | {direction} | {cap} | {risk} |")
        lines.append("")

        # Detail risks per integration
        risky = [i for i in in_scope if i["risks"]]
        if risky:
            lines.append("### Risk Details")
            lines.append("")
            for integ in risky:
                lines.append(f"**{integ['system']}** ({integ['risk_level']} risk):")
                for risk in integ["risks"]:
                    lines.append(f"- {risk}")
                lines.append("")

        # Readiness checklist
        lines.append("### Readiness Checklist")
        lines.append("")
        check_labels = {
            "api_documented": "API Documentation",
            "sandbox_access": "Sandbox/Test Environment",
            "auth_defined": "Authentication Defined",
            "error_handling": "Error Handling/Monitoring",
            "data_volume": "Data Volume/Limits Defined",
        }
        lines.append("| Check | " + " | ".join(i["system"] for i in in_scope) + " |")
        lines.append("|-------| " + " | ".join("---" for _ in in_scope) + " |")
        for check_key, check_label in check_labels.items():
            row = f"| {check_label} |"
            for integ in in_scope:
                val = integ["readiness"].get(check_key, False)
                row += f" {'Yes' if val else 'No'} |"
            lines.append(row)
        lines.append("")

    if out_scope:
        lines.append("### Out-of-Scope Integrations")
        lines.append("")
        for integ in out_scope:
            note = integ["exclusion_note"] or "Mentioned but excluded"
            lines.append(f"- **{integ['system']}**: {note}")
        lines.append("")

    return lines


def generate_report(pricing: dict, scope: dict, scores: dict, integrations: list[dict]) -> str:
    """Generate a Markdown report with analysis and suggestions."""
    lines = ["# Proposal Analysis Report", ""]

    # --- Executive Summary ---
    lines.extend(_generate_exec_summary(pricing, scope, scores, integrations))

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

    # --- Integration Validation ---
    lines.extend(_generate_integration_section(integrations))

    # --- Benchmark Scoring ---
    lines.append("## Benchmark Scoring")
    lines.append("")
    overall = scores.get("overall_score", 0)
    pricing_s = scores.get("pricing_score", 0)
    scope_s = scores.get("scope_score", 0)
    integ_s = scores.get("integration_score", 0)
    lines.append(f"**Overall: {overall}/10** (Pricing: {pricing_s}/10 | Scope: {scope_s}/10 | Integrations: {integ_s}/10)")
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
    suggestions = _generate_suggestions(scores, scope, integrations)
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
    integrations = parse_integrations(text)
    scores = score_proposal(pricing, scope, integrations)
    report = generate_report(pricing, scope, scores, integrations)

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
