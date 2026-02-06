import re
import json
from typing import List, Dict, Any, Optional

# -------------------------
# ION REQUIRED OUTPUT FIELDS
# -------------------------
ION_FIELDS = [
    "Direct Lender",
    "Lead/Primary Borrower",
    "Additional Borrower",
    "Business Description in Datalab",
    "SIG/NAICS in Datalab",
    "Pricing Date",
    "Instrument type",
    "Seniority Ranking",
    "Secured/Subordinated",
    "BDC Coupon Rate",
    "Basis",
    "Floor",
    "Margin",
    "PIK Margin",
    "Cash Rate",
    "Implied coupon",
    "Maturity Type",
    "Currency Type",
    "Principal Type",
    "Amortised Cost",
    "Fair Value",
    "Report Date",
    "Report Type",
    "Source",
]

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _is_empty_cell(x: Any) -> bool:
    return x is None or str(x).strip() == "" or str(x).strip() == "-"

def _looks_like_section_row(row: List[str]) -> bool:
    # A section row usually has meaningful text in 1st col and rest empty
    nonempty = [c for c in row if not _is_empty_cell(c)]
    return len(nonempty) == 1 and len(nonempty[0]) > 3

def _looks_like_header_row(row: List[str]) -> bool:
    t = _norm(" ".join(row))
    header_hits = [
        "portfolio", "issuer", "company", "industry", "sector",
        "security", "instrument", "maturity", "principal", "par",
        "cost", "amort", "fair value", "interest", "spread", "margin",
        "floor", "basis", "cash rate", "coupon", "notes", "% of"
    ]
    hits = sum(1 for h in header_hits if h in t)
    return hits >= 3

def _collapse_spacer_columns(rows: List[List[str]], min_keep_ratio: float = 0.02) -> List[List[str]]:
    """
    Many SOI tables are like: value, "", value, "", value ...
    We drop columns that are ~always empty across the table.
    """
    if not rows:
        return rows
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]

    nonempty_counts = [0] * width
    for r in padded:
        for i, v in enumerate(r):
            if not _is_empty_cell(v):
                nonempty_counts[i] += 1

    n = len(padded)
    keep = []
    for i, cnt in enumerate(nonempty_counts):
        if cnt / max(1, n) >= min_keep_ratio:
            keep.append(i)

    # If we kept almost everything (rare), still try to remove strict spacer pattern:
    if len(keep) > int(width * 0.9):
        # remove columns that are empty in the header + first 20 rows
        sample = padded[: min(20, n)]
        keep2 = []
        for i in range(width):
            if any(not _is_empty_cell(r[i]) for r in sample):
                keep2.append(i)
        if len(keep2) >= 3:
            keep = keep2

    return [[r[i] for i in keep] for r in padded]

def _merge_multiline_headers(header_rows: List[List[str]]) -> List[str]:
    """
    SOI headers often come in 2 lines (e.g., "Par Amount /" + "Quantity").
    We merge by column position.
    """
    if not header_rows:
        return []
    width = max(len(r) for r in header_rows)
    padded = [r + [""] * (width - len(r)) for r in header_rows]

    merged = []
    for col in range(width):
        parts = []
        for hr in padded:
            v = str(hr[col]).strip()
            if v and v != "-":
                parts.append(v)
        name = " ".join(parts).strip()
        merged.append(name if name else f"col_{col+1}")
    return merged

def _parse_number(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    t = str(s).strip()
    if t in ("", "-", "—"):
        return None
    neg = False
    # parentheses = negative (common in filings)
    if t.startswith("(") and t.endswith(")"):
        neg = True
        t = t[1:-1]
    t = t.replace("$", "").replace(",", "").strip()
    # keep percent separate
    if t.endswith("%"):
        t = t[:-1].strip()
    try:
        val = float(t)
        return -val if neg else val
    except:
        return None

def _infer_currency_from_row(row_dict: Dict[str, str]) -> Optional[str]:
    blob = " ".join(str(v) for v in row_dict.values())
    if "$" in blob:
        return "USD"
    if "€" in blob or "eur" in _norm(blob):
        return "EUR"
    if "gbp" in _norm(blob) or "£" in blob:
        return "GBP"
    return None

def _extract_rate_components(rate_text: str) -> Dict[str, Optional[str]]:
    """
    Converts things like:
    - "SOFR + 5.75%" -> Basis=SOFR, Margin=5.75%
    - "1M SOFR + 7.85% (12.21%)" -> Basis=SOFR, Margin=7.85%, Implied coupon=12.21%
    - "Prime + 4.25%" -> Basis=Prime, Margin=4.25%
    Also handles Floor if present like "SOFR (0.50%) + 6.00%" or "SOFR subject to 1.0% floor"
    """
    out = {"Basis": None, "Floor": None, "Margin": None, "Implied coupon": None, "BDC Coupon Rate": None, "Cash Rate": None, "PIK Margin": None}

    t = str(rate_text or "").strip()
    t_norm = _norm(t)

    # implied coupon in parentheses
    m = re.search(r"\(([-+]?\d+(\.\d+)?)\s*%\)", t)
    if m:
        out["Implied coupon"] = m.group(1)

    # basis
    if "sofr" in t_norm:
        out["Basis"] = "SOFR"
    elif "prime" in t_norm:
        out["Basis"] = "Prime"
    elif "libor" in t_norm:
        out["Basis"] = "LIBOR"
    elif "euribor" in t_norm:
        out["Basis"] = "EURIBOR"
    elif "sonia" in t_norm:
        out["Basis"] = "SONIA"

    # floor patterns
    m = re.search(r"floor\s*[:\-]?\s*([-+]?\d+(\.\d+)?)\s*%", t_norm)
    if m:
        out["Floor"] = m.group(1)
    m = re.search(r"\bsofr\w*\s*\(\s*([-+]?\d+(\.\d+)?)\s*%\s*\)", t_norm)
    if m and out["Floor"] is None:
        out["Floor"] = m.group(1)

    # margin/spread pattern (+ X%)
    m = re.search(r"(\+|plus)\s*([-+]?\d+(\.\d+)?)\s*%", t_norm)
    if m:
        out["Margin"] = m.group(2)

    # if the whole thing is a coupon like "7.25%"
    m = re.fullmatch(r"([-+]?\d+(\.\d+)?)\s*%", t_norm)
    if m:
        out["BDC Coupon Rate"] = m.group(1)

    # "cash + pik" style
    if "pik" in t_norm:
        # very rough: try to capture "... cash + X% pik"
        m = re.search(r"cash\s*\+\s*([-+]?\d+(\.\d+)?)\s*%\s*pik", t_norm)
        if m:
            out["PIK Margin"] = m.group(1)

    return out

# -------------------------
# COLUMN NORMALIZATION MAP
# -------------------------
# Each ION field can match multiple possible column names seen across BDCs.
SYNONYMS = {
    "Lead/Primary Borrower": [
        "portfolio company", "issuer", "company", "borrower", "investment", "name"
    ],
    "Additional Borrower": [
        "additional borrower", "co-borrower", "subsidiary", "guarantor"
    ],
    "Instrument type": [
        "security", "instrument", "type of investment", "investment type", "debt/equity", "loan type"
    ],
    "Industry": [
        "industry", "sector"
    ],
    "BDC Coupon Rate": [
        "interest rate", "coupon", "total coupon", "cash rate", "rate"
    ],
    "Pricing Date": [
        "pricing date", "initial acquisition date", "acquisition date", "issue date", "origination date"
    ],
    "Maturity Type": [
        "maturity", "maturity date", "due", "due date"
    ],
    "Principal Type": [
        "principal", "principal amount", "par", "par amount", "par amount / quantity", "quantity", "units", "shares"
    ],
    "Amortised Cost": [
        "amortized cost", "amortised cost", "cost", "cost of investments", "amortized"
    ],
    "Fair Value": [
        "fair value"
    ],
    "Currency Type": [
        "currency", "currency type"
    ],
    "Direct Lender": [
        "direct lender", "lender"
    ],
    "Seniority Ranking": [
        "seniority", "ranking", "lien"
    ],
    "Secured/Subordinated": [
        "secured", "subordinated", "unsecured"
    ],
    "Report Type": [
        "report type", "form type", "10-q", "10-k"
    ],
    "Source": [
        "source", "filing", "sec"
    ],
}

# Some fields are NOT direct columns, we derive them from others when possible:
DERIVED_FIELDS = {"Basis", "Floor", "Margin", "Cash Rate", "PIK Margin", "Implied coupon"}

def _best_match_column(header: List[str], ion_field: str) -> Optional[int]:
    """
    Simple rule-based matching. You can later upgrade this to fuzzy matching if needed.
    """
    targets = [_norm(x) for x in SYNONYMS.get(ion_field, [])]
    if not targets:
        return None
    header_norm = [_norm(h) for h in header]
    for i, h in enumerate(header_norm):
        for t in targets:
            if t and t in h:
                return i
    return None

def normalize_soi_table(
    raw_rows: List[List[str]],
    company: Optional[str] = None,
    report_date: Optional[str] = None,
    report_type: Optional[str] = None,
    source_url: Optional[str] = None,
) -> Dict[str, Any]:
    # 1) collapse spacer columns
    rows = _collapse_spacer_columns(raw_rows, min_keep_ratio=0.02)

    # 2) find header block (one or more header rows)
    header_rows = []
    header_idx = None
    for i, r in enumerate(rows[:60]):  # headers appear early in SOI section
        if _looks_like_header_row(r):
            header_idx = i
            header_rows.append(r)
            # include the next row too if it's also header-like (multi-line header)
            if i + 1 < len(rows) and _looks_like_header_row(rows[i + 1]):
                header_rows.append(rows[i + 1])
            break

    if header_idx is None:
        raise RuntimeError("Could not find SOI header row inside provided rows.")

    header = _merge_multiline_headers(header_rows)
    width = len(header)

    def pad(r):
        return (r + [""] * (width - len(r)))[:width]

    # 3) walk rows after header, build records
    records: List[Dict[str, Any]] = []
    current_section = None

    # precompute column index per ION field (direct columns)
    col_index = {f: _best_match_column(header, f) for f in ION_FIELDS}

    for r in rows[header_idx + len(header_rows):]:
        r = pad(r)

        if _looks_like_section_row(r):
            current_section = next((c for c in r if not _is_empty_cell(c)), None)
            continue

        # skip clearly empty rows
        if all(_is_empty_cell(x) for x in r):
            continue

        row_dict = {header[i]: (r[i] if not _is_empty_cell(r[i]) else None) for i in range(width)}

        # Heuristic: ignore non-data lines (like totals) unless they have a borrower/company
        borrower_idx = col_index.get("Lead/Primary Borrower")
        borrower_val = r[borrower_idx] if borrower_idx is not None and borrower_idx < len(r) else None
        if borrower_idx is None or _is_empty_cell(borrower_val):
            # still keep row if it has lots of numeric cells (data-like)
            numeric_cells = sum(1 for x in r if _parse_number(x) is not None)
            if numeric_cells < 3:
                continue

        rec = {k: None for k in ION_FIELDS}

        # section (not in ION fields list, but you probably want it)
        rec["_section"] = current_section

        # fill direct mappings
        for f in ION_FIELDS:
            idx = col_index.get(f)
            if idx is not None and idx < len(r):
                val = r[idx]
                rec[f] = val if not _is_empty_cell(val) else None

        # Derived: if "Currency Type" not present, infer
        if rec["Currency Type"] is None:
            rec["Currency Type"] = _infer_currency_from_row(row_dict)

        # Derived: parse rate components
        rate_text = rec.get("BDC Coupon Rate")
        if rate_text:
            comps = _extract_rate_components(rate_text)
            for k in DERIVED_FIELDS.union({"BDC Coupon Rate"}):
                if rec.get(k) is None and comps.get(k) is not None:
                    rec[k] = comps[k]

        # Numeric normalization for principal/cost/fair value (optional)
        for f in ["Principal Type", "Amortised Cost", "Fair Value"]:
            if rec.get(f) is not None:
                num = _parse_number(rec[f])
                if num is not None:
                    rec[f] = num

        # report metadata
        rec["Report Date"] = report_date
        rec["Report Type"] = report_type
        rec["Source"] = source_url

        records.append(rec)

    out = {
        "company": company,
        "table": "Schedule of Investments",
        "as_of": report_date,
        "currency": "USD" if any((r.get("Currency Type") == "USD") for r in records) else None,
        "records": records,
    }
    return out


# -------------------------
# Example usage (plug your extracted payload here)
# -------------------------
if __name__ == "__main__":
    # Load your current extracted JSON (the one with "table": {"columns":[...], "rows":[...]})
    with open("soi.json", "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_rows = payload["table"]["rows"]

    normalized = normalize_soi_table(
        raw_rows=raw_rows,
        company="Great Elm Capital Corp",     # you can also auto-extract from header text later
        report_date="2025-09-30",            # set from filing period / page header
        report_type="10-Q",
        source_url=payload.get("source_filing_url"),
    )

    with open("soi_normalized.json", "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)

    print("✅ wrote soi_normalized.json with", len(normalized["records"]), "records")
