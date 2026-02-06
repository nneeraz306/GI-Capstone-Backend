#!/usr/bin/env python3
"""
BDC SOI -> Normalized ION Lab Fields

INPUT:
  - a raw SOI extraction JSON in row/column format (like your current output)
    {
      "source_filing_url": "...",
      "soi_source_file_url": "...",
      "score": ...,
      "table": {"columns":[...], "rows":[ [...], [...], ... ]}
    }

OUTPUT:
  - normalized_ion.json  (records[] objects with ION fields; missing => null)
  - normalized_ion.csv   (flat csv)

KEY FIXES:
  - Handles spacer columns: Cost/Fair Value/Principal value columns found by scanning right
  - Handles multi-row headers by collapsing them
  - Detects section rows (e.g., "Investments at Fair Value") and attaches to subsequent rows
"""

import re
import json
import csv
from typing import List, Optional, Dict, Any, Tuple


# -----------------------------
# ION Required Output Schema
# -----------------------------
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
    "Section",  # (extra, useful)
]

# -----------------------------
# Helpers: normalize / parse
# -----------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _clean_cell(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _parse_number(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    t = str(s).strip()
    if t in ("", "-", "—"):
        return None
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg = True
        t = t[1:-1]
    t = t.replace("$", "").replace(",", "").strip()
    if t.endswith("%"):
        return None
    try:
        v = float(t)
        return -v if neg else v
    except:
        return None

def _parse_percent(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    t = str(s).strip()
    if t in ("", "-", "—"):
        return None
    # allow "7.0%" or "7.0 %"
    t = t.replace(" ", "")
    if t.endswith("%"):
        t = t[:-1]
    try:
        return float(t)
    except:
        return None

def _looks_like_date(s: str) -> bool:
    s = s.strip()
    # common BDC formats: 09/30/2025, 9/30/25, March 1, 2028
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", s):
        return True
    if re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b", s.lower()):
        return True
    return False

def _to_null_if_blank(x: Any) -> Optional[str]:
    s = _clean_cell(x)
    return s if s else None


# -----------------------------
# Header detection & collapsing
# -----------------------------
HEADER_HINTS = [
    "portfolio", "company", "issuer", "borrower", "industry", "sector",
    "security", "instrument", "investment",
    "principal", "par", "quantity", "shares", "units",
    "cost", "amortized", "amortised",
    "fair value", "maturity", "interest", "coupon", "rate", "spread", "margin", "floor", "basis",
    "%", "net assets",
    "notes", "ref", "reference",
]

def header_hit_count(row: List[str]) -> int:
    t = _norm(" ".join(row))
    return sum(1 for h in HEADER_HINTS if h in t)

def collapse_header_rows(rows: List[List[str]], max_header_rows: int = 4) -> Tuple[List[str], int]:
    """
    Find the best header start row and collapse up to max_header_rows into a single header list.
    Returns (header, header_end_idx) where header_end_idx is the index of last header row used.
    """
    best_i, best_hits = None, -1
    scan_n = min(80, len(rows))
    for i in range(scan_n):
        hits = header_hit_count(rows[i])
        if hits > best_hits:
            best_hits = hits
            best_i = i

    if best_i is None or best_hits < 3:
        # fallback: assume first row is header
        header0 = [_clean_cell(x) for x in rows[0]]
        return header0, 0

    # collapse best_i..best_i+k
    start = best_i
    end = min(len(rows) - 1, start + max_header_rows - 1)

    # choose how many rows to collapse: stop when next row is clearly data-like
    chosen_end = start
    for j in range(start, end + 1):
        chosen_end = j
        # if next row looks numeric-heavy, stop
        if j + 1 < len(rows):
            nxt = rows[j + 1]
            # data-like if contains numbers/dates and fewer header hits
            nxt_hits = header_hit_count(nxt)
            nxt_nums = sum(1 for c in nxt if _parse_number(_clean_cell(c)) is not None)
            nxt_dates = sum(1 for c in nxt if _looks_like_date(_clean_cell(c)))
            if nxt_hits <= 2 and (nxt_nums + nxt_dates) >= 2:
                break

    width = max(len(r) for r in rows[start:chosen_end + 1])
    header = []
    for col in range(width):
        parts = []
        for r in rows[start:chosen_end + 1]:
            if col < len(r):
                v = _clean_cell(r[col])
                if v:
                    parts.append(v)
        h = " ".join(parts).strip()
        header.append(h if h else f"col_{col+1}")
    return header, chosen_end


# -----------------------------
# Section row detection
# -----------------------------
def is_section_row(row: List[str]) -> bool:
    nonempty = [c for c in row if _clean_cell(c)]
    if len(nonempty) == 1:
        t = _norm(nonempty[0])
        # typical section labels
        if any(k in t for k in ["investments", "debt", "equity", "non-affiliate", "affiliate", "cash", "warrants"]):
            # avoid headers like "Portfolio Company"
            if header_hit_count(nonempty) < 2:
                return True
    return False


# -----------------------------
# Column mapping (normalization layer)
# -----------------------------
def find_header_col(header: List[str], contains_any: List[str]) -> Optional[int]:
    hn = [_norm(h) for h in header]
    for i, h in enumerate(hn):
        if any(k in h for k in contains_any):
            return i
    return None

def numeric_column_score(rows: List[List[str]], col_idx: int, sample_n: int = 120) -> float:
    data = rows[:sample_n]
    vals = [(_clean_cell(r[col_idx]) if col_idx < len(r) else "") for r in data]
    vals = [v for v in vals if v not in ("", "-", "—")]
    if not vals:
        return 0.0
    nums = sum(1 for v in vals if _parse_number(v) is not None)
    return nums / max(1, len(vals))

def find_numeric_value_col_to_right(
    header: List[str],
    data_rows: List[List[str]],
    label_col_idx: int,
    max_lookahead: int = 10,
    min_numeric_score: float = 0.20
) -> Optional[int]:
    best = None
    best_score = 0.0
    for j in range(label_col_idx, min(len(header), label_col_idx + max_lookahead + 1)):
        s = numeric_column_score(data_rows, j)
        if s > best_score:
            best_score = s
            best = j
        if s >= min_numeric_score:
            return j
    return best if best_score > 0 else None


# Header synonyms -> ION fields
# (You can extend these lists anytime without changing core code.)
SYNONYMS = {
    "Lead/Primary Borrower": ["portfolio company", "company", "issuer", "borrower", "investment", "portfolio"],
    "Instrument type": ["security", "instrument", "type of investment", "investment", "seniority", "loan", "bond", "revolver", "term loan"],
    "BDC Coupon Rate": ["interest rate", "coupon", "total coupon", "rate"],
    "Margin": ["margin", "spread"],
    "Floor": ["floor"],
    "Basis": ["basis", "index"],
    "Cash Rate": ["cash rate", "cash"],
    "PIK Margin": ["pik margin", "pik"],
    "Maturity Type": ["maturity", "maturity date"],
    "Pricing Date": ["initial acquisition date", "acquisition date", "pricing date"],
    "Principal Type": ["principal", "par amount", "par", "quantity", "principal amount"],
    "Amortised Cost": ["amortized cost", "amortised cost", "cost", "cost of investments"],
    "Fair Value": ["fair value"],
    "Report Type": ["report type", "10-q", "10-k", "n-q", "n-2", "10q", "10k"],
    "Currency Type": ["currency"],
    "Source": ["source", "edgar", "sec"],
    # "Direct Lender" is often the BDC itself; fill from metadata if you have it externally
    # "Additional Borrower", "Business Description", "SIG/NAICS" often not in SOI -> null
}

# Fields that must map to numeric columns (often have spacer columns)
NUMERIC_FIELDS = {"Principal Type", "Amortised Cost", "Fair Value"}


def build_col_map(header: List[str], data_rows: List[List[str]]) -> Dict[str, Optional[int]]:
    col_map: Dict[str, Optional[int]] = {}

    # 1) simple string match for most fields
    for out_field, keys in SYNONYMS.items():
        idx = find_header_col(header, [_norm(k) for k in keys])
        col_map[out_field] = idx

    # 2) robust numeric mapping: find actual numeric column to right
    for nf in NUMERIC_FIELDS:
        label_idx = col_map.get(nf)
        if label_idx is None:
            continue
        value_idx = find_numeric_value_col_to_right(header, data_rows, label_idx)
        col_map[nf] = value_idx

    return col_map


# -----------------------------
# Record building
# -----------------------------
def is_data_row(row: List[str]) -> bool:
    # data rows usually have a company/issuer text and at least one numeric/date
    txt_cells = sum(1 for c in row if _clean_cell(c) and _parse_number(_clean_cell(c)) is None)
    num_cells = sum(1 for c in row if _parse_number(_clean_cell(c)) is not None)
    date_cells = sum(1 for c in row if _looks_like_date(_clean_cell(c)))
    return (txt_cells >= 1) and ((num_cells + date_cells) >= 1)

def get_cell(row: List[str], idx: Optional[int]) -> Optional[str]:
    if idx is None:
        return None
    if idx < 0 or idx >= len(row):
        return None
    return _to_null_if_blank(row[idx])

def normalize_records(
    raw_rows: List[List[str]],
    header: List[str],
    header_end_idx: int,
    source_url: str,
    report_date: Optional[str] = None,
    report_type: Optional[str] = None,
    direct_lender: Optional[str] = None,
) -> List[Dict[str, Any]]:
    data_rows = raw_rows[header_end_idx + 1:]
    # clean row widths
    width = len(header)
    data_rows = [([_clean_cell(x) for x in r] + [""] * (width - len(r)))[:width] for r in data_rows]

    col_map = build_col_map(header, data_rows)

    records: List[Dict[str, Any]] = []
    current_section: Optional[str] = None

    for r in data_rows:
        if is_section_row(r):
            current_section = _to_null_if_blank(next(c for c in r if _clean_cell(c)))
            continue

        if not is_data_row(r):
            continue

        rec: Dict[str, Any] = {k: None for k in ION_FIELDS}
        rec["Source"] = source_url
        rec["Report Date"] = report_date
        rec["Report Type"] = report_type
        rec["Direct Lender"] = direct_lender
        rec["Section"] = current_section

        # borrower / issuer
        rec["Lead/Primary Borrower"] = get_cell(r, col_map.get("Lead/Primary Borrower"))

        # instrument
        rec["Instrument type"] = get_cell(r, col_map.get("Instrument type"))

        # rates / terms
        rec["BDC Coupon Rate"] = get_cell(r, col_map.get("BDC Coupon Rate"))
        rec["Pricing Date"] = get_cell(r, col_map.get("Pricing Date"))
        rec["Maturity Type"] = get_cell(r, col_map.get("Maturity Type"))
        rec["Basis"] = get_cell(r, col_map.get("Basis"))
        rec["Floor"] = get_cell(r, col_map.get("Floor"))
        rec["Margin"] = get_cell(r, col_map.get("Margin"))
        rec["Cash Rate"] = get_cell(r, col_map.get("Cash Rate"))
        rec["PIK Margin"] = get_cell(r, col_map.get("PIK Margin"))

        # numeric fields (spacer-safe mapping already done)
        p = get_cell(r, col_map.get("Principal Type"))
        c = get_cell(r, col_map.get("Amortised Cost"))
        fv = get_cell(r, col_map.get("Fair Value"))

        rec["Principal Type"] = _parse_number(p) if p is not None else None
        rec["Amortised Cost"] = _parse_number(c) if c is not None else None
        rec["Fair Value"] = _parse_number(fv) if fv is not None else None

        records.append(rec)

    return records


# -----------------------------
# IO
# -----------------------------
def load_raw_soi_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def write_csv(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ION_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)

def main(
    raw_soi_json_path: str = "soi.json",
    out_json: str = "normalized_ion.json",
    out_csv: str = "normalized_ion.csv",
    report_date: Optional[str] = None,
    report_type: Optional[str] = None,
    direct_lender: Optional[str] = None,
) -> None:
    raw = load_raw_soi_json(raw_soi_json_path)
    source_url = raw.get("soi_source_file_url") or raw.get("source_filing_url") or ""
    

    table = raw.get("table")

    if isinstance(table, dict):
        rows = table.get("rows", [])
    elif isinstance(table, list):
        # already rows
        rows = table
    else:
        raise RuntimeError(
            f"Unexpected table format: {type(table)}. "
            "Expected dict with {columns, rows} or list of rows."
        )
    rows = [[_clean_cell(x) for x in r] for r in rows if isinstance(r, list)]

    if not rows:
        raise RuntimeError("No rows found in input JSON. Check your SOI extraction step.")

    header, header_end_idx = collapse_header_rows(rows, max_header_rows=4)

    records = normalize_records(
        raw_rows=rows,
        header=header,
        header_end_idx=header_end_idx,
        source_url=source_url,
        report_date=report_date,
        report_type=report_type,
        direct_lender=direct_lender,
    )

    payload = {
        "source_filing_url": raw.get("source_filing_url"),
        "soi_source_file_url": raw.get("soi_source_file_url"),
        "header_used": header,
        "header_end_row_index": header_end_idx,
        "record_count": len(records),
        "records": records,
    }

    write_json(out_json, payload)
    write_csv(out_csv, records)

    print("✅ Normalization complete")
    print("Input:", raw_soi_json_path)
    print("Records:", len(records))
    print("Saved:", out_json, out_csv)


if __name__ == "__main__":
    # Example usage:
    #   python3.11 normalize_ion.py
    #   python3.11 normalize_ion.py
    # If you want, hardcode these from your pipeline metadata:
    # report_date="2025-09-30", report_type="10-Q", direct_lender="Great Elm Capital Corp"
    main(
        raw_soi_json_path="/Users/kirthick21/Desktop/great_elm_schedule_structured.json",
        out_json="normalized_ion.json",
        out_csv="normalized_ion.csv",
        report_date=None,
        report_type=None,
        direct_lender=None,
    )
