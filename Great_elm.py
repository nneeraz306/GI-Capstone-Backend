import re
import io
import time
import json
import requests
import pandas as pd
from typing import List, Tuple
from collections import OrderedDict

# ---------------------------
# 1) CONFIG
# ---------------------------

TENQ_HTML_URL = "https://www.sec.gov/Archives/edgar/data/1675033/000119312525264873/gecc-20250930.htm"
USER_AGENT = "Kirthick Easwar B kirthick31@gmail.com"  # use your email

KEYWORDS_STRONG = [
    "schedule of investments",
    "system of investments",
    "schedule of investment",
    "investments (unaudited)",
    "portfolio of investments",
]

KEYWORDS_SUPPORT = [
    "fair value", "amortized cost", "cost", "principal", "par",
    "maturity", "coupon", "interest rate", "issuer", "industry",
    "secured", "unsecured", "senior", "subordinated",
    "total investments", "investments", "floating rate", "fixed rate"
]

SCHEDULE_HEADER_HINTS = [
    "portfolio company", "industry", "security", "interest rate",
    "maturity", "par amount", "quantity", "cost", "fair value",
    "percentage of class", "initial acquisition"
]

TOC_HINTS = [
    "part i.", "part ii.", "item 1.", "item 2.", "financial information",
    "legal proceedings", "signatures", "exhibits"
]

REQUEST_SLEEP_SECONDS = 0.4  # be polite to SEC servers

# ---------------------------
# 2) DOWNLOAD HTML (SEC-friendly)
# ---------------------------

def fetch_html(url: str, user_agent: str, retries: int = 3, timeout: int = 30) -> str:
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    }

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            time.sleep(REQUEST_SLEEP_SECONDS)
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            last_err = e
            time.sleep(1.0 * attempt)

    raise RuntimeError(f"Failed to fetch HTML after {retries} tries. Last error: {last_err}")

# ---------------------------
# 3) TABLE SCORING (find the right one)
# ---------------------------

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def table_to_text(df: pd.DataFrame, max_cells: int = 400) -> str:
    parts = []
    try:
        parts.append(" ".join([str(c) for c in df.columns]))
    except Exception:
        pass

    count = 0
    sample = df.astype(str).head(30)
    for row in sample.itertuples(index=False):
        for cell in row:
            parts.append(str(cell))
            count += 1
            if count >= max_cells:
                break
        if count >= max_cells:
            break

    return normalize_text(" ".join(parts))

def score_table(df: pd.DataFrame) -> int:
    txt = table_to_text(df)
    score = 0

    for k in KEYWORDS_STRONG:
        if k in txt:
            score += 60

    for k in KEYWORDS_SUPPORT:
        if k in txt:
            score += 6

    if df.shape[1] >= 6:
        score += 5
    if df.shape[1] >= 10:
        score += 8
    if df.shape[0] >= 25:
        score += 5
    if df.shape[0] >= 60:
        score += 8

    return score

def rank_tables(tables: List[pd.DataFrame], top_k: int = 8) -> List[Tuple[int, int]]:
    scored = [(i, score_table(t)) for i, t in enumerate(tables)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

def is_schedule_like(df: pd.DataFrame) -> bool:
    if df.shape[1] < 6:
        return False

    txt = table_to_text(df)
    hit_count = sum(1 for h in SCHEDULE_HEADER_HINTS if h in txt)
    toc_hits = sum(1 for t in TOC_HINTS if t in txt)
    return hit_count >= 4 and toc_hits <= 1

def pick_schedule_table_index(tables: List[pd.DataFrame]) -> int:
    candidates = []
    for i, t in enumerate(tables):
        if is_schedule_like(t):
            candidates.append((i, score_table(t)))

    if not candidates:
        wide = [(i, score_table(t)) for i, t in enumerate(tables) if t.shape[1] >= 6]
        wide.sort(key=lambda x: x[1], reverse=True)
        if not wide:
            raise ValueError("No tables found that look like Schedule of Investments.")
        return wide[0][0]

    # SEC filings list current period schedule FIRST, prior period second.
    # Among schedule-like candidates, prefer the first occurrence (lowest index).
    # This ensures we get the current period, not the prior-period comparison.
    candidates.sort(key=lambda x: x[0])  # sort by table index ascending
    return candidates[0][0]

# ---------------------------
# 4) MERGE: schedules are often split across multiple tables
# ---------------------------

def looks_like_continuation(prev: pd.DataFrame, nxt: pd.DataFrame) -> bool:
    # Both tables must be wide (schedule-like) and similar column count
    # This prevents merging narrow summary/totals tables into the schedule
    prev_cols = prev.shape[1]
    nxt_cols = nxt.shape[1]
    if nxt_cols < 15:
        return False  # summary/totals tables have few columns
    if abs(prev_cols - nxt_cols) <= 4:
        return True
    txt = table_to_text(nxt)
    if nxt_cols >= 15 and any(k in txt for k in ["fair value", "amortized cost", "principal", "maturity"]):
        return True
    return False

def merge_consecutive_tables(tables: List[pd.DataFrame], start_index: int, max_follow: int = 10) -> pd.DataFrame:
    parts = [tables[start_index]]
    for j in range(start_index + 1, min(len(tables), start_index + 1 + max_follow)):
        if looks_like_continuation(parts[-1], tables[j]):
            parts.append(tables[j])
        else:
            break
    return pd.concat(parts, ignore_index=True, sort=False)

# ---------------------------
# 5) CLEANING HELPERS
# ---------------------------

def drop_empty_rows_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.replace(r"^\s*$", pd.NA, regex=True, inplace=True)
    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    return df

def remove_repeated_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    header_text = normalize_text(" ".join([str(c) for c in df.columns]))

    def row_is_header_like(row) -> bool:
        row_text = normalize_text(" ".join([str(x) for x in row]))
        hits = sum(1 for k in KEYWORDS_SUPPORT if k in row_text)
        return (hits >= 3) or (row_text == header_text)

    # mark header-like rows
    mask = df.apply(lambda r: row_is_header_like(r.values), axis=1)

    # ✅ IMPORTANT: keep the FIRST header-like row (it’s usually the actual schedule header)
    if mask.any():
        first_header_idx = mask[mask].index[0]
        mask.loc[first_header_idx] = False

    return df[~mask]

def finalize_table(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_empty_rows_cols(df)
    df.columns = [str(c).strip() for c in df.columns]
    df = remove_repeated_header_rows(df)
    df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
    return df

# ---------------------------
# 6) EXTRACT SCHEDULE OF INVESTMENTS
# ---------------------------

def extract_schedule_of_investments(tenq_url: str) -> pd.DataFrame:
    html = fetch_html(tenq_url, USER_AGENT)
    tables = pd.read_html(io.BytesIO(html.encode("utf-8")), flavor="lxml")
    print(f"Found {len(tables)} tables.")

    top = rank_tables(tables, top_k=8)
    print("Top candidate tables (index, score):", top)

    best_idx = pick_schedule_table_index(tables)
    best_score = score_table(tables[best_idx])

    if best_score < 20:
        print("Warning: Low confidence detection. You may need to inspect candidate tables manually.")

    merged = merge_consecutive_tables(tables, best_idx, max_follow=10)
    cleaned = finalize_table(merged)
    return cleaned
# ---------------------------
# 7) ROBUST STRUCTURING (FIXED v2: header promote + fallback profiling)
# ---------------------------

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [" ".join([str(x) for x in tup if str(x) != "nan"]).strip() for tup in d.columns.values]
    else:
        d.columns = [str(c).strip() for c in d.columns]
    # replace empty col names
    d.columns = [c if c and c.lower() != "nan" else "unnamed" for c in d.columns]
    return d

def _find_header_row(d: pd.DataFrame, max_scan: int = 20):
    scan_n = min(max_scan, len(d))
    best_i, best_hits = None, 0

    for i in range(scan_n):
        # ✅ convert everything to string safely
        row = ["" if (x is None or (isinstance(x, float) and pd.isna(x))) else str(x) for x in d.iloc[i].tolist()]
        row_text = normalize_text(" ".join(row))

        hits = sum(1 for h in SCHEDULE_HEADER_HINTS if h in row_text)
        if hits > best_hits:
            best_hits = hits
            best_i = i

    if best_hits >= 3:  # relaxed threshold
        return best_i
    return None

def _norm_col(c: str) -> str:
    c = str(c).replace("\n", " ")
    c = re.sub(r"\(\d+\)", "", c)  # strip superscript markers like (1), (2), (3)
    c = re.sub(r"\s+", " ", c).strip().lower()
    return c

def _pick_col_by_keywords(cols: list[str], include: list[str], prefer: list[str] | None = None) -> str | None:
    best = None
    best_score = 0
    for c in cols:
        cc = _norm_col(c)
        score = 0
        for k in include:
            if k in cc:
                score += 3
        if prefer:
            for k in prefer:
                if k in cc:
                    score += 2
        if score > best_score:
            best_score = score
            best = c
    return best if best_score > 0 else None

def _col_profile_score(series: pd.Series, kind: str) -> int:
    """
    Score a column by what it "looks like" based on values.
    kinds: company, industry, security, notes, rate, date, pct, moneyint
    """
    s = series.dropna().astype(str).head(80)
    if s.empty:
        return 0

    def count_match(pat):
        return sum(bool(re.search(pat, x)) for x in s)

    if kind == "date":
        return count_match(r"\b\d{2}/\d{2}/\d{4}\b") * 5
    if kind == "pct":
        return count_match(r"%\b") * 4
    if kind == "moneyint":
        # lots of digits/commas, sometimes parentheses
        return count_match(r"^\(?-?[\d,]+(\.\d+)?\)?$") * 3
    if kind == "rate":
        return count_match(r"(sofr|libor|prime|%|\+)\b") * 3
    if kind == "notes":
        return count_match(r"^\d+(\s*,\s*\d+)*$") * 4
    if kind in ("company", "industry", "security"):
        # mainly text, not mostly digits
        texty = sum(1 for x in s if re.search(r"[A-Za-z]", x) and not re.fullmatch(r"[\d,().%\- ]+", x))
        return texty * 2
    return 0

def _fallback_map_by_profile(d: pd.DataFrame) -> dict:
    """
    If header-based matching fails, pick best columns by profiling.
    """
    cols = list(d.columns)

    scores = {c: {} for c in cols}
    for c in cols:
        col = d[c]
        scores[c]["company"] = _col_profile_score(col, "company")
        scores[c]["industry"] = _col_profile_score(col, "industry")
        scores[c]["security"] = _col_profile_score(col, "security")
        scores[c]["notes"] = _col_profile_score(col, "notes")
        scores[c]["rate"] = _col_profile_score(col, "rate")
        scores[c]["date"] = _col_profile_score(col, "date")
        scores[c]["pct"] = _col_profile_score(col, "pct")
        scores[c]["moneyint"] = _col_profile_score(col, "moneyint")

    def pick_best(kind, exclude=set()):
        best_c, best_v = None, -1
        for c in cols:
            if c in exclude:
                continue
            v = scores[c].get(kind, 0)
            if v > best_v:
                best_v = v
                best_c = c
        return best_c if best_v > 0 else None

    used = set()
    c_portfolio = pick_best("company", used); used.add(c_portfolio) if c_portfolio else None
    c_industry   = pick_best("industry", used); used.add(c_industry) if c_industry else None
    c_security   = pick_best("security", used); used.add(c_security) if c_security else None
    c_notes      = pick_best("notes", used); used.add(c_notes) if c_notes else None
    c_rate       = pick_best("rate", used); used.add(c_rate) if c_rate else None

    # dates: there are two date columns (acq + maturity). pick top 2 date-like columns
    date_cols = sorted([(c, scores[c]["date"]) for c in cols], key=lambda x: x[1], reverse=True)
    date_cols = [c for c,v in date_cols if v > 0]
    c_acq = date_cols[0] if len(date_cols) >= 1 else None
    c_mat = date_cols[1] if len(date_cols) >= 2 else None
    if c_acq: used.add(c_acq)
    if c_mat: used.add(c_mat)

    # numeric money columns: par, cost, fair value are three money-like columns
    money_cols = sorted([(c, scores[c]["moneyint"]) for c in cols], key=lambda x: x[1], reverse=True)
    money_cols = [c for c,v in money_cols if v > 0]

    c_par  = money_cols[0] if len(money_cols) >= 1 else None
    c_cost = money_cols[1] if len(money_cols) >= 2 else None
    c_fv   = money_cols[2] if len(money_cols) >= 3 else None
    if c_par: used.add(c_par)
    if c_cost: used.add(c_cost)
    if c_fv: used.add(c_fv)

    c_pct = pick_best("pct", used)

    return {
        "portfolio_company": c_portfolio,
        "industry": c_industry,
        "security": c_security,
        "notes": c_notes,
        "interest_rate": c_rate,
        "initial_acquisition_date": c_acq,
        "maturity": c_mat,
        "par_amount_or_quantity": c_par,
        "cost": c_cost,
        "fair_value": c_fv,
        "percentage_of_class": c_pct,
    }

def _clean_spacer_and_duplicate_cols(d: pd.DataFrame) -> pd.DataFrame:
    """
    SEC HTML tables have NaN spacer columns between every real column,
    and duplicate-named columns where one is empty and the other has data.
    This cleans both issues.
    """
    d = d.copy()
    d = d.replace(r"^\s*$", pd.NA, regex=True)

    # 1. Drop columns with NaN/empty/None names (spacer columns from HTML)
    valid_mask = [str(c).strip().lower() not in ("nan", "", "none") for c in d.columns]
    d = d.loc[:, valid_mask]

    # 2. Merge duplicate-named columns (coalesce: first non-null per row)
    if d.columns.duplicated().any():
        merged = {}
        seen_order = []
        for col_name in d.columns:
            if col_name in merged:
                # coalesce: fill NaN in first with values from this duplicate
                merged[col_name] = merged[col_name].fillna(d[col_name].iloc[:, -1] if isinstance(d[col_name], pd.DataFrame) else d[col_name])
            else:
                cols_with_name = d.loc[:, d.columns == col_name]
                if isinstance(cols_with_name, pd.DataFrame) and cols_with_name.shape[1] > 1:
                    # bfill across duplicates, take first
                    merged[col_name] = cols_with_name.bfill(axis=1).iloc[:, 0]
                else:
                    merged[col_name] = cols_with_name.iloc[:, 0] if isinstance(cols_with_name, pd.DataFrame) else cols_with_name
                seen_order.append(col_name)
        d = pd.DataFrame({c: merged[c] for c in seen_order})

    # 3. Drop any remaining all-NaN columns
    d = d.dropna(axis=1, how="all")
    return d


def build_structured_schedule(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # cleanup
    d = d.replace(r"^\s*$", pd.NA, regex=True)
    d = d.dropna(axis=1, how="all")
    d = d.dropna(axis=0, how="all")
    d = _flatten_columns(d)

    # 1) try to find a header row by hints
    header_i = _find_header_row(d)

    # 2) if not found, try "first non-empty row looks like header"
    if header_i is None:
        for i in range(min(10, len(d))):
            row_text = normalize_text(" ".join(d.iloc[i].astype(str).tolist()))
            if any(h in row_text for h in ["portfolio", "industry", "security", "interest", "maturity", "fair value", "cost", "par"]):
                header_i = i
                break

    # 3) promote header row if found
    if header_i is not None:
        new_cols = d.iloc[header_i].tolist()
        new_cols = [str(x).strip() if x is not None else "" for x in new_cols]
        d = d.iloc[header_i + 1:].copy()
        d.columns = new_cols
        d = _flatten_columns(d)

    # 3b) Clean spacer columns and merge duplicate-named columns
    d = _clean_spacer_and_duplicate_cols(d)

    # Also remove remaining repeated header rows that survived merge
    d = d.replace(r"^\s*$", pd.NA, regex=True)
    header_keywords = ["portfolio company", "industry", "security", "interest rate", "maturity", "par amount", "cost", "fair value", "percentage"]
    def _is_repeat_header(row):
        row_text = normalize_text(" ".join(str(x) for x in row))
        hits = sum(1 for k in header_keywords if k in row_text)
        return hits >= 4
    mask_hdr = d.apply(lambda r: _is_repeat_header(r.values), axis=1)
    d = d[~mask_hdr].reset_index(drop=True)

    # normalize duplicate column names
    cols = []
    seen = {}
    for c in d.columns:
        name = str(c).strip()
        if name == "" or name.lower() == "nan":
            name = "unnamed"
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}_{seen[name]}"
        cols.append(name)
    d.columns = cols

    colnames = list(d.columns)

    # A) header-based mapping
    mapped = {
        "portfolio_company": _pick_col_by_keywords(colnames, ["portfolio", "company"], ["portfolio company"]),
        "industry": _pick_col_by_keywords(colnames, ["industry"], ["industry"]),
        "security": _pick_col_by_keywords(colnames, ["security"], ["security"]),
        "notes": _pick_col_by_keywords(colnames, ["notes", "note"], ["notes"]),
        "interest_rate": _pick_col_by_keywords(colnames, ["interest", "rate", "coupon"], ["interest rate"]),
        "initial_acquisition_date": _pick_col_by_keywords(colnames, ["initial", "acquisition"], ["initial acquisition"]),
        "maturity": _pick_col_by_keywords(colnames, ["maturity"], ["maturity"]),
        "par_amount_or_quantity": _pick_col_by_keywords(colnames, ["par", "amount", "quantity", "principal"], ["par amount", "quantity"]),
        "cost": _pick_col_by_keywords(colnames, ["cost"], ["cost"]),
        "fair_value": _pick_col_by_keywords(colnames, ["fair", "value"], ["fair value"]),
        "percentage_of_class": _pick_col_by_keywords(colnames, ["percentage", "class"], ["percentage of class"]),
    }

    # If mapping completely failed, use fallback profiling
    if all(v is None for v in mapped.values()):
        mapped = _fallback_map_by_profile(d)

    missing = [k for k, v in mapped.items() if v is None]
    if missing:
        print("⚠️ Could not detect these columns (will be null):", missing)

    found = {k: v for k, v in mapped.items() if v is not None}
    print("✅ Detected column mapping:", found)
    print("📊 DataFrame columns after cleanup:", list(d.columns))

    keep_map = {k: v for k, v in mapped.items() if v is not None}

    if not keep_map:
        # hard stop with useful debug
        print("❌ Could not map any columns. Here are d.columns:")
        print(list(d.columns))
        print("Here are first 5 rows:")
        print(d.head(5).to_string(index=False))
        raise ValueError("Could not detect schedule columns. Inspect printed columns/rows above.")

    out = d[list(keep_map.values())].copy()
    out.columns = list(keep_map.keys())

    # clean whitespace and encoding artifacts
    def _clean_cell(x):
        if isinstance(x, str):
            x = x.strip()
            # Fix encoding artifacts: Â (from UTF-8 NBSP decoded as Latin-1)
            x = x.replace("\u00c2\u00a0", " ").replace("\u00c2", "").replace("\u00a0", " ")
            x = re.sub(r"\s+", " ", x).strip()
        return x
    out = out.apply(lambda col: col.map(_clean_cell))
    out = out.replace(r"^\s*$", pd.NA, regex=True)

    # Add section column safely (only if portfolio_company exists)
    section = None
    sections = []
    is_section_row = []

    if "portfolio_company" in out.columns:
        for _, row in out.iterrows():
            pc = row.get("portfolio_company")
            others_empty = row.drop(labels=["portfolio_company"], errors="ignore").isna().all()

            if isinstance(pc, str) and others_empty:
                txt = normalize_text(pc)
                if "investments" in txt and ("fair value" in txt or "cost" in txt or "controlled" in txt or "non-controlled" in txt):
                    section = pc
                    sections.append(section)
                    is_section_row.append(True)
                    continue

            sections.append(section)
            is_section_row.append(False)

        out.insert(0, "section", sections)
        out = out[~pd.Series(is_section_row, index=out.index)].reset_index(drop=True)

        # continuation forward-fill (only if cols exist)
        def row_has_any_data(r):
            check_cols = [c for c in out.columns if c not in ("section", "portfolio_company", "industry")]
            return r[check_cols].notna().any() if check_cols else False

        if "portfolio_company" in out.columns:
            mask_cont = out["portfolio_company"].isna() & out.apply(row_has_any_data, axis=1)
            out["portfolio_company"] = out["portfolio_company"].ffill()

        if "industry" in out.columns:
            mask_ind = out["industry"].isna() & out.apply(row_has_any_data, axis=1)
            out["industry"] = out["industry"].ffill()

    # final cleanup
    out = out.dropna(how="all").reset_index(drop=True)
    return out

# ---------------------------
# 8) DATAFRAME -> ORDERED JSON
# ---------------------------

def clean_colname(c: str) -> str:
    c = str(c).strip()
    c = re.sub(r"\s+", " ", c)
    c = c.replace("\n", " ")
    c = re.sub(r"[^A-Za-z0-9 _%()-]", "", c)
    c = c.replace(" ", "_")
    return c

def parse_number(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if s in {"", "—", "-", "–", "nan", "None"}:
        return None

    s2 = s.replace("$", "").replace(",", "")
    s2 = s2.replace("\u00a0", " ").strip()  # NBSP

    neg = False
    if re.match(r"^\(.*\)$", s2):
        neg = True
        s2 = s2[1:-1].strip()

    if s2.endswith("%") and len(s2) > 1:
        try:
            val = float(s2[:-1])
            return -val if neg else val
        except:
            return s

    # keep notes like "2, 16" as string (don’t force number)
    if re.fullmatch(r"\d+\s*,\s*\d+.*", s2):
        return s

    try:
        val = int(s2)
        return -val if neg else val
    except:
        try:
            val = float(s2)
            return -val if neg else val
        except:
            return s

def df_to_ordered_records(df: pd.DataFrame) -> list:
    df2 = df.copy()
    df2.columns = [clean_colname(c) for c in df2.columns]
    df2 = df2.where(pd.notnull(df2), None)

    ordered_cols = list(df2.columns)
    records = []

    for _, row in df2.iterrows():
        rec = OrderedDict()
        for col in ordered_cols:
            rec[col] = parse_number(row[col])
        if any(v not in (None, "", "—", "-", "–") for v in rec.values()):
            records.append(rec)

    return records

def write_schedule_json(
    df: pd.DataFrame,
    out_path: str,
    company: str = "Great Elm Capital Corp",
    as_of: str | None = None,
    currency: str = "USD"
):
    payload = OrderedDict()
    payload["company"] = company
    payload["table"] = "Schedule of Investments"
    payload["as_of"] = as_of
    payload["currency"] = currency
    payload["records"] = df_to_ordered_records(df)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"✅ Wrote ordered JSON: {out_path}")

# ---------------------------
# 9) RUN
# ---------------------------

if __name__ == "__main__":
    df_raw = extract_schedule_of_investments(TENQ_HTML_URL)

    # ✅ fixed structuring (your friend's working logic)
    df_structured = build_structured_schedule(df_raw)

    # ---- macOS output paths ----
    out_csv  = "/Users/kirthick21/Desktop/great_elm_schedule_structured.csv"
    out_xlsx = "/Users/kirthick21/Desktop/great_elm_schedule_structured.xlsx"
    out_json = "/Users/kirthick21/Desktop/great_elm_schedule_structured.json"

    # Save outputs
    df_structured.to_csv(out_csv, index=False)
    print(f"✅ Saved CSV: {out_csv}")

    try:
        df_structured.to_excel(out_xlsx, index=False)
        print(f"✅ Saved Excel: {out_xlsx}")
    except ModuleNotFoundError:
        print("⚠️ openpyxl not installed, skipping Excel")

    write_schedule_json(
        df_structured,
        out_json,
        company="Great Elm Capital Corp",
        as_of="2025-09-30",
        currency="USD"
    )
    print(f"✅ JSON saved at: {out_json}")

    # ---- open JSON on macOS (works even if VS Code CLI isn't installed) ----
    print("\nOpen JSON now using Finder Quick Look / default app:")
    print(f"open '{out_json}'")

    # If you later enable VS Code CLI, this will work:
    print("\nIf VS Code 'code' command is enabled, you can also run:")
    print(f"code '{out_json}'")

    print("\nPreview:")
    print(df_structured.head(15).to_string(index=False))
