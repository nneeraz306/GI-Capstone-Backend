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

TENQ_HTML_URL = "https://www.sec.gov/Archives/edgar/data/0001925309/000119312525266848/ck0001925309-20250930.htm"
USER_AGENT = "Kirthick Easwar B kirthick31@gmail.com"  # use your email

COMPANY_NAME = "Sixth Street Lending Partners"
AS_OF_DATE   = "2025-09-30"

KEYWORDS_STRONG = [
    "schedule of investments",
    "consolidated schedule of investments",
    "schedule of investment",
    "investments (unaudited)",
    "portfolio of investments",
]

KEYWORDS_SUPPORT = [
    "fair value", "amortized cost", "cost", "principal", "par",
    "maturity", "coupon", "interest rate", "spread",
    "first-lien", "second-lien", "secured", "unsecured",
    "total investments", "investments", "debt investments",
    "equity investments", "reference rate", "net assets",
]

SCHEDULE_HEADER_HINTS = [
    "company", "investment", "initial acquisition", "reference rate",
    "spread", "interest rate", "amortized cost", "fair value",
    "percentage", "net assets",
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
    if df.shape[1] < 15:
        return False

    txt = table_to_text(df)
    hit_count = sum(1 for h in SCHEDULE_HEADER_HINTS if h in txt)
    toc_hits = sum(1 for t in TOC_HINTS if t in txt)
    return hit_count >= 4 and toc_hits <= 1

def find_all_schedule_indices(tables: List[pd.DataFrame]) -> List[int]:
    """Find ALL schedule-of-investments table indices (both periods)."""
    candidates = []
    for i, t in enumerate(tables):
        if t.shape[1] < 15:
            continue
        if is_schedule_like(t):
            candidates.append(i)

    if not candidates:
        # fallback: find wide tables with investment keywords
        for i, t in enumerate(tables):
            if t.shape[1] >= 15:
                txt = table_to_text(t)
                if any(k in txt for k in ["debt investments", "equity investments",
                                           "amortized cost", "fair value"]):
                    candidates.append(i)

    if not candidates:
        raise ValueError("No tables found that look like Schedule of Investments.")
    return candidates

def group_schedule_tables(candidates: List[int]) -> List[List[int]]:
    """Group candidates by proximity (gap > 5 = new group)."""
    if not candidates:
        return []
    groups = [[candidates[0]]]
    for i in range(1, len(candidates)):
        if candidates[i] - candidates[i - 1] <= 5:
            groups[-1].append(candidates[i])
        else:
            groups.append([candidates[i]])
    return groups

def pick_current_period_group(tables: List[pd.DataFrame], groups: List[List[int]]) -> List[int]:
    """
    SEC filings contain current-period and prior-period schedules.
    Identify the current-period group by looking for the AS_OF_DATE
    in the context tables near each group.
    Fallback: pick the FIRST group (SEC 10-Q filings typically put
    the current period schedule first, followed by the prior period).
    """
    if len(groups) == 1:
        return groups[0]

    # Try to find AS_OF_DATE context near each group
    # Parse AS_OF_DATE into searchable date strings
    from datetime import datetime
    try:
        dt = datetime.strptime(AS_OF_DATE, "%Y-%m-%d")
        date_patterns = [
            dt.strftime("%B %d, %Y"),          # "September 30, 2025"
            dt.strftime("%B %d,  %Y"),         # "September 30,  2025" (double space)
            dt.strftime("%b %d, %Y"),          # "Sep 30, 2025"
            dt.strftime("%b. %d, %Y"),         # "Sep. 30, 2025"
            dt.strftime("%m/%d/%Y"),           # "09/30/2025"
            dt.strftime("%-m/%-d/%Y"),         # "9/30/2025"
            AS_OF_DATE,                         # "2025-09-30"
            dt.strftime("%B %d, %Y").replace(" 0", " "),  # strip leading zero from day
        ]
    except Exception:
        date_patterns = [AS_OF_DATE]

    # For each group, search the 5 tables before the group's first index
    # AND the first few tables in the group itself for a period date reference
    group_scores = {}
    for gi, group in enumerate(groups):
        first_idx = group[0]
        score = 0
        # Scan tables before and within this group for date context
        check_range = list(range(max(0, first_idx - 5), first_idx)) + group[:3]
        for check_idx in check_range:
            if check_idx >= len(tables):
                continue
            t = tables[check_idx]
            txt = t.astype(str).to_string().lower()
            for pat in date_patterns:
                if pat.lower() in txt:
                    score += 1
                    print(f"  Period match: '{pat}' found near group {gi} (table {check_idx})")
        group_scores[gi] = score

    # Pick the group with the most date matches
    if any(v > 0 for v in group_scores.values()):
        best_group = max(group_scores, key=group_scores.get)
        print(f"  Selected group {best_group} with {group_scores[best_group]} date matches")
        return groups[best_group]

    # Fallback: first group is typically the current period in SEC filings
    print("  No period date match found; defaulting to first group (current period).")
    return groups[0]

# ---------------------------
# 4) MERGE: schedules are often split across multiple tables
# ---------------------------

def looks_like_continuation(prev: pd.DataFrame, nxt: pd.DataFrame) -> bool:
    prev_cols = prev.shape[1]
    nxt_cols = nxt.shape[1]
    if nxt_cols < 15:
        return False
    if abs(prev_cols - nxt_cols) <= 4:
        return True
    txt = table_to_text(nxt)
    if nxt_cols >= 15 and any(k in txt for k in ["fair value", "amortized cost", "principal", "maturity"]):
        return True
    return False

def merge_schedule_tables(tables: List[pd.DataFrame], indices: List[int],
                          max_extend: int = 4) -> pd.DataFrame:
    """
    Merge schedule tables by normalizing each individually first.
    Tables may have different column counts (debt=25, equity=23),
    so we promote headers per-table, then concat by column name.
    """
    # Start with identified indices, extend for adjacent tables
    all_indices = set(indices)
    last_idx = max(indices)
    for j in range(last_idx + 1, min(len(tables), last_idx + 1 + max_extend)):
        t = tables[j]
        if t.shape[1] < 15:
            continue
        if looks_like_continuation(tables[last_idx], t):
            txt = table_to_text(t)
            toc_hits = sum(1 for tc in TOC_HINTS if tc in txt)
            # Also check if this table belongs to a different period (prior period schedule)
            # by looking for date patterns that DON'T match our target AS_OF_DATE
            is_different_period = False
            full_txt = t.astype(str).to_string().lower()
            if "schedule of investments" in full_txt or "consolidated schedule" in full_txt:
                # This looks like a NEW schedule header, likely a different period
                from datetime import datetime
                try:
                    dt = datetime.strptime(AS_OF_DATE, "%Y-%m-%d")
                    target_date_str = dt.strftime("%B %d, %Y").lower()
                    # If we find a schedule header but NOT our target date nearby, skip
                    if target_date_str not in full_txt:
                        is_different_period = True
                        print(f"  Skipping table {j} (appears to be from a different period)")
                except Exception:
                    pass
            if toc_hits <= 1 and not is_different_period:
                all_indices.add(j)
                last_idx = j

    sorted_indices = sorted(all_indices)
    print(f"  Candidate tables: {sorted_indices}")

    normalized = []
    raw_colnames = None  # RAW column names (before spacer cleanup) from first header-bearing table

    for idx in sorted_indices:
        t = tables[idx].copy()
        txt = table_to_text(t)

        # Skip swap tables entirely
        if "interest rate swap" in txt and "company receives" in txt:
            print(f"  Skipping table {idx} (Interest Rate Swaps)")
            continue

        norm, raw_cn = _normalize_one_table(t, idx, raw_colnames)
        if raw_cn is not None and raw_colnames is None:
            raw_colnames = raw_cn
        if norm is not None and not norm.empty:
            normalized.append(norm)
            print(f"  Table {idx}: {norm.shape}, cols={list(norm.columns)[:6]}...")

    if not normalized:
        raise ValueError("No schedule tables found after normalization.")

    # Concat by column name (not position)
    combined = pd.concat(normalized, ignore_index=True, sort=False)
    return combined




def _normalize_one_table(t: pd.DataFrame, table_idx: int, raw_colnames: list = None) -> Tuple[pd.DataFrame, list]:
    d = t.copy()
    d = d.replace(r"^\s*$", pd.NA, regex=True)
    d = _flatten_columns(d)

    returned_raw = None

    # Find header row
    header_i = _find_header_row(d)

    if header_i is None:
        for i in range(min(10, len(d))):
            # ✅ safe join (no float crash)
            row_list = ["" if (x is None or (isinstance(x, float) and pd.isna(x))) else str(x) for x in d.iloc[i].tolist()]
            row_text = normalize_text(" ".join(row_list))

            fallback_keywords = ["company", "investment", "acquisition",
                                 "interest rate", "amortized cost",
                                 "fair value", "reference rate"]
            fallback_hits = sum(1 for h in fallback_keywords if h in row_text)
            if fallback_hits >= 2:
                header_i = i
                break

    if header_i is not None:
        new_cols = d.iloc[header_i].tolist()
        new_cols = [str(x).strip() if x is not None and not (isinstance(x, float) and pd.isna(x)) else "" for x in new_cols]
        d = d.iloc[header_i + 1:].copy()
        d.columns = new_cols
        d = _flatten_columns(d)
        returned_raw = list(d.columns)  # RAW col names before cleaning
        d = _clean_spacer_and_duplicate_cols(d)

    elif raw_colnames is not None and d.shape[1] == len(raw_colnames):
        d.columns = raw_colnames
        d = _clean_spacer_and_duplicate_cols(d)

    else:
        # equity / other differently-shaped table fallback (keep your existing logic)
        is_equity = False
        for i in range(min(5, len(d))):
            row_list = ["" if (x is None or (isinstance(x, float) and pd.isna(x))) else str(x) for x in d.iloc[i].tolist()]
            row_text = normalize_text(" ".join(row_list))
            if "equity and other" in row_text or "equity investments" in row_text:
                is_equity = True
                break

        if is_equity:
            ncols = d.shape[1]
            new_names = [f"col_{i}" for i in range(ncols)]
            if ncols >= 23:
                new_names[0] = "Company (1)(9)"
                if ncols > 2:
                    new_names[2] = "Investment"
                if ncols > 4:
                    new_names[4] = "Initial Acquisition Date"
                for c in [12, 13]:
                    if c < ncols:
                        new_names[c] = "Amortized Cost (2) (7)"
                for c in [16, 17]:
                    if c < ncols:
                        new_names[c] = "Fair Value (6)"
                for c in [21, 22]:
                    if c < ncols:
                        new_names[c] = "Percentage of Net Assets"
            d.columns = new_names
            d = _clean_spacer_and_duplicate_cols(d)
        else:
            d = _clean_spacer_and_duplicate_cols(d)

    d = d.dropna(axis=0, how="all").reset_index(drop=True)
    return d, returned_raw

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

    # 1) Find all schedule-like tables
    all_schedule = find_all_schedule_indices(tables)
    print(f"Schedule-like table indices: {all_schedule}")

    # 2) Group by proximity and pick current-period group
    groups = group_schedule_tables(all_schedule)
    print(f"Table groups: {groups}")
    current_group = pick_current_period_group(tables, groups)
    print(f"Current period group: {current_group}")

    # 3) Merge tables (normalize each individually, then concat by column name)
    merged = merge_schedule_tables(tables, current_group, max_extend=4)
    cleaned = finalize_table(merged)
    print(f"Merged schedule shape: {cleaned.shape}")
    print(f"Merged columns: {list(cleaned.columns)}")
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

def _find_header_row(d: pd.DataFrame, max_scan: int = 20) -> int | None:
    scan_n = min(max_scan, len(d))
    best_i, best_hits = None, 0

    for i in range(scan_n):
        # ✅ force everything to string safely (NaN -> "")
        row = ["" if (x is None or (isinstance(x, float) and pd.isna(x))) else str(x) for x in d.iloc[i].tolist()]
        row_text = normalize_text(" ".join(row))

        hits = sum(1 for h in SCHEDULE_HEADER_HINTS if h in row_text)
        if hits > best_hits:
            best_hits = hits
            best_i = i

    return best_i if best_hits >= 3 else None

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

    # 0. Treat lone currency/percent symbols as NaN (they appear in separate
    #    cells beside the actual numbers in SEC HTML tables)
    d = d.replace(r"^\s*[\$€£¥%]\s*$", pd.NA, regex=True)

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

def _pick_best_percent_col(d: pd.DataFrame, cols: list[str]) -> str | None:
    """
    Choose the best % column by looking at values, not just headers.
    Prefers columns with lots of values like:
      7.0, 7.0%, 0.3, 12.5, etc (and mostly within 0..100)
    """
    if not cols:
        return None

    def is_pct_token(x: str) -> bool:
        t = str(x).strip()
        if t in {"", "—", "-", "–", "nan", "None"}:
            return False
        t = t.replace("\u00a0", " ").strip()
        t = t.replace(" %", "%").replace("% ", "%")
        # allow "7.0" or "7.0%" or "(7.0)" style
        t2 = re.sub(r"[()%]", "", t).strip()
        return bool(re.fullmatch(r"-?\d+(\.\d+)?", t2))

    def to_float_loose(x: str) -> float | None:
        t = str(x).strip()
        t = t.replace("\u00a0", " ").strip()
        t = t.replace(" %", "%").replace("% ", "%")
        neg = False
        if re.fullmatch(r"\(.*\)", t):
            neg = True
            t = t[1:-1].strip()
        t = t.replace("%", "")
        t = t.replace(",", "")
        try:
            v = float(t)
            return -v if neg else v
        except:
            return None

    best_col = None
    best_score = -1

    for c in cols:
        s = d[c].dropna().astype(str).head(300)
        if s.empty:
            continue

        pct_like = sum(is_pct_token(v) for v in s)
        vals = [to_float_loose(v) for v in s]
        vals = [v for v in vals if v is not None]

        # prefer typical % ranges
        in_range = sum(0 <= v <= 100 for v in vals)
        out_range = sum((v < 0 or v > 100) for v in vals)

        # many SEC schedules have 1 decimal; give small bonus
        one_decimal = sum(bool(re.fullmatch(r"-?\d+\.\d", re.sub(r"[()%\s]", "", str(v)))) for v in s)

        score = pct_like * 5 + in_range * 2 + one_decimal - out_range * 5

        if score > best_score:
            best_score = score
            best_col = c

    return best_col
def build_structured_schedule(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # cleanup
    d = d.replace(r"^\s*$", pd.NA, regex=True)
    d = d.dropna(axis=1, how="all")
    d = d.dropna(axis=0, how="all")
    d = _flatten_columns(d)

    # Check if columns are already named from per-table normalization
    named_cols = [c for c in d.columns if _norm_col(c) not in ("unnamed", "nan", "")]
    has_named_cols = any(h in _norm_col(" ".join(named_cols))
                         for h in ["company", "investment", "amortized cost",
                                   "fair value", "reference rate", "interest rate"])

    if not has_named_cols:
        # 1) Find header row (only needed if columns aren't already named)
        header_i = _find_header_row(d)

        if header_i is None:
            for i in range(min(10, len(d))):
                row_text = normalize_text(" ".join(d.iloc[i].astype(str).tolist()))
                if any(h in row_text for h in ["company", "investment", "acquisition",
                                                "interest rate", "amortized cost",
                                                "fair value", "reference rate"]):
                    header_i = i
                    break

        # 2) Promote header row
        if header_i is not None:
            new_cols = d.iloc[header_i].tolist()
            new_cols = [str(x).strip() if x is not None else "" for x in new_cols]
            d = d.iloc[header_i + 1:].copy()
            d.columns = new_cols
            d = _flatten_columns(d)

        # 3) Clean spacers and duplicates
        d = _clean_spacer_and_duplicate_cols(d)
    else:
        # Columns already named from per-table normalization; just clean up
        d = _clean_spacer_and_duplicate_cols(d)

    # 4) Remove repeated header rows from merged tables
    d = d.replace(r"^\s*$", pd.NA, regex=True)
    header_keywords = ["company", "investment", "acquisition", "interest rate",
                       "amortized cost", "fair value", "reference rate", "percentage"]

    def _is_repeat_header(row):
        row_text = normalize_text(" ".join(str(x) for x in row))
        hits = sum(1 for k in header_keywords if k in row_text)
        return hits >= 4

    mask_hdr = d.apply(lambda r: _is_repeat_header(r.values), axis=1)
    d = d[~mask_hdr].reset_index(drop=True)

    # 5) Normalize duplicate column names
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
    print("📊 Columns after cleanup:", colnames)

    # 6) Map columns to our schema (Sixth Street layout)
    mapped = {
    "company": _pick_col_by_keywords(
        colnames,
        include=["company", "portfolio"],
        prefer=["company", "portfolio company"]
    ),

    "investment": _pick_col_by_keywords(
        colnames,
        include=["investment", "security", "issuer"],
        prefer=["investment", "security"]
    ),

    "initial_acquisition_date": _pick_col_by_keywords(
        colnames,
        include=["initial", "acquisition", "acquired"],
        prefer=["initial acquisition", "acquisition date"]
    ),

    "reference_rate_and_spread": _pick_col_by_keywords(
        colnames,
        include=["reference", "spread"],
        prefer=["reference rate", "reference rate and spread", "spread"]
    ),

    "interest_rate": _pick_col_by_keywords(
        colnames,
        include=["interest rate", "coupon"],
        prefer=["interest rate", "coupon"]
    ),

    # ✅ cost can be "Amortized Cost" in many BDC filings
    "amortized_cost": (
        _pick_col_by_keywords(colnames, include=["amortized", "cost"], prefer=["amortized cost"])
        or _pick_col_by_keywords(colnames, include=["cost"], prefer=["cost"])
    ),

    "fair_value": _pick_col_by_keywords(
        colnames,
        include=["fair", "value"],
        prefer=["fair value"]
    ),
}

# ✅ pick percent column by VALUES (avoids wrong adjacent column => 6.9 instead of 7.0)
    pct_candidates = [
        c for c in colnames
        if any(k in _norm_col(c) for k in ["percent", "percentage", "net assets", "class"])
    ]
    mapped["percentage_of_net_assets"] = _pick_best_percent_col(d, pct_candidates)

    # Avoid mapping same column to both reference_rate and interest_rate
    if mapped.get("reference_rate_and_spread") and mapped.get("interest_rate"):
        if mapped["reference_rate_and_spread"] == mapped["interest_rate"]:
            remaining = [c for c in colnames if c != mapped["reference_rate_and_spread"]]
            mapped["interest_rate"] = _pick_col_by_keywords(remaining, ["interest", "rate"], ["interest rate"])

    # If header-based mapping completely failed, try fallback profiling
    if all(v is None for v in mapped.values()):
        mapped = _fallback_map_by_profile(d)

    missing = [k for k, v in mapped.items() if v is None]
    if missing:
        print("⚠️ Could not detect these columns (will be null):", missing)

    found = {k: v for k, v in mapped.items() if v is not None}
    print("✅ Detected column mapping:", found)

    keep_map = {k: v for k, v in mapped.items() if v is not None}

    if not keep_map:
        print("❌ Could not map any columns. Columns:", list(d.columns))
        print(d.head(5).to_string(index=False))
        raise ValueError("Could not detect schedule columns.")

    out = d[list(keep_map.values())].copy()
    out.columns = list(keep_map.keys())

    # 7) Clean cell encoding artifacts and $ symbols
    def _clean_cell(x):
        if isinstance(x, str):
            x = x.strip()
            x = x.replace("\u00c2\u00a0", " ").replace("\u00c2", "").replace("\u00a0", " ")
            x = re.sub(r"\s+", " ", x).strip()
        return x
    out = out.apply(lambda col: col.map(_clean_cell))
    out = out.replace(r"^\s*$", pd.NA, regex=True)

    # 8) Detect section and industry from section-header rows
    #    Sixth Street uses "Debt Investments" / "Equity and Other Investments" as sections,
    #    and industry names (Automotive, Business Services, etc.) as row-level sub-headers.
    section = None
    industry = None
    sections = []
    industries = []
    is_section_row = []

    SKIP_ROW_KEYWORDS = ["total debt", "total equity", "total investments",
                          "total hedge", "cash collateral", "net assets",
                          "net unrealized", "total fair value", "subtotal"]

    if "company" in out.columns:
        for _, row in out.iterrows():
            company_val = row.get("company")
            others_empty = row.drop(labels=["company"], errors="ignore").isna().all()

            if isinstance(company_val, str) and others_empty:
                txt = normalize_text(company_val)

                # Skip total/summary rows
                if any(k in txt for k in SKIP_ROW_KEYWORDS):
                    sections.append(section)
                    industries.append(industry)
                    is_section_row.append(True)
                    continue

                # Section header (Debt Investments, Equity and Other Investments, etc.)
                if any(k in txt for k in ["debt investments", "equity and other",
                                           "equity investments", "hedge instruments"]):
                    section = company_val
                    industry = None  # reset industry for new section
                    sections.append(section)
                    industries.append(industry)
                    is_section_row.append(True)
                    continue

                # Industry sub-header (Automotive, Business Services, Chemicals, etc.)
                # These are short text labels with no digits in the first 20 chars
                if len(txt) < 60 and not any(c.isdigit() for c in txt[:20]):
                    industry = company_val
                    sections.append(section)
                    industries.append(industry)
                    is_section_row.append(True)
                    continue

            sections.append(section)
            industries.append(industry)
            is_section_row.append(False)

        out.insert(0, "section", sections)
        out.insert(1, "industry", industries)
        out = out[~pd.Series(is_section_row, index=out.index)].reset_index(drop=True)

        # Forward-fill company for continuation rows
        def row_has_any_data(r):
            check_cols = [c for c in out.columns if c not in ("section", "industry", "company")]
            return r[check_cols].notna().any() if check_cols else False

        if "company" in out.columns:
            out["company"] = out["company"].where(
                out["company"].notna() | ~out.apply(row_has_any_data, axis=1),
                other=pd.NA
            )
            out["company"] = out["company"].ffill()

        if "industry" in out.columns:
            out["industry"] = out["industry"].ffill()

    # 9) Remove total/subtotal rows
    if "company" in out.columns:
        total_mask = out["company"].astype(str).str.lower().str.contains(
            r"^total\b|^subtotal\b|^net assets\b|^cash collateral\b|^net unrealized",
            na=False
        )
        out = out[~total_mask].reset_index(drop=True)

    # 10) Final cleanup
    out = out.dropna(how="all").reset_index(drop=True)

    # Drop rows where all data columns are null
    data_cols = [c for c in out.columns if c not in ("section", "industry", "company")]
    if data_cols:
        only_labels = out[data_cols].isna().all(axis=1)
        out = out[~only_labels].reset_index(drop=True)

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

def _normalize_percent_string(s: str) -> str | None:
    """
    Keep percent EXACTLY like filing display.
    - Accept: "7.0", "7.0%", " 7.0 % ", "7", "7 %"
    - Return canonical: "7.0%" or "7%"
    """
    if s is None:
        return None
    t = str(s).strip()
    if t in {"", "—", "-", "–", "nan", "None"}:
        return None

    # remove NBSP and spaces around %
    t = t.replace("\u00a0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    t = t.replace(" %", "%").replace("% ", "%")

    # If it's just a bare number, keep as-is but add %
    if re.fullmatch(r"-?\d+(\.\d+)?", t):
        return t + "%"

    # If it's already a percent number, keep exactly
    if re.fullmatch(r"-?\d+(\.\d+)?%", t):
        return t

    # Otherwise keep original string (don’t “float” it)
    return t


def parse_number(x, col_name: str | None = None):
    """
    Updated: prevents tiny % drift by keeping percent as STRING.
    Also still parses money/int fields to int/float.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None

    s = str(x).strip()
    if s in {"", "—", "-", "–", "nan", "None"}:
        return None

    # ✅ If this is a percent column, NEVER convert to float
    if col_name and ("percent" in col_name.lower() or "percentage" in col_name.lower()):
        return _normalize_percent_string(s)

    # Keep interest rates with PIK/incl. as string
    if re.search(r"\b(PIK|incl\.)\b", s, re.IGNORECASE):
        return s

    # Clean common artifacts
    s2 = s.replace("\u00a0", " ").strip()

    # Keep values with currency codes as string (e.g., "53,764 (EUR 51,863)")
    if re.search(r"\b(EUR|GBP|NOK|SEK|CAD|CHF|AUD|JPY)\b", s2, re.IGNORECASE):
        return s2

    # Keep notes like "2, 16" as string
    if re.fullmatch(r"\d+\s*,\s*\d+.*", s2):
        return s

    # Money/number parse
    s2 = s2.replace("$", "").replace(",", "").strip()

    neg = False
    if re.fullmatch(r"\(.*\)", s2):
        neg = True
        s2 = s2[1:-1].strip()

    # If it still has %, keep as string (no float)
    if s2.endswith("%"):
        return _normalize_percent_string(s2)

    # int / float parsing
    try:
        v = int(s2)
        return -v if neg else v
    except:
        try:
            v = float(s2)
            return -v if neg else v
        except:
            return s


def df_to_ordered_records(df: pd.DataFrame) -> list:
    """
    Important: pass column name into parse_number so % columns stay exact.
    """
    df2 = df.copy()
    df2.columns = [clean_colname(c) for c in df2.columns]
    df2 = df2.where(pd.notnull(df2), None)

    ordered_cols = list(df2.columns)
    records = []

    for _, row in df2.iterrows():
        rec = OrderedDict()
        for col in ordered_cols:
            rec[col] = parse_number(row[col], col_name=col)
        if any(v not in (None, "", "—", "-", "–") for v in rec.values()):
            records.append(rec)

    return records



def write_schedule_json(
    df: pd.DataFrame,
    out_path: str,
    company: str = COMPANY_NAME,
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

    # ✅ fixed structuring
    df_structured = build_structured_schedule(df_raw)

    # Save outputs
    csv_path = "/Users/kirthick21/Desktop/great_elm_schedule_structured3.csv"
    xlsx_path = "/Users/kirthick21/Desktop/great_elm_schedule_structured3.xlsx"
    json_path = "/Users/kirthick21/Desktop/great_elm_schedule_structured3.json"

    df_structured.to_csv(csv_path, index=False)

    try:
        df_structured.to_excel(xlsx_path, index=False)
        print("✅ Saved Excel")
    except ModuleNotFoundError:
        print("⚠️ openpyxl not installed, skipping Excel")

    write_schedule_json(
        df_structured,
        json_path,
        company=COMPANY_NAME,
        as_of=AS_OF_DATE,
        currency="USD"
    )

    print(f"✅ JSON saved at: {json_path}")
    print(f"Total records: {len(df_structured)}")
    print(df_structured.head(15).to_string(index=False))
