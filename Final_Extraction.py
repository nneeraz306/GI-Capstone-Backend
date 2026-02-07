#!/usr/bin/env python3
"""
SEC BDC filing -> SOI tables only -> merged CSV + holdings-only CSV

What this does:
1) Fetch filing HTML
2) Find best "Schedule of Investments" anchor (or follow the TOC link like R7.htm)
3) Collect nearby tables
4) KEEP ONLY tables that look like SOI tables (strong header keywords + SOI-ish text)
5) Parse tables (pandas.read_html first, then BS4 fallback)
6) Save:
   - soi_raw_only_{n}.csv        (only SOI tables merged)
   - soi_holdings_only_{n}.csv   (trimmed to holdings rows, drops footnotes/headers)
"""

import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

# ---------------------------
# CONFIG
# ---------------------------
HEADERS = {
    "User-Agent": "Kirthick Easwar B kirthick31@gmail.com",
    "Accept-Encoding": "gzip, deflate, br",
}

SOI_TITLE_PATTERNS = [
    r"\bconsolidated schedule of investments\b",
    r"\bschedule of investments\b",
    r"\bschedule of investment\b",
    r"\bportfolio of investments\b",
    r"\binvestments\s*\(unaudited\)\b",
]

# SOI-ish terms anywhere in the table text (broad)
SOI_TEXT_HINTS = [
    "portfolio", "company", "issuer", "industry", "sector", "security",
    "type of investment", "investment", "principal", "par", "cost", "amortized",
    "fair value", "maturity", "interest", "rate", "acquisition", "net assets",
    "notes", "index", "margin", "floor", "ceiling"
]

# Strong header terms that usually appear in the top rows of an SOI table
SOI_HEADER_HINTS = [
    "portfolio company", "company", "issuer",
    "industry", "sector",
    "security", "type of investment", "investment",
    "maturity",
    "principal", "par",
    "cost",
    "fair value",
    "% of", "net assets",
    "interest", "rate", "cash rate",
    "index", "margin", "spread",
    "floor", "ceiling",
]

# ---------------------------
# Utils
# ---------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def _abs_url(base: str, href: str) -> str:
    if not href:
        return href
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("/"):
        return "https://www.sec.gov" + href
    return base.rsplit("/", 1)[0] + "/" + href

def _score_table_tag(table_tag) -> int:
    t = _norm(table_tag.get_text(" ", strip=True))
    return sum(1 for k in SOI_TEXT_HINTS if k in t)

def _table_header_score(table_tag, max_rows: int = 5) -> int:
    """
    Score only the first few rows of the table, since that’s where true SOI headers show up.
    This avoids pulling in random financial statement tables later in the filing.
    """
    rows = table_tag.find_all("tr", limit=max_rows)
    if not rows:
        return 0
    top_text = " ".join(_norm(r.get_text(" ", strip=True)) for r in rows)
    return sum(1 for k in SOI_HEADER_HINTS if k in top_text)

def _is_soi_table(table_tag) -> bool:
    """
    Hard filter: table must look SOI-like in the first rows and overall text.
    Tuned to avoid EPS / debt facility / roll-forward tables.
    """
    header_score = _table_header_score(table_tag, max_rows=6)
    body_score = _score_table_tag(table_tag)

    # Require strong header evidence
    if header_score >= 5:
        return True

    # Or: moderate header + strong body text
    if header_score >= 3 and body_score >= 8:
        return True

    return False

def _parse_table_fallback_bs4(table_tag) -> pd.DataFrame | None:
    """Manual parser for SEC tables when pandas.read_html fails."""
    rows = []
    for tr in table_tag.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        row = [c.get_text(" ", strip=True) for c in cells]
        if any(x.strip() for x in row):
            rows.append(row)

    if not rows:
        return None

    max_len = max(len(r) for r in rows)
    rows = [r + [""] * (max_len - len(r)) for r in rows]

    header = rows[0]
    body = rows[1:] if len(rows) > 1 else []

    def numericish(x: str) -> bool:
        x = x.strip().replace(",", "")
        return bool(re.fullmatch(r"[\(\)\-\$]*\d+(\.\d+)?%?[\(\)\-\$]*", x)) if x else False

    if sum(1 for x in header if numericish(x)) >= max(1, len(header) // 2):
        cols = [f"col_{i+1}" for i in range(max_len)]
        return pd.DataFrame(rows, columns=cols)

    cols = [h.strip() if h.strip() else f"col_{i+1}" for i, h in enumerate(header)]
    return pd.DataFrame(body, columns=cols)

def _parse_table(table_tag) -> pd.DataFrame | None:
    """Try pandas first, then fallback manual parse."""
    try:
        dfs = pd.read_html(str(table_tag))
        if dfs and isinstance(dfs[0], pd.DataFrame):
            df = dfs[0]
            if df.shape[0] >= 2 and df.shape[1] >= 2:
                return df
    except Exception:
        pass
    return _parse_table_fallback_bs4(table_tag)

def _find_best_soi_anchor_by_text(soup) -> tuple[object, int] | tuple[None, int]:
    title_regex = re.compile("|".join(SOI_TITLE_PATTERNS), re.IGNORECASE)

    candidates = []
    for tag in soup.find_all(["h1","h2","h3","h4","p","div","span","td","a"]):
        txt = tag.get_text(" ", strip=True)
        if not txt or not title_regex.search(txt):
            continue

        next_tables = tag.find_all_next("table", limit=30)
        scores = sorted([_score_table_tag(t) for t in next_tables], reverse=True)
        best = scores[0] if scores else 0
        top3 = sum(scores[:3]) if scores else 0

        penalty = 0
        if tag.name == "a" and tag.get("href"):
            penalty += 2
        if len(txt) <= 40:
            penalty += 1

        final = (top3 * 2 + best * 3) - penalty
        candidates.append((final, tag, best))

    if not candidates:
        return None, 0

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], candidates[0][2]

def _maybe_follow_soi_link(anchor_tag, base_url: str) -> str | None:
    if anchor_tag and anchor_tag.name == "a":
        href = anchor_tag.get("href")
        if href and (href.lower().endswith(".htm") or href.lower().endswith(".html")):
            return _abs_url(base_url, href)
    a = anchor_tag.find("a", href=True) if anchor_tag else None
    if a:
        href = a.get("href")
        if href and (href.lower().endswith(".htm") or href.lower().endswith(".html")):
            return _abs_url(base_url, href)
    return None

# ---------------------------
# HOLDINGS-ONLY CLEANER (your edited version)
# ---------------------------
def _row_text(row):
    return " ".join(_norm(str(x)) for x in row.values if _norm(str(x)))

def _has_money_or_number(s: str) -> bool:
    s = s.replace(",", "")
    return bool(re.search(r"(\$?\(?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?|\$?\d+(?:\.\d+)?%?)", s))

def _is_footnote_like(s: str) -> bool:
    s_low = s.lower()
    return (
        re.search(r"^\(\s*\d+\s*\)", s.strip()) is not None
        or "all investments of the company are in entities" in s_low
        or "portfolio company is a public company" in s_low
        or "value as a percent of net assets" in s_low
        or "for debt investments" in s_low
        or "three months ended" in s_low
        or "nine months ended" in s_low
        or "for the three months ended" in s_low
        or "for the nine months ended" in s_low
    )

def _looks_like_header_row(s: str) -> bool:
    s_low = s.lower()
    hits = sum(k in s_low for k in [
        "portfolio", "company", "sector", "type of investment", "maturity",
        "principal", "cost", "fair value", "cash rate", "index", "margin", "floor", "ceiling"
    ])
    return hits >= 4

def clean_soi_holdings_block(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2 = df2.loc[~df2.apply(lambda r: all(str(x).strip()=="" for x in r.values), axis=1)].reset_index(drop=True)

    start_idx = None
    seen_header = False
    for i, row in df2.iterrows():
        txt = _row_text(row)
        if _looks_like_header_row(txt):
            seen_header = True
            continue
        if not seen_header:
            continue
        if _is_footnote_like(txt):
            continue
        first_cell = str(row.iloc[0]).strip()
        if first_cell and len(first_cell) > 2 and _has_money_or_number(txt):
            start_idx = i
            break

    if start_idx is None:
        for i, row in df2.iterrows():
            txt = _row_text(row)
            if _is_footnote_like(txt):
                continue
            if _has_money_or_number(txt):
                start_idx = i
                break

    if start_idx is None:
        return df2

    end_idx = len(df2)
    no_num_streak = 0
    for i in range(start_idx, len(df2)):
        txt = _row_text(df2.iloc[i])
        if _is_footnote_like(txt) and re.search(r"^\(\s*\d+\s*\)", txt.strip()):
            end_idx = i
            break
        if not _has_money_or_number(txt):
            no_num_streak += 1
        else:
            no_num_streak = 0
        if no_num_streak >= 12:
            end_idx = max(start_idx, i - 12)
            break

    cleaned = df2.iloc[start_idx:end_idx].reset_index(drop=True)
    cleaned = cleaned.loc[~cleaned.apply(lambda r: _is_footnote_like(_row_text(r)), axis=1)].reset_index(drop=True)
    return cleaned

# ---------------------------
# Main extractor
# ---------------------------
def extract_soi_full(url: str, num: int):
    print(f"\n[{num}] Fetching: {url}")
    html = requests.get(url, headers=HEADERS).text
    soup = BeautifulSoup(html, "lxml")

    anchor, best_tbl_score = _find_best_soi_anchor_by_text(soup)
    if not anchor:
        raise RuntimeError("Could not find any SOI title text in this document.")

    follow_url = _maybe_follow_soi_link(anchor, url)
    if follow_url and best_tbl_score < 3:
        print(f"↪️ Following SOI link: {follow_url}")
        html2 = requests.get(follow_url, headers=HEADERS).text
        soup2 = BeautifulSoup(html2, "lxml")
        anchor2, _ = _find_best_soi_anchor_by_text(soup2)
        soup = soup2
        anchor = anchor2 if anchor2 else soup2

    # Get many next tables, but filter strictly to SOI tables ONLY
    tables = anchor.find_all_next("table", limit=160)

    soi_tables = []
    for t in tables:
        if _is_soi_table(t):
            soi_tables.append(t)

    if not soi_tables:
        # Debug scores to tune thresholds
        scored = [( _table_header_score(t, 6), _score_table_tag(t), t) for t in tables[:30]]
        top = sorted(scored, key=lambda x: (x[0], x[1]), reverse=True)[:10]
        dbg = [(hs, bs) for hs, bs, _ in top]
        raise RuntimeError(
            "Found SOI anchor, but did not find any tables that pass SOI-only filters.\n"
            f"Top (header_score, body_score) among first 30 tables: {dbg}\n"
            "If this filing is PDF/image-based, you’ll need OCR."
        )

    dfs = []
    for t in soi_tables:
        df = _parse_table(t)
        if df is not None and df.shape[0] >= 2 and df.shape[1] >= 2:
            dfs.append(df)

    if not dfs:
        raise RuntimeError("SOI tables matched filters, but parsing failed for all of them.")

    out = pd.concat(dfs, ignore_index=True)

    raw_path = f"soi_raw_only_{num}.csv"
    out.to_csv(raw_path, index=False)
    print(f"✅ Saved {raw_path} | soi_tables={len(dfs)} | shape={out.shape}")

    # holdings-only
    out_clean = clean_soi_holdings_block(out)
    clean_path = f"soi_holdings_only_{num}.csv"
    out_clean.to_csv(clean_path, index=False)
    print(f"✅ Saved {clean_path} | shape={out_clean.shape}")

    return out, out_clean

# ---------------------------
# RUN
# ---------------------------
urls = [
    
    "https://www.sec.gov/Archives/edgar/data/0001837532/000119312525277723/ck0001837532-20250930.htm"
]

for i, u in enumerate(urls):
    extract_soi_full(u, i)
