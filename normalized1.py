import pandas as pd
import re

def clean_header_name(header_str):
    """
    Cleans a single header string:
    1. Removes footnotes like (1), (2).
    2. Strips whitespace.
    """
    if pd.isna(header_str) or str(header_str).strip() == "":
        return None
    
    # Remove things like (1), (12) from the string
    clean = re.sub(r'\s*\(\d+\)', '', str(header_str))
    return clean.strip()

def find_header_row_index(df):
    """
    Robustly searches for the header row.
    Looks for a row containing both 'portfolio' and 'company'.
    """
    # Scan the first 20 rows (headers are rarely lower than that)
    for idx, row in df.head(20).iterrows():
        # Convert the entire row to a single lowercase string for easy searching
        # .fillna('') ensures we don't crash on empty cells
        row_text = " ".join(row.fillna("").astype(str)).lower()
        
        # Loose check: does this row contain key words?
        if "portfolio" in row_text and "company" in row_text:
            return idx
    return None

def run_layer_1(input_file, output_file):
    print(f"--- Processing {input_file} ---")

    # 1. Read 'raw' with no header formatting
    #    We use engine='python' which is sometimes more robust for weird CSVs
    try:
        df_raw = pd.read_csv(input_file, header=None, dtype=str, engine='python')
    except Exception as e:
        print(f"CRITICAL ERROR: Could not read CSV file. Details: {e}")
        return

    # 2. Locate the real Header Row using the robust finder
    header_row_index = find_header_row_index(df_raw)
    
    if header_row_index is None:
        print("\n!!! ERROR: Could not find the header row. !!!")
        print("Here is what Python sees in the first 5 rows. Check for typos or encoding issues:")
        print("-" * 50)
        print(df_raw.head(5))
        print("-" * 50)
        raise ValueError("Header not found (checked for 'portfolio' + 'company').")

    print(f"Found headers at row index: {header_row_index}")

    # 3. Extract Headers and Identify Columns to Keep
    raw_header_row = df_raw.iloc[header_row_index]
    
    indices_to_keep = []
    final_header_names = []
    seen_headers = set()

    for col_idx, val in raw_header_row.items():
        cleaned_name = clean_header_name(val)
        
        if cleaned_name:
            # ONLY keep the first time we see a header (avoids the empty duplicate columns)
            if cleaned_name not in seen_headers:
                indices_to_keep.append(col_idx)
                final_header_names.append(cleaned_name)
                seen_headers.add(cleaned_name)

    # 4. Create the new Clean DataFrame
    df_clean = df_raw.iloc[header_row_index + 1:, indices_to_keep].copy()
    df_clean.columns = final_header_names

    # 5. Filter Data Rows
    #    Drop rows where 'Portfolio Company' is NaN
    df_clean = df_clean.dropna(subset=['Portfolio Company'])
    
    #    Drop section headers (rows that just repeat the title or sub-headers)
    df_clean = df_clean[~df_clean['Portfolio Company'].str.contains("Investments at Fair Value", case=False, na=False)]

    # 6. Save
    df_clean.to_csv(output_file, index=False)
    print(f"--- Layer 1 Complete. Saved to {output_file} ---")