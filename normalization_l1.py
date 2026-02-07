# import pandas as pd
# import numpy as np

# def run_header_consolidation(input_path, output_path):
#     print(f"--- Running Layer 1: Header Consolidation on {input_path} ---")

#     # 1. Read the CSV (Header is on row 3, which is index 2)
#     df = pd.read_csv(input_path, header=2)

#     # 2. Drop "Ghost" Columns (Unnamed)
#     valid_header_cols = [c for c in df.columns if not str(c).startswith("Unnamed")]
#     df = df[valid_header_cols]

#     # 3. Clean Duplicate/Empty Columns FIRST
#     # We do this first so the empty version of "Cost" is deleted, 
#     # leaving only "Cost.1" (which has the data).
#     df = df.dropna(axis=1, how='all')

#     # 4. Clean Header Names (The Universal Fix)
#     # Step A: Remove suffixes like .1, .2 at the end (e.g., "Cost.1" -> "Cost")
#     # Regex: \. matches a dot, \d+ matches numbers, $ matches the end of the line
#     df.columns = df.columns.str.replace(r'\.\d+$', '', regex=True)

#     # Step B: Remove footnotes like (1), (2), (15) (e.g., "Security(1)" -> "Security")
#     # Regex: \s* matches optional space, \( matches (, \d+ matches number, \) matches )
#     df.columns = df.columns.str.replace(r'\s*\(\d+\)', '', regex=True)

#     # Step C: Strip any remaining whitespace
#     df.columns = df.columns.str.strip()

#     # 5. Filter Rows (Remove Section Headers)
#     if 'Portfolio Company' in df.columns:
#         df = df[df['Portfolio Company'].notna()]
#         df = df[df['Portfolio Company'] != "Investments at Fair Value"]
        
#         # Reset the index
#         df.reset_index(drop=True, inplace=True)

#     # 6. Save the Cleaned File
#     df.to_csv(output_path, index=False)
#     print(f"Layer 1 Complete. Data saved to: {output_path}")

import pandas as pd
import numpy as np

def run_header_consolidation(input_path, output_path):
    print(f"--- Running Layer 1: Header Consolidation on {input_path} ---")

    # 1. Read the CSV (Header is on row 3, which is index 2)
    df = pd.read_csv(input_path, header=2)

    # 2. Drop "Ghost" Columns (Unnamed)
    valid_header_cols = [c for c in df.columns if not str(c).startswith("Unnamed")]
    df = df[valid_header_cols]

    # 3. Clean Duplicate/Empty Columns FIRST
    # This removes columns that are completely empty (NaNs)
    df = df.dropna(axis=1, how='all')

    # 4. Clean Header Names (The Universal Fix)
    # Remove suffixes like .1, .2 and footnotes like (1), (3)
    df.columns = df.columns.str.replace(r'\.\d+$', '', regex=True)
    df.columns = df.columns.str.replace(r'\s*\(\d+\)', '', regex=True)
    df.columns = df.columns.str.strip()

    # --- NEW ADDITION: Remove First Occurrence of Duplicates ---
    # After cleaning, "Percentage of Class" appears twice.
    # duplicated(keep='last') marks the first occurrence as a duplicate to be removed.
    # The '~' symbol inverts the selection, so we KEEP the ones that are NOT duplicates.
    df = df.loc[:, ~df.columns.duplicated(keep='last')]
    # -----------------------------------------------------------

    # 6. Filter Rows (Remove Section Headers)
    if 'Portfolio Company' in df.columns:
        df = df[df['Portfolio Company'].notna()]
        df = df[df['Portfolio Company'] != "Investments at Fair Value"]
        df.reset_index(drop=True, inplace=True)

    # 7. Save the Cleaned File
    df.to_csv(output_path, index=False)
    print(f"Layer 1 Complete. Data saved to: {output_path}")