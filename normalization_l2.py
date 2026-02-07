import pandas as pd
import numpy as np
import re

def run_security_normalization(input_path, output_path):
    print(f"--- Running Layer 2: Normalization on {input_path} ---")

    # 1. Read the CSV
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File {input_path} not found.")
        return

    # =========================================================
    # LOGIC FOR BDC 1 (data_l1_consolidated.csv)
    # =========================================================
    if 'Investment' in df.columns and 'Reference Rate and Spread' in df.columns:
        print("Detected BDC 1 format (data_l1_consolidated). Applying specific logic...")

        # --- A. Split 'Investment' ---
        def parse_investment(val):
            val = str(val)
            seniority = ""
            par_amount = ""
            maturity_date = ""

            if '(' in val:
                parts = val.split('(', 1)
                seniority = parts[0].strip()
                inner = parts[1].replace(')', '')
                
                # Extract Par Amount
                par_match = re.search(r'([A-Z]{3}\s?[\d,]+|\$[\d,]+)\s*par', inner, re.IGNORECASE)
                if par_match:
                    raw_par = par_match.group(1)
                    par_amount = re.sub(r'[^\d.]', '', raw_par)

                # Extract Maturity Date
                due_match = re.search(r'due\s+(\d{1,2}/\d{4}|\d{1,2}/\d{1,2}/\d{2,4})', inner, re.IGNORECASE)
                if due_match:
                    maturity_date = due_match.group(1)
            else:
                seniority = val.strip()

            return seniority, par_amount, maturity_date

        print("Splitting 'Investment' column...")
        new_inv_cols = ['Seniority Ranking', 'Par Amount / Quantity', 'Maturity Date']
        df[new_inv_cols] = df['Investment'].apply(lambda x: pd.Series(parse_investment(x)))
        
        # DROP ORIGINAL COLUMN
        df.drop(columns=['Investment'], inplace=True)

        # --- B. Split 'Reference Rate and Spread' ---
        def parse_ref_rate(val):
            val = str(val)
            base_rate = ""
            margin = ""
            
            if '+' in val:
                parts = val.split('+')
                base_rate = parts[0].strip()
                margin = parts[1].strip()
            elif '%' in val:
                margin = val.strip()
            else:
                base_rate = val.strip()
            
            return base_rate, margin

        print("Splitting 'Reference Rate and Spread' column...")
        new_ref_cols = ['Base Rate', 'Margin']
        df[new_ref_cols] = df['Reference Rate and Spread'].apply(lambda x: pd.Series(parse_ref_rate(x)))
        
        # DROP ORIGINAL COLUMN
        df.drop(columns=['Reference Rate and Spread'], inplace=True)

        # --- C. Split 'Interest Rate' ---
        def parse_interest_rate_bdc1(val):
            val = str(val)
            pik = ""
            cash = ""
            
            pik_match = re.search(r'(\d{1,2}\.?\d*)%\s*PIK', val, re.IGNORECASE)
            if pik_match:
                pik = pik_match.group(1) + "%"
            
            if pd.notna(val) and val.strip() != 'nan':
                 cash = val.strip()
            
            return cash, pik

        print("Splitting 'Interest Rate' column...")
        new_int_cols = ['Cash Rate', 'PIK']
        df[new_int_cols] = df['Interest Rate'].apply(lambda x: pd.Series(parse_interest_rate_bdc1(x)))
        
        # DROP ORIGINAL COLUMN
        df.drop(columns=['Interest Rate'], inplace=True)

    # =========================================================
    # LOGIC FOR BDC 0 (Standard / data_l0_normalized)
    # =========================================================
    elif 'Security' in df.columns:
        print("Detected BDC 0 format (Standard). Running existing logic...")
        
        # 1. Security Split
        def split_security_col(val):
            val_str = str(val).strip()
            if ',' in val_str:
                parts = val_str.split(',', 1) 
                return parts[0].strip(), parts[1].strip()
            else:
                return val_str, ""

        new_sec_cols = ['Seniority Ranking', 'Secured/Subordinated']
        df[new_sec_cols] = df['Security'].apply(lambda x: pd.Series(split_security_col(x)))

        # Reorder & Drop Original 'Security'
        cols = list(df.columns)
        sec_idx = cols.index('Security')
        
        # Insert new cols
        for i, col_name in enumerate(new_sec_cols):
             # Remove if exists at end (default append behavior)
            if col_name in cols: cols.remove(col_name)
            cols.insert(sec_idx + i, col_name)
        
        # Remove 'Security'
        if 'Security' in cols: cols.remove('Security')
        
        df = df[cols]


        # 2. Interest Rate Split
        if 'Interest Rate' in df.columns:
            def parse_interest_rate(val):
                base_rate, floor, margin, pik, cash_rate = "", "", "", "", ""
                s = str(val).strip()
                if not s or s.lower() == 'nan': return base_rate, floor, margin, pik, cash_rate

                base_match = re.search(r'((?:1M|3M|6M|12M)\s+SOFR|SOFR|LIBOR|PRIME|EURIBOR)', s, re.IGNORECASE)
                if base_match: base_rate = base_match.group(1).strip()

                margin_match = re.search(r'\+\s*(\d{1,2}\.?\d*)%', s)
                if margin_match: margin = margin_match.group(1) + "%"

                floor_match = re.search(r'Floor\s*(\d{1,2}\.?\d*)%', s, re.IGNORECASE)
                if floor_match: floor = floor_match.group(1) + "%"

                pik_match = re.search(r'(\d{1,2}\.?\d*)%\s*PIK', s, re.IGNORECASE)
                if pik_match: pik = pik_match.group(1) + "%"

                cash_explicit = re.search(r'(\d{1,2}\.?\d*)%\s*Cash', s, re.IGNORECASE)
                total_in_parens = re.search(r'\(\s*(\d{1,2}\.?\d*)%\s*\)', s)
                standalone_pct = re.search(r'^(\d{1,2}\.?\d*)%$', s)

                if cash_explicit: cash_rate = cash_explicit.group(1) + "%"
                elif total_in_parens: cash_rate = total_in_parens.group(1) + "%"
                elif standalone_pct and not base_rate: cash_rate = standalone_pct.group(1) + "%"

                return base_rate, floor, margin, pik, cash_rate

            new_ir_cols = ['Base Rate', 'Floor', 'Margin', 'PIK', 'Cash Rate']
            df[new_ir_cols] = df['Interest Rate'].apply(lambda x: pd.Series(parse_interest_rate(x)))

            # Reorder & Drop Original 'Interest Rate'
            cols = list(df.columns)
            ir_idx = cols.index('Interest Rate')
            
            # Insert new cols
            for i, col_name in enumerate(new_ir_cols):
                if col_name in cols: cols.remove(col_name)
                cols.insert(ir_idx + i, col_name)

            # Remove 'Interest Rate'
            if 'Interest Rate' in cols: cols.remove('Interest Rate')

            df = df[cols]

    else:
        print("Warning: Unknown file format. No normalization applied.")

    # Save
    df.to_csv(output_path, index=False)
    print(f"Layer 2 Complete. Data saved to: {output_path}")

if __name__ == "__main__":
    pass