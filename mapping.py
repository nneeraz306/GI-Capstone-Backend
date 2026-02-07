import pandas as pd
import numpy as np
import re

def clean_pct(val):
    if pd.isna(val) or str(val).strip() == '': return np.nan
    s = str(val).replace('%', '').replace(',', '').strip()
    try: return float(s)
    except: return np.nan

def run_mapping(l0_path, l1_path, l2_path, output_path):
    print(f"--- Running Mapping on {l0_path}, {l1_path}, {l2_path} ---")

    # Load DataFrames
    try:
        df0 = pd.read_csv(l0_path)
        df1 = pd.read_csv(l1_path)
        df2 = pd.read_csv(l2_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    mapped_dfs = []

    # ==========================================
    # 1. PROCESS DF0 (Blackstone)
    # ==========================================
    print(f"Processing Blackstone (DF0): {len(df0)} rows")
    m0 = pd.DataFrame(index=df0.index)
    m0['Direct Lender'] = "Blackstone Secured Lending Fund"
    m0['Lead/Primary Borrower'] = df0['Portfolio Company']
    m0['Additional Borrower(s)'] = ""
    m0['Business Description in Datalab'] = df0['Industry']
    m0['SIG/NAICS in Datalab'] = df0['Industry']
    m0['Pricing Date'] = df0['Initial Acquisition Date']
    
    def get_inst_desc_0(row):
        sen = str(row['Seniority Ranking']) if pd.notna(row['Seniority Ranking']) else ""
        sec = str(row['Secured/Subordinated']) if pd.notna(row['Secured/Subordinated']) else ""
        if sen == sec: return sen
        if sen in sec: return sec
        if sec in sen: return sen
        return (sen + " " + sec).strip()
    m0['Instrument Description'] = df0.apply(get_inst_desc_0, axis=1)
    
    m0['Seniority Ranking'] = df0['Seniority Ranking']
    m0['Secured/Subordinated'] = df0['Secured/Subordinated']
    m0['BDC Coupon Rate'] = df0['Base Rate'].apply(lambda x: 'Floating' if pd.notna(x) and str(x).strip() != '' else 'Fixed')
    m0['Basis'] = df0['Base Rate']
    
    m0['Floor'] = df0['Floor'].apply(clean_pct)
    m0['Margin'] = df0['Margin'].apply(lambda x: clean_pct(x) * 100 if pd.notna(x) else np.nan)
    m0['PIK Margin'] = df0['PIK'].apply(lambda x: clean_pct(x) * 100 if pd.notna(x) else np.nan)
    m0['Cash Rate'] = df0['Cash Rate']
    
    m0['Implied Coupon'] = df0.apply(lambda row: (clean_pct(row['Cash Rate']) if pd.notna(clean_pct(row['Cash Rate'])) else 0) + 
                                                (clean_pct(row['PIK']) if pd.notna(clean_pct(row['PIK'])) else 0), axis=1)
    
    m0['Maturity Type'] = df0['Maturity Date']
    m0['Currency Type'] = "USD"
    
    def is_equity(row):
        s = (str(row['Seniority Ranking']) + " " + str(row['Secured/Subordinated'])).lower()
        return 'equity' in s or 'warrant' in s or 'common' in s
    
    m0['Shares/Unit'] = df0.apply(lambda row: row['Par Amount / Quantity'] if is_equity(row) else np.nan, axis=1)
    m0['Principal Type'] = df0.apply(lambda row: row['Par Amount / Quantity'] if not is_equity(row) else np.nan, axis=1)
    m0['Amortised Cost'] = df0['Cost']
    m0['Fair Value'] = df0['Fair Value']
    m0['Report Date'] = "09/30/2024"
    m0['Report Type'] = "10Q"
    m0['Source'] = ""
    
    mapped_dfs.append(m0)

    # ==========================================
    # 2. PROCESS DF1 (Blue Owl)
    # ==========================================
    print(f"Processing Blue Owl (DF1): {len(df1)} rows")
    # Identify Industry Headers
    industry_list = []
    is_header_mask = []
    current_industry = None
    
    for idx, row in df1.iterrows():
        company = str(row['Company'])
        cost = row['Amortized Cost']
        # Header check: Cost is NaN/Empty
        if pd.isna(cost) or str(cost).strip() == '' or str(cost).lower() == 'nan':
            if pd.notna(company) and str(company).strip() != '' and str(company).lower() != 'nan':
                 current_industry = company
            is_header_mask.append(True)
        else:
            is_header_mask.append(False)
        industry_list.append(current_industry)
            
    df1['Industry'] = industry_list
    df1_clean = df1[~pd.Series(is_header_mask)].copy()
    print(f"  Cleaned Blue Owl rows: {len(df1_clean)}")
    
    m1 = pd.DataFrame(index=df1_clean.index)
    m1['Direct Lender'] = "Blue Owl Capital Corporation"
    m1['Lead/Primary Borrower'] = df1_clean['Company']
    m1['Additional Borrower(s)'] = ""
    m1['Business Description in Datalab'] = df1_clean['Industry']
    m1['SIG/NAICS in Datalab'] = df1_clean['Industry']
    m1['Pricing Date'] = df1_clean['Initial Acquisition Date']
    m1['Instrument Description'] = df1_clean['Seniority Ranking']
    m1['Seniority Ranking'] = df1_clean['Seniority Ranking']
    
    def infer_secured(val):
        s = str(val).lower()
        if 'first' in s or 'second' in s or 'secured' in s or 'lien' in s: return "Secured"
        if 'subordinated' in s or 'unsecured' in s: return "Unsecured"
        return "Secured"
    
    m1['Secured/Subordinated'] = df1_clean['Seniority Ranking'].apply(infer_secured)
    m1['BDC Coupon Rate'] = df1_clean['Base Rate'].apply(lambda x: 'Floating' if pd.notna(x) and str(x).strip() != '' and str(x).lower() != 'nan' else 'Fixed')
    m1['Basis'] = df1_clean['Base Rate']
    m1['Floor'] = 0.0
    m1['Margin'] = df1_clean['Margin'].apply(lambda x: clean_pct(x) * 100 if pd.notna(x) else np.nan)
    m1['PIK Margin'] = df1_clean['PIK'].apply(lambda x: clean_pct(x) * 100 if pd.notna(x) else np.nan)
    m1['Cash Rate'] = df1_clean['Cash Rate']
    
    m1['Implied Coupon'] = df1_clean.apply(lambda row: (clean_pct(row['Cash Rate']) if pd.notna(clean_pct(row['Cash Rate'])) else 0) + 
                                                    (clean_pct(row['PIK']) if pd.notna(clean_pct(row['PIK'])) else 0), axis=1)
    
    m1['Maturity Type'] = df1_clean['Maturity Date']
    
    def infer_currency(val):
        s = str(val).upper()
        if 'EUR' in s: return 'EUR'
        if 'GBP' in s: return 'GBP'
        if 'CAD' in s: return 'CAD'
        return 'USD'

    m1['Currency Type'] = df1_clean['Par Amount / Quantity'].apply(infer_currency)
    
    def is_equity_1(val):
        s = str(val).lower()
        return 'equity' in s or 'warrant' in s or 'common' in s
        
    m1['Shares/Unit'] = df1_clean.apply(lambda row: row['Par Amount / Quantity'] if is_equity_1(row['Seniority Ranking']) else np.nan, axis=1)
    m1['Principal Type'] = df1_clean.apply(lambda row: row['Par Amount / Quantity'] if not is_equity_1(row['Seniority Ranking']) else np.nan, axis=1)
    m1['Amortised Cost'] = df1_clean['Amortized Cost']
    m1['Fair Value'] = df1_clean['Fair Value']
    m1['Report Date'] = "09/30/2024"
    m1['Report Type'] = "10Q"
    m1['Source'] = ""
    
    mapped_dfs.append(m1)

    # ==========================================
    # 3. PROCESS DF2 (Ares)
    # ==========================================
    print(f"Processing Ares (DF2): {len(df2)} rows")
    
    current_industry_2 = None
    industry_list_2 = []
    is_header_mask_2 = []
    
    for idx, row in df2.iterrows():
        inst = str(row['Instrument'])
        principal = row['Principal']
        
        # Ares header logic: Principal is usually NaN
        if pd.isna(principal) or str(principal).strip() == '' or str(principal).lower() == 'nan':
             if pd.notna(inst) and 'Debt Investments' not in inst and 'Investments' not in inst:
                current_industry_2 = inst
             is_header_mask_2.append(True)
        else:
            is_header_mask_2.append(False)
        industry_list_2.append(current_industry_2)
            
    df2['Industry'] = industry_list_2
    df2_clean = df2[~pd.Series(is_header_mask_2)].copy()
    print(f"  Cleaned Ares rows: {len(df2_clean)}")
    
    m2 = pd.DataFrame(index=df2_clean.index)
    m2['Direct Lender'] = "Ares Capital Corp"
    m2['Lead/Primary Borrower'] = df2_clean['Issuer']
    m2['Additional Borrower(s)'] = ""
    m2['Business Description in Datalab'] = df2_clean['Industry']
    m2['SIG/NAICS in Datalab'] = df2_clean['Industry']
    m2['Pricing Date'] = "" 
    m2['Instrument Description'] = df2_clean['Instrument']
    
    def infer_sen_2(val):
        s = str(val).lower()
        if 'first' in s or '1st' in s: return "First Lien"
        if 'second' in s or '2nd' in s: return "Second Lien"
        if 'senior' in s: return "Senior"
        if 'subordinated' in s: return "Subordinated"
        return val 
    
    m2['Seniority Ranking'] = df2_clean['Instrument'].apply(infer_sen_2)
    m2['Secured/Subordinated'] = df2_clean['Instrument'].apply(lambda x: "Secured" if "secured" in str(x).lower() or "lien" in str(x).lower() else "Unsecured")
    
    m2['BDC Coupon Rate'] = df2_clean['Ref'].apply(lambda x: 'Floating' if pd.notna(x) and 'SOFR' in str(x).upper() else 'Fixed')
    m2['Basis'] = df2_clean['Ref']
    m2['Floor'] = df2_clean['Floor'].apply(clean_pct)
    
    def parse_spread(val):
        val = str(val)
        margin = np.nan
        pik = np.nan
        
        pik_match = re.search(r'(\d+\.?\d*)%?\s*PIK', val, re.IGNORECASE)
        if pik_match:
            pik = float(pik_match.group(1)) * 100 
            
        cash_match = re.search(r'(\d+\.?\d*)%?\s*Cash', val, re.IGNORECASE)
        if cash_match:
            margin = float(cash_match.group(1)) * 100 
            
        if not pik_match and not cash_match:
            clean_val = clean_pct(val)
            if pd.notna(clean_val):
                margin = clean_val * 100
        
        return margin, pik
        
    spread_parsed = df2_clean['Spread'].apply(lambda x: pd.Series(parse_spread(x)))
    m2['Margin'] = spread_parsed[0]
    m2['PIK Margin'] = spread_parsed[1]
    
    m2['Implied Coupon'] = df2_clean['Total Coupon'].apply(clean_pct)
    
    def calc_cash_rate_2(row):
        total = row['Implied Coupon']
        if pd.isna(total): return np.nan
        
        pik_bps = row['PIK Margin']
        pik_val = pik_bps / 100 if pd.notna(pik_bps) else 0
        return total - pik_val
        
    m2['Cash Rate'] = m2.apply(calc_cash_rate_2, axis=1)
    
    m2['Maturity Type'] = df2_clean['Maturity']
    m2['Currency Type'] = "USD"
    
    m2['Shares/Unit'] = df2_clean.apply(lambda row: row['Principal'] if is_equity_1(row['Instrument']) else np.nan, axis=1)
    m2['Principal Type'] = df2_clean.apply(lambda row: row['Principal'] if not is_equity_1(row['Instrument']) else np.nan, axis=1)
    
    m2['Amortised Cost'] = df2_clean['Cost']
    m2['Fair Value'] = df2_clean['Fair Value']
    m2['Report Date'] = "09/30/2024"
    m2['Report Type'] = "10Q"
    m2['Source'] = ""
    
    mapped_dfs.append(m2)

    # ==========================================
    # 4. CONSOLIDATE & SAVE
    # ==========================================
    final_df = pd.concat(mapped_dfs, ignore_index=True)
    
    target_cols = [
        'Direct Lender', 'Lead/Primary Borrower', 'Additional Borrower(s)', 'Business Description in Datalab',
        'SIG/NAICS in Datalab', 'Pricing Date', 'Instrument Description', 'Seniority Ranking', 'Secured/Subordinated',
        'BDC Coupon Rate', 'Basis', 'Floor', 'Margin', 'PIK Margin', 'Cash Rate', 'Implied Coupon',
        'Maturity Type', 'Currency Type', 'Shares/Unit', 'Principal Type', 'Amortised Cost', 'Fair Value',
        'Report Date', 'Report Type', 'Source'
    ]
    
    for col in target_cols:
        if col not in final_df.columns:
            final_df[col] = ""
            
    final_df = final_df[target_cols]
    final_df.to_csv(output_path, index=False)
    print(f"Mapping Complete. Consolidated data saved to: {output_path}")

if __name__ == "__main__":
    run_mapping('data_l0_normalized.csv', 'data_l1_normalized.csv', 'data_l2_normalized.csv', 'mapped_portfolio_data.csv')