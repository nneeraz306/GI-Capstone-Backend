import pandas as pd
import os
import json
import re
from datetime import datetime

# BDC name mapping with source URLs
BDC_MAPPING = {
    0: {
        "bdc": "gecc",
        "bdcName": "Great Elm Capital Corp.",
        "reportDate": "2025-09-30",
        "reportType": "10-Q",
        "sourceUrl": "https://www.sec.gov/Archives/edgar/data/1675033/000119312525264873/gecc-20250930.htm"
    },
    1: {
        "bdc": "obdc",
        "bdcName": "Blue Owl Capital Corporation",
        "reportDate": "2025-09-30",
        "reportType": "10-Q",
        "sourceUrl": "https://www.sec.gov/Archives/edgar/data/0001925309/000119312525266848/ck0001925309-20250930.htm"
    },
    2: {
        "bdc": "tcpc",
        "bdcName": "BlackRock TCP Capital Corp.",
        "reportDate": "2025-09-30",
        "reportType": "10-Q",
        "sourceUrl": "https://www.sec.gov/Archives/edgar/data/0001370755/000119312525268125/tcpc-20250930.htm"
    }
}

def safe_get(row, columns, default=""):
    """Safely get value from row trying multiple column names"""
    for col in columns:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    return default

def extract_borrower_name(row, df_columns):
    """Extract borrower/company name - different per BDC"""
    possible_columns = [
        'Portfolio Company', 'Company', 'Issuer', 'Borrower', 
        'Investment', 'Name', 'Investments at Fair Value'
    ]
    
    # Try each column in order
    for col in possible_columns:
        val = safe_get(row, [col])
        if val and val not in ['', 'nan', 'Total', 'Subtotal']:
            # Clean up footnote markers like (1), (2), etc.
            val = re.sub(r'\(\d+\)', '', val).strip()
            return val
    
    # Fallback: return first non-empty value
    for col in df_columns:
        val = safe_get(row, [col])
        if val and len(val) > 2:
            return val
    
    return ""

def extract_industry_sector(row, df_columns):
    """Extract industry/sector for SIG/NAICS mapping"""
    sector_columns = ['Industry', 'Sector', 'Industry Sector', 'Business Segment']
    return safe_get(row, sector_columns, "General / Diversified")

def parse_instrument_info(instrument_str):
    """
    Parse instrument description to extract:
    - Instrument type (Term Loan, Revolver, Bond, Equity, etc.)
    - Seniority (First Lien, Second Lien, Unsecured, etc.)
    - Secured/Subordinated status
    """
    if not instrument_str:
        return {
            "instrumentType": "Term Loan",
            "seniorityRanking": "First Lien",
            "securedSubordinated": "Secured"
        }
    
    instrument_str = instrument_str.lower()
    
    # Determine instrument type
    if 'revolver' in instrument_str or 'revolving' in instrument_str:
        inst_type = "Revolver"
    elif 'bond' in instrument_str or 'note' in instrument_str:
        inst_type = "Bond/Note"
    elif 'equity' in instrument_str or 'common' in instrument_str or 'preferred' in instrument_str:
        inst_type = "Equity"
    elif 'warrant' in instrument_str:
        inst_type = "Warrant"
    else:
        inst_type = "Term Loan"
    
    # Determine seniority
    if 'first lien' in instrument_str or '1st lien' in instrument_str:
        seniority = "First Lien"
    elif 'second lien' in instrument_str or '2nd lien' in instrument_str:
        seniority = "Second Lien"
    elif 'senior secured' in instrument_str:
        seniority = "Senior Secured"
    elif 'senior unsecured' in instrument_str or 'unsecured' in instrument_str:
        seniority = "Senior Unsecured"
    elif 'subordinated' in instrument_str:
        seniority = "Subordinated"
    elif 'unitranche' in instrument_str:
        seniority = "Unitranche"
    else:
        seniority = "First Lien"  # Default assumption
    
    # Determine secured/subordinated
    if 'secured' in instrument_str and 'unsecured' not in instrument_str:
        sec_sub = "Secured"
    elif 'unsecured' in instrument_str:
        sec_sub = "Unsecured"
    elif 'subordinated' in instrument_str:
        sec_sub = "Subordinated"
    else:
        sec_sub = "Secured"  # Default
    
    return {
        "instrumentType": inst_type,
        "seniorityRanking": seniority,
        "securedSubordinated": sec_sub
    }

def extract_interest_rate_components(row, df_columns):
    """
    Extract interest rate components:
    - Coupon type (Fixed/Floating)
    - Basis (SOFR, LIBOR, Prime, etc.)
    - Floor
    - Margin (in bps)
    - PIK Margin
    - Cash Rate
    - Implied Coupon
    """
    result = {
        "couponType": "Floating",
        "basis": "SOFR",
        "floor": "",
        "margin": "",
        "pikMargin": "",
        "cashRate": "",
        "impliedCoupon": ""
    }
    
    # Get interest rate string
    interest_rate = safe_get(row, ['Interest Rate', 'Reference Rate and Spread', 'Total Coupon', 'Spread'])
    
    if not interest_rate:
        return result
    
    # Check if fixed or floating
    if 'fixed' in interest_rate.lower() or '%' in interest_rate and 'sofr' not in interest_rate.lower() and 'libor' not in interest_rate.lower():
        result["couponType"] = "Fixed"
    
    # Extract basis
    if 'sofr' in interest_rate.lower():
        result["basis"] = "SOFR"
    elif 'libor' in interest_rate.lower():
        result["basis"] = "LIBOR"
    elif 'prime' in interest_rate.lower():
        result["basis"] = "Prime"
    elif 'euribor' in interest_rate.lower() or 'e +' in interest_rate.lower():
        result["basis"] = "EURIBOR"
    
    # Extract margin (convert % to bps)
    # Pattern: "+5.00%" or "+ 5.00%" or "SOFR + 5.00%"
    margin_match = re.search(r'\+\s*(\d+\.?\d*)\%', interest_rate)
    if margin_match:
        margin_pct = float(margin_match.group(1))
        result["margin"] = str(int(margin_pct * 100))  # Convert to bps
    
    # Check for separate Spread column (BlackRock format)
    spread_val = safe_get(row, ['Spread'])
    if spread_val:
        # Could be like "5.88%" or "575 bps"
        spread_match = re.search(r'(\d+\.?\d*)', spread_val)
        if spread_match:
            spread_num = float(spread_match.group(1))
            if spread_num > 50:  # Already in bps
                result["margin"] = str(int(spread_num))
            else:  # In percentage
                result["margin"] = str(int(spread_num * 100))
    
    # Extract floor
    floor_val = safe_get(row, ['Floor'])
    if floor_val:
        floor_match = re.search(r'(\d+\.?\d*)', floor_val)
        if floor_match:
            result["floor"] = floor_match.group(1)
    
    # Extract PIK margin
    if 'pik' in interest_rate.lower():
        pik_match = re.search(r'(\d+\.?\d*)\%\s*pik', interest_rate.lower())
        if pik_match:
            pik_pct = float(pik_match.group(1))
            result["pikMargin"] = str(int(pik_pct * 100))
    
    # Extract cash rate
    if 'cash' in interest_rate.lower():
        cash_match = re.search(r'(\d+\.?\d*)\%\s*cash', interest_rate.lower())
        if cash_match:
            result["cashRate"] = cash_match.group(1) + "%"
    
    # Extract implied coupon (total rate)
    # Pattern: "10.5%" or "10.50%"
    coupon_match = re.search(r'(\d+\.?\d+)\%', interest_rate)
    if coupon_match:
        result["impliedCoupon"] = coupon_match.group(1) + "%"
    
    # Check for Total Coupon column (BlackRock format)
    total_coupon = safe_get(row, ['Total Coupon'])
    if total_coupon:
        coupon_match = re.search(r'(\d+\.?\d+)\%', total_coupon)
        if coupon_match:
            result["impliedCoupon"] = coupon_match.group(1) + "%"
    
    return result

def extract_dates(row, df_columns):
    """Extract acquisition date and maturity date"""
    acq_date = safe_get(row, ['Initial Acquisition Date', 'Acquisition Date', 'Pricing Date'])
    maturity = safe_get(row, ['Maturity Date', 'Maturity'])
    
    return {
        "pricingDate": acq_date,
        "maturityType": maturity
    }

def extract_financial_values(row, df_columns):
    """Extract principal, cost, and fair value"""
    principal = safe_get(row, ['Principal', 'Par Amount / Quantity', 'Par Amount/Quantity', 'Shares/Units'])
    cost = safe_get(row, ['Cost', 'Amortised Cost', 'Amortized Cost'])
    fair_value = safe_get(row, ['Fair Value'])
    
    def clean_numeric(val):
        if not val:
            return ""
        # Remove $, commas, and convert to number
        cleaned = re.sub(r'[\$,]', '', val)
        try:
            num = float(cleaned)
            return f"${num:,.0f}"
        except:
            return val
    
    return {
        "principal": clean_numeric(principal),
        "amortisedCost": clean_numeric(cost),
        "fairValue": clean_numeric(fair_value)
    }

def extract_currency(row, df_columns):
    """Extract currency type"""
    # Check for EUR, GBP indicators in various columns
    full_row_text = ' '.join([str(v) for v in row if pd.notna(v)])
    
    if 'eur' in full_row_text.lower() or '€' in full_row_text:
        return "EUR"
    elif 'gbp' in full_row_text.lower() or '£' in full_row_text:
        return "GBP"
    else:
        return "USD"

def map_row_to_output(row, df_columns, bdc_info):
    """Map a single row to standardized output format"""
    
    # Extract borrower name
    borrower_name = extract_borrower_name(row, df_columns)
    if not borrower_name or len(borrower_name) < 3:
        return None  # Skip invalid rows
    
    # Get instrument information
    instrument_str = safe_get(row, ['Security', 'Investment', 'Instrument', 'Type'])
    instrument_info = parse_instrument_info(instrument_str)
    
    # Get interest rate components
    rate_info = extract_interest_rate_components(row, df_columns)
    
    # Get dates
    date_info = extract_dates(row, df_columns)
    
    # Get financial values
    financial_info = extract_financial_values(row, df_columns)
    
    # Get industry/sector
    industry = extract_industry_sector(row, df_columns)
    
    # Get currency
    currency = extract_currency(row, df_columns)
    
    # Get notes for additional borrowers
    notes = safe_get(row, ['Notes', 'Note', 'Footnotes'])
    
    return {
        "directLender": bdc_info["bdcName"],
        "leadPrimaryBorrower": borrower_name,
        "additionalBorrower": "",  # Would need to parse from notes
        "businessDescription": f"Investment in {borrower_name}",
        "sigNaics": industry,
        "pricingDate": date_info["pricingDate"],
        "instrumentType": instrument_info["instrumentType"],
        "seniorityRanking": instrument_info["seniorityRanking"],
        "securedSubordinated": instrument_info["securedSubordinated"],
        "bdcCouponRate": rate_info["couponType"],
        "basis": rate_info["basis"],
        "floor": rate_info["floor"],
        "margin": rate_info["margin"],
        "pikMargin": rate_info["pikMargin"],
        "cashRate": rate_info["cashRate"],
        "impliedCoupon": rate_info["impliedCoupon"],
        "maturityType": date_info["maturityType"],
        "currencyType": currency,
        "principalType": financial_info["principal"],
        "amortisedCost": financial_info["amortisedCost"],
        "fairValue": financial_info["fairValue"],
        "reportDate": bdc_info["reportDate"],
        "reportType": bdc_info["reportType"],
        "source": bdc_info["sourceUrl"]
    }

def map_csv_to_output(input_file, file_index):
    """
    Map CSV to standardized output format
    """
    print(f"Mapping {input_file}...")
    
    try:
        df = pd.read_csv(input_file)
        
        # Get BDC info
        bdc_info = BDC_MAPPING.get(file_index, {
            "bdc": f"bdc{file_index}",
            "bdcName": f"BDC {file_index}",
            "reportDate": "2025-09-30",
            "reportType": "10-Q",
            "sourceUrl": ""
        })
        
        print(f"  BDC: {bdc_info['bdcName']}")
        print(f"  Columns found: {list(df.columns)[:5]}...")  # Show first 5 columns
        
        # Filter out invalid rows
        # Skip rows that are all NaN, header rows, or total rows
        df = df.dropna(how='all')
        
        # Map each row
        mapped_data = []
        skipped = 0
        
        for idx, row in df.iterrows():
            try:
                mapped_row = map_row_to_output(row, df.columns, bdc_info)
                if mapped_row:
                    mapped_data.append(mapped_row)
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1
                if skipped <= 3:  # Only show first 3 errors
                    print(f"  Warning: Could not map row {idx}: {e}")
        
        print(f"  Successfully mapped: {len(mapped_data)} records")
        print(f"  Skipped: {skipped} rows")
        return mapped_data
    
    except Exception as e:
        print(f"Error mapping {input_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def map_all_files(normalized_dir=".", output_file="mapped_portfolio_data.json"):
    """
    Map all CSV files to output JSON (works with soi_table*.csv files directly)
    """
    import glob
    import re
    
    # Look for either normalized files or original extracted files
    csv_files = sorted(glob.glob(os.path.join(normalized_dir, "soi_table*.csv")))
    
    if not csv_files:
        print(f"No CSV files found in {normalized_dir}")
        return None
    
    print(f"Found {len(csv_files)} CSV files to map\n")
    
    all_mapped_data = []
    
    for i, file_path in enumerate(csv_files):
        # Extract file index from filename (e.g., soi_table0.csv -> 0)
        match = re.search(r'soi_table(\d+)\.csv', file_path)
        file_index = int(match.group(1)) if match else i
        
        mapped_data = map_csv_to_output(file_path, file_index)
        all_mapped_data.extend(mapped_data)
        print()
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(all_mapped_data, f, indent=2)
    
    print(f"Successfully mapped {len(all_mapped_data)} total records")
    print(f"Output saved to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    import re
    map_all_files()
