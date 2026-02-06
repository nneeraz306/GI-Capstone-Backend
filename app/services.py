import pandas as pd
import os
import json
import re

# Path to the mapped JSON file (from extraction pipeline)
DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mapped_portfolio_data.json')

def generate_borrower_id(name):
    """Generate a simple ID from borrower name"""
    if not name:
        return "unknown"
    return re.sub(r'[^a-z0-9]', '', name.lower())[:20]

def generate_bdc_id(name):
    """Generate BDC ID from name"""
    if not name:
        return "unknown"
    # Map common names to short codes
    name_lower = name.lower()
    if 'great elm' in name_lower or 'gecc' in name_lower:
        return 'gecc'
    elif 'blue owl' in name_lower or 'obdc' in name_lower:
        return 'obdc'
    elif 'blackrock' in name_lower or 'tcp' in name_lower or 'tcpc' in name_lower:
        return 'tcpc'
    else:
        return re.sub(r'[^a-z0-9]', '', name_lower)[:10]

def enrich_record_with_ids(record):
    """Add BDC and borrower IDs to the record for filtering"""
    enriched = record.copy()
    enriched['bdc'] = generate_bdc_id(record.get('directLender', ''))
    enriched['borrower'] = generate_borrower_id(record.get('leadPrimaryBorrower', ''))
    return enriched

def get_portfolio_data(bdc_filter=None, borrower_filter=None):
    """
    Reads data from mapped JSON and applies filters.
    Returns all detailed fields.
    """
    try:
        # Load JSON
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        
        # Enrich with IDs for filtering
        enriched_data = [enrich_record_with_ids(record) for record in data]
        
        # Convert to DataFrame for filtering
        df = pd.DataFrame(enriched_data)

        # Apply Filters
        if bdc_filter and bdc_filter.lower() != 'all':
            df = df[df['bdc'] == bdc_filter]

        if borrower_filter and borrower_filter.lower() != 'all':
            df = df[df['borrower'] == borrower_filter]

        # Convert back to list of dicts
        return df.to_dict(orient='records')
        
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return []

def get_unique_options():
    """
    Get unique BDCs and Borrowers for frontend dropdowns
    """
    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        
        # Enrich with IDs
        enriched_data = [enrich_record_with_ids(record) for record in data]
        df = pd.DataFrame(enriched_data)
        
        # Get unique BDCs
        bdcs = df[['bdc', 'directLender']].drop_duplicates()
        bdcs.columns = ['bdc', 'bdcName']
        
        # Get unique Borrowers
        borrowers = df[['borrower', 'leadPrimaryBorrower']].drop_duplicates()
        borrowers.columns = ['borrower', 'borrowerName']
        
        return {
            "bdcs": bdcs.to_dict(orient='records'),
            "borrowers": borrowers.to_dict(orient='records')
        }
    except Exception as e:
        print(f"Error getting options: {e}")
        return {"bdcs": [], "borrowers": []}