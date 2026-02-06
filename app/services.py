import pandas as pd
import os
import json

# Path to your JSON file
DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'portfolio_data.json')

def get_portfolio_data(bdc_filter=None, borrower_filter=None):
    """
    Reads data from JSON, loads into Pandas, applies filters, and returns records.
    """
    try:
        # Load JSON into Pandas DataFrame
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)

        # --- Apply Filters using Pandas ---
        
        # Filter by BDC if provided and not 'all'
        if bdc_filter and bdc_filter.lower() != 'all':
            df = df[df['bdc'] == bdc_filter]

        # Filter by Borrower if provided and not 'all'
        if borrower_filter and borrower_filter.lower() != 'all':
            df = df[df['borrower'] == borrower_filter]

        # Convert back to list of dicts for JSON response
        # orient='records' creates the format: [{col1: val1}, {col1: val2}...]
        return df.to_dict(orient='records')
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return []

def get_unique_options():
    """
    Helper to get unique BDCs and Borrowers for your frontend dropdowns
    """
    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        
        return {
            "bdcs": df[['bdc', 'bdcName']].drop_duplicates().to_dict(orient='records'),
            "borrowers": df[['borrower', 'borrowerName']].drop_duplicates().to_dict(orient='records')
        }
    except Exception:
        return {}