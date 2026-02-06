"""
BDC Portfolio Data Extraction Pipeline

This script orchestrates the complete workflow:
1. Extract SOI tables from SEC filings (Justin1.py)
2. Normalize the extracted CSV files (already exists if using existing normalize.py)
3. Map to output format (mapping.py)
"""

import os
import sys
import glob
from datetime import datetime

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def run_extraction():
    """Run the extraction script"""
    print_header("STEP 1: EXTRACTING SOI TABLES FROM SEC FILINGS")
    
    try:
        # Import and run Justin1.py
        import Justin1
        
        # Check if CSV files were created
        csv_files = glob.glob("soi_table*.csv")
        if csv_files:
            print(f"✓ Extraction completed successfully")
            print(f"  Created {len(csv_files)} CSV files")
            return True
        else:
            print("✗ No CSV files were created")
            return False
    except Exception as e:
        print(f"✗ Extraction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_mapping():
    """Run the mapping script directly on extracted CSVs"""
    print_header("STEP 2: MAPPING TO OUTPUT FORMAT")
    
    try:
        from mapping import map_all_files
        
        # Map directly from extracted CSV files
        output_file = map_all_files(normalized_dir=".", output_file="mapped_portfolio_data.json")
        
        if output_file and os.path.exists(output_file):
            print(f"✓ Mapping completed successfully")
            print(f"  Output file: {output_file}")
            
            # Show sample of output
            import json
            with open(output_file, 'r') as f:
                data = json.load(f)
                print(f"  Total records: {len(data)}")
                if data:
                    print(f"  Sample record:")
                    print(f"    BDC: {data[0].get('bdcName')}")
                    print(f"    Borrower: {data[0].get('borrowerName')}")
            
            return True, output_file
        else:
            print("✗ Mapping failed - output file not created")
            return False, None
    except Exception as e:
        print(f"✗ Mapping failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """Run the complete pipeline"""
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("  BDC PORTFOLIO DATA EXTRACTION PIPELINE")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    
    # Step 1: Extract
    if not run_extraction():
        print("\n❌ Pipeline failed at extraction step")
        sys.exit(1)
    
    # Step 2: Map (normalize logic is within mapping if needed)
    success, output_file = run_mapping()
    if not success:
        print("\n❌ Pipeline failed at mapping step")
        sys.exit(1)
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print_header("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"  Total duration: {duration:.2f} seconds")
    print(f"  Final output: {output_file}")
    print(f"  Location: {os.path.abspath(output_file)}")
    print("\n✓ All steps completed successfully!\n")

if __name__ == "__main__":
    main()
