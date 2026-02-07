import pandas as pd
import numpy as np

def run_security_normalization(input_path, output_path):
    print(f"--- Running Layer 2: Security Column Normalization on {input_path} ---")

    # 1. Read the CSV (Input comes from Layer 1 output)
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File {input_path} not found.")
        return

    # 2. Check Validation: Ensure 'Security' Column Exists
    if 'Security' not in df.columns:
        print("Warning: 'Security' column missing. Skipping split logic.")
        df.to_csv(output_path, index=False)
        return

    # 3. Define the Split Logic
    def split_security_col(val):
        val_str = str(val).strip()
        
        if ',' in val_str:
            # Split only on the FIRST comma found
            parts = val_str.split(',', 1) 
            return parts[0].strip(), parts[1].strip()
        else:
            # No comma found? Return original value as Ranking, empty string as Type
            return val_str, ""

    # 4. Apply Logic to Create New Columns
    print("Splitting 'Security' into 'Seniority Ranking' and 'Secured/Subordinated'...")
    df['Seniority Ranking'], df['Secured/Subordinated'] = zip(*df['Security'].apply(split_security_col))

    # 5. Reorder Columns & Drop 'Security'
    # Get current list of columns
    cols = list(df.columns)
    
    # Find where 'Security' is currently located
    security_idx = cols.index('Security')

    # Remove the new columns from the end (where pandas added them automatically)
    cols.remove('Seniority Ranking')
    cols.remove('Secured/Subordinated')
    
    # Insert the new columns exactly where 'Security' was
    # We insert them at security_idx, pushing 'Security' to the right
    cols.insert(security_idx, 'Seniority Ranking')
    cols.insert(security_idx + 1, 'Secured/Subordinated')
    
    # Now remove the original 'Security' column from the list
    cols.remove('Security')
    
    # Re-index the dataframe (This drops 'Security' and applies the new order)
    df = df[cols]

    # 6. Save the Normalized File
    df.to_csv(output_path, index=False)
    print(f"Layer 2 Complete. 'Security' dropped. Data saved to: {output_path}")``