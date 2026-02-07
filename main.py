# import os
# from normalization_l1 import run_header_consolidation

# # --- Configuration ---
# INPUT_FILE = 'soi_table0.csv'  # Replace with your actual csv file name
# L1_OUTPUT = 'data_l0_consolidated.csv'

# def main():
#     # Check if input file exists
#     if not os.path.exists(INPUT_FILE):
#         print(f"Error: {INPUT_FILE} not found.")
#         return

#     # --- Step 1: Header Consolidation ---
#     try:
#         run_header_consolidation(INPUT_FILE, L1_OUTPUT)
#     except Exception as e:
#         print(f"Error in Layer 1: {e}")
#         return

#     # --- Future Steps (Placeholder) ---
#     # from normalization_l2 import run_atomization
#     # run_atomization(L1_OUTPUT, 'data_l2_atomized.csv')

#     print("\nPipeline finished successfully.")

# if __name__ == "__main__":
#     main()


import os
# Import Layer 1
from normalization_l1 import run_header_consolidation
# Import Layer 2 (The new file)
from normalization_l2 import run_security_normalization

# --- Configuration ---
INPUT_FILE = 'soi_table0.csv'  # Your original raw file
L1_OUTPUT = 'data_l0_consolidated.csv' # Output of Layer 1 / Input for Layer 2
L2_OUTPUT = 'data_l0_normalized.csv'   # Output of Layer 2

def main():
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    # --- Step 1: Header Consolidation ---
    try:
        # This cleans headers and removes ghost columns
        run_header_consolidation(INPUT_FILE, L1_OUTPUT)
    except Exception as e:
        print(f"Error in Layer 1: {e}")
        return

    # --- Step 2: Security Column Normalization ---
    try:
        # This takes the clean file from Step 1 and splits the Security column
        run_security_normalization(L1_OUTPUT, L2_OUTPUT)
    except Exception as e:
        print(f"Error in Layer 2: {e}")
        return

    print("\nPipeline finished successfully.")
    print(f"Final file saved as: {L2_OUTPUT}")

if __name__ == "__main__":
    main()