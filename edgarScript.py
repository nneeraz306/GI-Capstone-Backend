from edgar import Company, set_identity

# 1. REQUIRED: Set your identity (User-Agent) as per SEC rules
# Format: "Name email@domain.com"
set_identity("StudentResearcher myname@example.com")

# 2. Initialize the Company using its Ticker (TCPC)
# You could also use CIK='0001377936'
bdc = Company("TCPC")
print(f"Connected to: {bdc.name} (CIK: {bdc.cik})")

# 3. Get the latest 10-K filing (Annual Report)
# BDCs file 10-Ks just like regular companies
filings = bdc.get_filings(form="10-K")
latest_10k = filings.latest()

print(f"Pulling filing date: {latest_10k.filing_date}")

# 4. Open the filing to access content
# This automatically parses the HTML into workable objects
doc = latest_10k.obj()

# 5. Extract specific data
# Example A: Get the "Consolidated Schedule of Investments"
# This is the most important table for a BDC (lists all loans/equity they hold)
print("\n--- Searching for Investment Schedule ---")
# We search for the specific Item usually found in BDC 10-Ks
Item6 = doc['Item 6'] 
print(Item6) # Prints first 500 characters of the financial section

# # Example B: Extract "Risk Factors" (Item 1A)
# risk_factors = doc['Item 1A']
# print("\n--- Sample Risk Factor ---")
# print(risk_factors[:500])