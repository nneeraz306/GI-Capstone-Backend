import requests
import pandas as pd
from bs4 import BeautifulSoup
import re

def extract_soi_table(url, num):
    print(num)
    bdc_html = requests.get(url, headers=headers).text
    soup = BeautifulSoup(bdc_html, "lxml")

    # soi_p_tag = soup.find("p", id="consolidated_schedule_investments")
    soi_p_tag = soup.find("p", id=re.compile(r'^(?=.*consolidated)(?=.*schedule)(?=.*investments).*', re.IGNORECASE))
    soi_table = soi_p_tag.find_next("table")

    soi_df = pd.read_html(str(soi_table))[0]
    soi_df.to_csv(f"soi_table{num}.csv", index=False)
    print(f"soi_table{num}.csv saved successfully!")


def is_soi_id(id):
    keywords = ["consolidated", "schedule", "investments"]
    for keyword in keywords:
        if keyword not in id.lower():
            return False
    return True


headers = {
    "User-Agent": "Kirthick kirthick31@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

urls = [
        "https://www.sec.gov/Archives/edgar/data/1675033/000119312525264873/gecc-20250930.htm",
        "https://www.sec.gov/Archives/edgar/data/0001925309/000119312525266848/ck0001925309-20250930.htm",
        "https://www.sec.gov/Archives/edgar/data/0001370755/000119312525268125/tcpc-20250930.htm",
        "https://www.sec.gov/Archives/edgar/data/0001487428/000143774925031988/hrzn20250930_10q.htm"
    ]

for i, url in enumerate(urls):
    extract_soi_table(url, i)

