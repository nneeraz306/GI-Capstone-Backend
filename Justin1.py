import requests
import pandas as pd
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Kirthick kirthick31@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

# url = "https://www.sec.gov/Archives/edgar/data/1675033/000119312525264873/gecc-20250930.htm"
url = "https://www.sec.gov/Archives/edgar/data/0001925309/000119312525266848/ck0001925309-20250930.htm"
bdc_html = requests.get(url, headers=headers).text
soup = BeautifulSoup(bdc_html, "lxml")

soi_p_tag = soup.find("p", id="consolidated_schedule_investments")
soi_table = soi_p_tag.find_next("table")

soi_df = pd.read_html(str(soi_table))[0]
soi_df.to_csv("soi_table2.csv", index=False)

