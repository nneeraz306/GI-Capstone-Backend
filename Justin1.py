import requests
import pandas as pd
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Kirthick kirthick31@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

url = "https://www.sec.gov/Archives/edgar/data/1675033/000119312525264873/gecc-20250930.htm"

html = requests.get(url, headers=headers).text
soup = BeautifulSoup(html, "html.parser")

tables = soup.find_all("table")
print(tables)
print(len(tables))
