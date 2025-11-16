from typing import Dict, Any
from bs4 import BeautifulSoup
import requests
import re

def extract_financials_from_screener(company_url: str, quarters: int = 2) -> Dict[str, Any]:
    """
    Best-effort scrape of screener.in tables for numeric fields.
    Returns {"revenue": [...], "net_income": [...], "operating_margin": [...]}
    """
    r = requests.get(company_url, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    data = {"revenue": [], "net_income": [], "operating_margin": []}

    # Naive: find tables that mention revenue
    tables = soup.select("table")
    for t in tables:
        text = t.get_text().lower()
        if "total revenue" in text or "revenue" in text:
            # extract numbers (simple)
            nums = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?", t.get_text())
            nums = [float(n.replace(",", "")) for n in nums]
            if nums:
                data["revenue"] = nums[:quarters]
                break

    # fallback heuristics can be added - for now return what we found
    return data
