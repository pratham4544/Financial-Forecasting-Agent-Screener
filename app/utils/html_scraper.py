import requests
from bs4 import BeautifulSoup
from typing import List
from urllib.parse import urljoin

def find_document_links(screener_company_url: str) -> List[str]:
    r = requests.get(screener_company_url, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    docs_section = soup.find(id="documents") or soup
    links = []
    for a in docs_section.select("a"):
        href = a.get("href", "")
        if href and href.lower().endswith(".pdf"):
            links.append(urljoin(screener_company_url, href))
    # unique preserve order
    return list(dict.fromkeys(links))
