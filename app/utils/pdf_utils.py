import requests
from pathlib import Path
import tempfile
from typing import List
from pdfminer.high_level import extract_text

def download_pdfs(urls: List[str]) -> List[str]:
    out_paths = []
    tmp = Path(tempfile.gettempdir()) / "tcs_agent_pdfs"
    tmp.mkdir(parents=True, exist_ok=True)
    for u in urls:
        r = requests.get(u, timeout=30)
        r.raise_for_status()
        filename = tmp / Path(u).name
        with open(filename, "wb") as f:
            f.write(r.content)
        out_paths.append(str(filename))
    return out_paths

def extract_pdf_text(path: str) -> str:
    try:
        return extract_text(path)
    except Exception:
        return ""
