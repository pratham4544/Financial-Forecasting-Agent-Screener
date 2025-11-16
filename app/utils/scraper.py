"""
Web scraping utilities for Screener.in and financial data extraction.
Handles HTML parsing and data extraction from financial websites.
"""
import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple, Optional, Any
from app.config import settings
import logging

logger = logging.getLogger(__name__)


def scrape_screener_pdfs(company_url: str) -> List[Tuple[str, str]]:
    """
    Scrape PDF links from Screener.in company documents page.

    Args:
        company_url: URL to the company's Screener.in page

    Returns:
        List of tuples containing (pdf_url, suggested_filename)
    """
    logger.info(f"Scraping PDFs from: {company_url}")

    try:
        headers = {"User-Agent": settings.USER_AGENT}
        response = requests.get(
            company_url,
            headers=headers,
            timeout=settings.REQUEST_TIMEOUT
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Find all PDF links in the documents section
        pdf_links = []

        # Try multiple selectors for robustness
        selectors = [
            ".documents a",  # Primary selector
            "#documents a",  # Alternative ID-based
            "a[href$='.pdf']"  # Generic PDF link
        ]

        for selector in selectors:
            links = soup.select(selector)
            if links:
                break

        for link in links:
            href = link.get("href", "")
            if href and (".pdf" in href.lower() or "AnnPdf" in href):
                # Clean up the link text to create filename
                text = link.text.strip().replace("\n", "_").replace(" ", "_")
                text = re.sub(r'[^\w\-_.]', '_', text)

                if not text or text == "_":
                    text = "Document"

                # Ensure .pdf extension
                if not text.endswith(".pdf"):
                    text += ".pdf"

                pdf_links.append((href, text))

        logger.info(f"Found {len(pdf_links)} PDF links")
        return pdf_links

    except Exception as e:
        logger.error(f"Error scraping PDFs: {e}")
        return []


def extract_financial_table(company_url: str) -> List[Dict[str, Any]]:
    """
    Extract financial metrics table from Screener.in.

    Args:
        company_url: URL to the company's Screener.in page

    Returns:
        List of dictionaries containing financial metrics by quarter
    """
    logger.info(f"Extracting financial table from: {company_url}")

    try:
        headers = {"User-Agent": settings.USER_AGENT}
        response = requests.get(
            company_url,
            headers=headers,
            timeout=settings.REQUEST_TIMEOUT
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Look for quarterly results table
        financial_data = []

        # Try to find the quarterly results section
        quarterly_section = soup.find("section", {"id": "quarters"})

        if not quarterly_section:
            logger.warning("Quarterly results section not found")
            return financial_data

        # Find the table with quarterly data
        table = quarterly_section.find("table")

        if not table:
            logger.warning("Financial table not found")
            return financial_data

        # Extract headers
        headers_row = table.find("thead")
        if not headers_row:
            headers_row = table.find("tr")

        headers = []
        if headers_row:
            for th in headers_row.find_all(["th", "td"]):
                headers.append(th.text.strip())

        # Extract data rows
        tbody = table.find("tbody") or table
        rows = tbody.find_all("tr")[1:] if headers_row else tbody.find_all("tr")

        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue

            # First cell is usually the metric name
            metric_name = cells[0].text.strip()

            # Remaining cells are values for different quarters
            row_data = {"metric": metric_name}

            for i, cell in enumerate(cells[1:], 1):
                if i <= len(headers) - 1:
                    quarter_key = headers[i] if i < len(headers) else f"Q{i}"
                    value_text = cell.text.strip()

                    # Try to parse numeric value
                    value = parse_financial_value(value_text)
                    row_data[quarter_key] = value

            financial_data.append(row_data)

        logger.info(f"Extracted {len(financial_data)} rows of financial data")
        return financial_data

    except Exception as e:
        logger.error(f"Error extracting financial table: {e}")
        return []


def parse_financial_value(value_str: str) -> Optional[float]:
    """
    Parse financial value string to float.
    Handles formats like: "1,234.56", "5.2%", "12.3 Cr", etc.

    Args:
        value_str: String representation of financial value

    Returns:
        Parsed float value or None if parsing fails
    """
    if not value_str or value_str in ["-", "N/A", ""]:
        return None

    try:
        # Remove common separators and units
        cleaned = value_str.replace(",", "").replace("%", "").strip()

        # Handle special cases
        if "Cr" in cleaned or "cr" in cleaned:
            cleaned = cleaned.replace("Cr", "").replace("cr", "").strip()

        # Try to convert to float
        return float(cleaned)

    except (ValueError, AttributeError):
        return None


def extract_company_symbol(company_url: str) -> str:
    """
    Extract company symbol from Screener.in URL.

    Args:
        company_url: URL like "https://www.screener.in/company/TCS/consolidated/"

    Returns:
        Company symbol (e.g., "TCS")
    """
    match = re.search(r'/company/([^/]+)/', company_url)
    if match:
        return match.group(1)
    return "UNKNOWN"
