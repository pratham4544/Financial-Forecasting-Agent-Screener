"""
PDF downloading and classification utilities.
Handles downloading PDFs from various sources and classifying them as
transcripts, presentations, or financial reports.
"""
import os
import re
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from typing import List, Tuple, Optional
from pathlib import Path
import logging

from app.config import settings

logger = logging.getLogger(__name__)


def download_pdfs(pdf_links: List[Tuple[str, str]], max_pdfs: int = 10) -> List[str]:
    """
    Download PDFs from provided links and save to data directory.

    Args:
        pdf_links: List of (url, filename) tuples
        max_pdfs: Maximum number of PDFs to download

    Returns:
        List of paths to downloaded PDF files
    """
    downloaded_files = []

    for i, (url, filename) in enumerate(pdf_links[:max_pdfs]):
        try:
            logger.info(f"Downloading PDF {i+1}/{min(len(pdf_links), max_pdfs)}: {filename}")

            # Determine download strategy based on URL type
            if "AnnPdfOpen.aspx" in url:
                filepath = _download_bse_annpdf(url, filename)
            elif "xml-data/corpfiling" in url:
                filepath = _download_bse_iframe_pdf(url, filename)
            else:
                filepath = _download_direct_pdf(url, filename)

            if filepath:
                # Classify and potentially rename
                classified_path = classify_and_rename_pdf(filepath)
                downloaded_files.append(classified_path)

        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            continue

    logger.info(f"Successfully downloaded {len(downloaded_files)} PDFs")
    return downloaded_files


def _download_direct_pdf(url: str, filename: str) -> Optional[str]:
    """Download PDF directly from URL."""
    try:
        # Make URL absolute if needed
        if not url.startswith("http"):
            url = "https://www.screener.in" + url

        response = requests.get(
            url,
            headers={"User-Agent": settings.USER_AGENT},
            timeout=settings.REQUEST_TIMEOUT
        )
        response.raise_for_status()

        # Save to file
        filepath = settings.PDF_DIR / clean_filename(filename)
        with open(filepath, "wb") as f:
            f.write(response.content)

        logger.info(f"Downloaded: {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"Error in direct download: {e}")
        return None


def _download_bse_annpdf(url: str, filename: str) -> Optional[str]:
    """Download PDF from BSE AnnPdfOpen.aspx format."""
    try:
        response = requests.get(
            url,
            headers={"User-Agent": settings.USER_AGENT},
            timeout=settings.REQUEST_TIMEOUT
        )
        response.raise_for_status()

        filepath = settings.PDF_DIR / clean_filename(filename)
        with open(filepath, "wb") as f:
            f.write(response.content)

        logger.info(f"Downloaded BSE AnnPdf: {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"Error downloading BSE AnnPdf: {e}")
        return None


def _download_bse_iframe_pdf(url: str, filename: str) -> Optional[str]:
    """Download PDF from BSE iframe format (requires extracting actual PDF URL)."""
    try:
        # First, get the page with the iframe
        response = requests.get(
            url,
            headers={"User-Agent": settings.USER_AGENT},
            timeout=settings.REQUEST_TIMEOUT
        )
        response.raise_for_status()

        # Parse HTML to find iframe
        soup = BeautifulSoup(response.text, "html.parser")
        iframe = soup.find("iframe")

        if not iframe:
            logger.warning("No iframe found in BSE page")
            return None

        # Get actual PDF URL from iframe src
        pdf_url = iframe.get("src")
        if not pdf_url.startswith("http"):
            pdf_url = "https://www.bseindia.com" + pdf_url

        # Download the actual PDF
        pdf_response = requests.get(
            pdf_url,
            headers={"User-Agent": settings.USER_AGENT},
            timeout=settings.REQUEST_TIMEOUT
        )
        pdf_response.raise_for_status()

        filepath = settings.PDF_DIR / clean_filename(filename)
        with open(filepath, "wb") as f:
            f.write(pdf_response.content)

        logger.info(f"Downloaded BSE iframe PDF: {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"Error downloading BSE iframe PDF: {e}")
        return None


def classify_pdf_type(pdf_path: str) -> Optional[str]:
    """
    Classify PDF as 'transcript', 'presentation', or 'financial_report'.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Classification type or None if unable to classify
    """
    try:
        reader = PdfReader(pdf_path)

        # Extract text from first few pages
        text_sample = ""
        for page_num in range(min(3, len(reader.pages))):
            text_sample += reader.pages[page_num].extract_text().lower()

        # Classification keywords
        if any(keyword in text_sample for keyword in ["transcript", "earnings call", "conference call"]):
            return "transcript"

        if any(keyword in text_sample for keyword in ["presentation", "investor presentation", "investor deck"]):
            return "presentation"

        if any(keyword in text_sample for keyword in ["financial results", "quarterly results", "annual report"]):
            return "financial_report"

        return None

    except Exception as e:
        logger.error(f"Error classifying PDF {pdf_path}: {e}")
        return None


def classify_and_rename_pdf(pdf_path: str) -> str:
    """
    Classify PDF and add type prefix to filename.

    Args:
        pdf_path: Path to PDF file

    Returns:
        New path (may be same as original if not renamed)
    """
    doc_type = classify_pdf_type(pdf_path)

    if not doc_type:
        return pdf_path

    # Add type prefix to filename
    path_obj = Path(pdf_path)
    new_name = f"{doc_type}_{path_obj.name}"
    new_path = path_obj.parent / new_name

    # Avoid overwriting existing files
    counter = 1
    while new_path.exists() and new_path != path_obj:
        new_name = f"{doc_type}_{counter}_{path_obj.name}"
        new_path = path_obj.parent / new_name
        counter += 1

    try:
        if new_path != path_obj:
            os.rename(pdf_path, new_path)
            logger.info(f"Classified as '{doc_type}' and renamed to: {new_path.name}")
            return str(new_path)
    except Exception as e:
        logger.error(f"Error renaming file: {e}")

    return pdf_path


def clean_filename(filename: str) -> str:
    """
    Clean filename by removing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Cleaned filename safe for file system
    """
    # Remove invalid characters
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)

    # Replace multiple underscores with single
    cleaned = re.sub(r'_+', '_', cleaned)

    # Remove leading/trailing spaces and dots
    cleaned = cleaned.strip(' .')

    # Limit length
    if len(cleaned) > 200:
        name, ext = os.path.splitext(cleaned)
        cleaned = name[:195] + ext

    return cleaned
