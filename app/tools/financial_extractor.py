"""
FinancialDataExtractorTool: Extracts key financial metrics from quarterly reports.

This tool is designed to:
1. Parse financial documents (PDFs) and web pages
2. Extract structured financial metrics (revenue, profit, margins, etc.)
3. Calculate growth rates and trends
4. Return data in a structured format for the agent to analyze
"""
import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
from langchain_core.tools import Tool

from app.config import settings
from app.utils.scraper import extract_financial_table, extract_company_symbol
from app.models.schemas import FinancialMetrics

logger = logging.getLogger(__name__)


class FinancialDataExtractorTool:
    """
    A robust tool for extracting financial metrics from quarterly reports.

    This tool combines multiple extraction strategies:
    - Web scraping from Screener.in tables
    - PDF text extraction and parsing
    - Pattern matching for key financial indicators
    """

    def __init__(self):
        self.name = "FinancialDataExtractor"
        self.description = """
        Extracts key financial metrics from quarterly reports.
        Input should be a company URL or path to financial PDF.
        Returns structured financial data including revenue, profit, margins, and growth rates.
        """

    def extract_from_url(self, company_url: str, quarters: int = 2) -> List[Dict[str, Any]]:
        """
        Extract financial metrics from Screener.in URL.

        Args:
            company_url: URL to company's Screener.in page
            quarters: Number of recent quarters to extract

        Returns:
            List of dictionaries with financial metrics per quarter
        """
        logger.info(f"Extracting financial data from URL: {company_url}")

        try:
            # Extract financial table from web page
            financial_data = extract_financial_table(company_url)

            if not financial_data:
                logger.warning("No financial data extracted from URL")
                return []

            # Transform raw data into structured metrics
            metrics = self._transform_table_to_metrics(financial_data, quarters)
            return metrics

        except Exception as e:
            logger.error(f"Error extracting from URL: {e}")
            return []

    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract financial metrics from PDF document.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with extracted financial metrics
        """
        logger.info(f"Extracting financial data from PDF: {pdf_path}")

        try:
            # Try multiple PDF extraction methods
            text = self._extract_text_from_pdf(pdf_path)

            if not text:
                logger.warning(f"No text extracted from PDF: {pdf_path}")
                return {}

            # Parse text for financial metrics
            metrics = self._parse_financial_text(text)
            return metrics

        except Exception as e:
            logger.error(f"Error extracting from PDF {pdf_path}: {e}")
            return {}

    def extract_from_pdfs(self, pdf_directory: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract financial metrics from all financial PDFs in directory.

        Args:
            pdf_directory: Directory containing PDFs (defaults to settings.PDF_DIR)

        Returns:
            List of metrics dictionaries from all financial reports
        """
        pdf_dir = Path(pdf_directory) if pdf_directory else settings.PDF_DIR

        if not pdf_dir.exists():
            logger.warning(f"PDF directory does not exist: {pdf_dir}")
            return []

        all_metrics = []

        # Only process PDFs classified as financial reports
        for pdf_file in pdf_dir.glob("*.pdf"):
            # Prioritize financial_report PDFs
            if "financial_report" in pdf_file.name.lower() or "result" in pdf_file.name.lower():
                metrics = self.extract_from_pdf(str(pdf_file))
                if metrics:
                    metrics["source_file"] = pdf_file.name
                    all_metrics.append(metrics)

        logger.info(f"Extracted metrics from {len(all_metrics)} financial PDFs")
        return all_metrics

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using multiple strategies."""
        text = ""

        # Strategy 1: Try PyPDF2
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages[:10]:  # First 10 pages usually have financials
                text += page.extract_text() + "\n"
            if text.strip():
                return text
        except Exception as e:
            logger.debug(f"PyPDF2 extraction failed: {e}")

        # Strategy 2: Try pdfminer
        try:
            text = extract_text(pdf_path, maxpages=10)
            if text.strip():
                return text
        except Exception as e:
            logger.debug(f"pdfminer extraction failed: {e}")

        return text

    def _parse_financial_text(self, text: str) -> Dict[str, Any]:
        """
        Parse financial text to extract key metrics using pattern matching.

        Args:
            text: Raw text from financial document

        Returns:
            Dictionary with extracted metrics
        """
        metrics = {}

        # Patterns for key financial metrics
        patterns = {
            "total_revenue": [
                r"total\s+revenue[:\s]+(?:Rs\.?\s*)?([0-9,]+\.?[0-9]*)\s*(?:crore|cr)",
                r"revenue\s+from\s+operations[:\s]+(?:Rs\.?\s*)?([0-9,]+\.?[0-9]*)",
            ],
            "net_profit": [
                r"net\s+profit[:\s]+(?:Rs\.?\s*)?([0-9,]+\.?[0-9]*)\s*(?:crore|cr)",
                r"profit\s+after\s+tax[:\s]+(?:Rs\.?\s*)?([0-9,]+\.?[0-9]*)",
            ],
            "operating_margin": [
                r"operating\s+margin[:\s]+([0-9]+\.?[0-9]*)\s*%",
                r"EBIT\s+margin[:\s]+([0-9]+\.?[0-9]*)\s*%",
            ],
            "eps": [
                r"earnings\s+per\s+share[:\s]+(?:Rs\.?\s*)?([0-9]+\.?[0-9]*)",
                r"EPS[:\s]+(?:Rs\.?\s*)?([0-9]+\.?[0-9]*)",
            ]
        }

        # Apply patterns
        text_lower = text.lower()

        for metric_name, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    try:
                        value_str = match.group(1).replace(",", "")
                        metrics[metric_name] = float(value_str)
                        break
                    except (ValueError, IndexError):
                        continue

        # Try to identify quarter
        quarter_match = re.search(r"(Q[1-4])\s*(FY)?'?(\d{2,4})", text, re.IGNORECASE)
        if quarter_match:
            metrics["quarter"] = quarter_match.group(0)

        return metrics

    def _transform_table_to_metrics(self, table_data: List[Dict[str, Any]], quarters: int) -> List[Dict[str, Any]]:
        """
        Transform scraped table data into structured FinancialMetrics format.

        Args:
            table_data: Raw table data from scraper
            quarters: Number of quarters to include

        Returns:
            List of structured metrics dictionaries
        """
        # Identify quarter columns (skip first column which is metric names)
        if not table_data:
            return []

        # Get all column names (quarters)
        first_row = table_data[0]
        quarter_columns = [k for k in first_row.keys() if k != "metric"][:quarters]

        # Build metrics for each quarter
        metrics_by_quarter = []

        for quarter_col in quarter_columns:
            quarter_metrics = {"quarter": quarter_col}

            # Extract relevant metrics
            for row in table_data:
                metric_name = row.get("metric", "").lower()
                value = row.get(quarter_col)

                # Map metric names to standard fields
                if "revenue" in metric_name or "sales" in metric_name:
                    if "total" in metric_name or "operations" in metric_name:
                        quarter_metrics["total_revenue"] = value

                elif "profit" in metric_name:
                    if "net" in metric_name or "after tax" in metric_name:
                        quarter_metrics["net_profit"] = value

                elif "margin" in metric_name:
                    if "operating" in metric_name or "opm" in metric_name:
                        quarter_metrics["operating_margin"] = value

                elif "eps" in metric_name:
                    quarter_metrics["eps"] = value

            metrics_by_quarter.append(quarter_metrics)

        # Calculate growth rates
        metrics_by_quarter = self._calculate_growth_rates(metrics_by_quarter)

        return metrics_by_quarter

    def _calculate_growth_rates(self, metrics_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate YoY growth rates if sufficient data is available."""
        if len(metrics_list) < 2:
            return metrics_list

        # Calculate growth for latest quarter
        latest = metrics_list[0]
        previous = metrics_list[1]

        if "total_revenue" in latest and "total_revenue" in previous:
            if previous["total_revenue"] and previous["total_revenue"] != 0:
                growth = ((latest["total_revenue"] - previous["total_revenue"]) /
                          previous["total_revenue"]) * 100
                latest["revenue_growth_yoy"] = round(growth, 2)

        if "net_profit" in latest and "net_profit" in previous:
            if previous["net_profit"] and previous["net_profit"] != 0:
                growth = ((latest["net_profit"] - previous["net_profit"]) /
                          previous["net_profit"]) * 100
                latest["profit_growth_yoy"] = round(growth, 2)

        return metrics_list

    def run(self, input_str: str) -> str:
        """
        Run the tool with given input.
        Used by LangChain agent.

        Args:
            input_str: Either a URL or comma-separated "url,quarters"

        Returns:
            String representation of extracted metrics
        """
        try:
            # Parse input
            parts = input_str.split(",")
            url = parts[0].strip()
            quarters = int(parts[1].strip()) if len(parts) > 1 else 2

            # Extract from URL
            metrics = self.extract_from_url(url, quarters)

            if not metrics:
                # Try extracting from PDFs as fallback
                metrics = self.extract_from_pdfs()

            if not metrics:
                return "No financial metrics could be extracted."

            # Format output for LLM
            output = "Financial Metrics Extracted:\n\n"
            for i, quarter_data in enumerate(metrics[:quarters], 1):
                output += f"Quarter {i}: {quarter_data.get('quarter', 'Unknown')}\n"
                output += f"  Total Revenue: {quarter_data.get('total_revenue', 'N/A')} Cr\n"
                output += f"  Net Profit: {quarter_data.get('net_profit', 'N/A')} Cr\n"
                output += f"  Operating Margin: {quarter_data.get('operating_margin', 'N/A')}%\n"
                output += f"  EPS: {quarter_data.get('eps', 'N/A')}\n"

                if "revenue_growth_yoy" in quarter_data:
                    output += f"  Revenue Growth YoY: {quarter_data['revenue_growth_yoy']}%\n"
                if "profit_growth_yoy" in quarter_data:
                    output += f"  Profit Growth YoY: {quarter_data['profit_growth_yoy']}%\n"

                output += "\n"

            return output

        except Exception as e:
            logger.error(f"Error in FinancialDataExtractorTool.run: {e}")
            return f"Error extracting financial data: {str(e)}"

    def as_langchain_tool(self) -> Tool:
        """Convert to LangChain Tool for use in agent."""
        return Tool(
            name=self.name,
            description=self.description,
            func=self.run
        )
