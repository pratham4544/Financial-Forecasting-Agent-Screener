"""
MarketDataTool: Fetches live market data for stocks.

This tool:
1. Retrieves current stock price from NSE
2. Provides market context for the forecast
3. Can be extended to include more market metrics
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from langchain_core.tools import Tool

logger = logging.getLogger(__name__)


class MarketDataTool:
    """
    Tool for fetching live market data.

    Uses NSE API (via nse-live-stocks or similar) to get current stock prices.
    """

    def __init__(self):
        self.name = "MarketData"
        self.description = """
        Fetches live market data for a given stock symbol.
        Input should be a stock symbol (e.g., "TCS", "INFY").
        Returns current price and basic market information.
        """

    def get_live_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get live stock price for given symbol.

        Args:
            symbol: Stock symbol (e.g., "TCS")

        Returns:
            Dictionary with market data
        """
        logger.info(f"Fetching market data for: {symbol}")

        try:
            # Try using nse-live-stocks library
            from nse_live_stocks import Nse

            nse = Nse()
            data = nse.get_current_price(symbol)

            if data and "current_value" in data:
                return {
                    "symbol": symbol,
                    "current_price": data["current_value"],
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "success",
                    "additional_data": data
                }
            else:
                logger.warning(f"No price data returned for {symbol}")
                return {
                    "symbol": symbol,
                    "status": "no_data",
                    "error": "No price data available"
                }

        except ImportError:
            logger.error("nse-live-stocks library not available")
            return self._fallback_market_data(symbol)

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return {
                "symbol": symbol,
                "status": "error",
                "error": str(e)
            }

    def _fallback_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fallback method when primary data source is unavailable.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with fallback message
        """
        logger.warning(f"Using fallback for {symbol}")

        return {
            "symbol": symbol,
            "status": "fallback",
            "message": f"Live market data unavailable. Install nse-live-stocks for real-time prices.",
            "timestamp": datetime.utcnow().isoformat()
        }

    def get_market_context(self, symbol: str) -> str:
        """
        Get market context as formatted string.

        Args:
            symbol: Stock symbol

        Returns:
            Formatted string with market information
        """
        data = self.get_live_price(symbol)

        if data.get("status") == "success":
            output = f"Live Market Data for {symbol}:\n"
            output += f"  Current Price: ₹{data['current_price']}\n"
            output += f"  Timestamp: {data['timestamp']}\n"

            # Add additional metrics if available
            additional = data.get("additional_data", {})
            if additional:
                if "change" in additional:
                    output += f"  Change: {additional['change']}\n"
                if "pChange" in additional:
                    output += f"  % Change: {additional['pChange']}%\n"

            return output

        elif data.get("status") == "fallback":
            return data.get("message", "Market data unavailable")

        else:
            error_msg = data.get("error", "Unknown error")
            return f"Unable to fetch market data for {symbol}: {error_msg}"

    def run(self, input_str: str) -> str:
        """
        Run the tool with given input.
        Used by LangChain agent.

        Args:
            input_str: Stock symbol

        Returns:
            String with market data
        """
        try:
            # Clean input
            symbol = input_str.strip().upper()

            # Get market context
            result = self.get_market_context(symbol)

            return result

        except Exception as e:
            logger.error(f"Error in MarketDataTool.run: {e}")
            return f"Error fetching market data: {str(e)}"

    def as_langchain_tool(self) -> Tool:
        """Convert to LangChain Tool for use in agent."""
        return Tool(
            name=self.name,
            description=self.description,
            func=self.run
        )
