from nse_live_stocks import Nse
from typing import Dict, Any

def get_live_price(symbol: str = "TCS") -> Dict[str, Any]:
    stock = Nse()
    result = stock.get_current_price(symbol)
    if not result:
        return {"symbol": symbol, "error": True, "message": "no response"}
    if result.get("error"):
        return {"symbol": symbol, "error": True, "message": result.get("message", "error")}
    # normalize
    try:
        price = float(result.get("current_value"))
    except Exception:
        price = None
    return {
        "symbol": result.get("nse_symbol", symbol),
        "current_price": price,
        "timestamp": result.get("date"),
        "error": result.get("error", False)
    }
