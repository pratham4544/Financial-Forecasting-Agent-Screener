from sqlalchemy import Column, Integer, Text, DateTime
from datetime import datetime
from app.db import Base
from pydantic import BaseModel
from typing import Any, Dict, Optional

class RequestLog(Base):
    __tablename__ = "request_logs"
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    company = Column(Text)
    request_payload = Column(Text)
    response_payload = Column(Text)

# Pydantic models for endpoint
class ForecastRequest(BaseModel):
    company_url: str
    quarters: int = 2

class ForecastResponse(BaseModel):
    company: str
    quarters_analyzed: int
    financial_metrics: Dict[str, Any]
    qualitative_summary: Dict[str, Any]
    risks_and_opportunities: Dict[str, Any]
    market_price: Optional[Dict[str, Any]] = None
    forecast: Dict[str, Any]
