"""
FastAPI application for TCS Financial Forecasting Agent.

This is the main entry point for the API service.
Provides endpoints for:
- Generating financial forecasts
- Retrieving historical requests
- Health checks
"""
import logging
import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import settings
from app.models import (
    Base,
    engine,
    get_db,
    ForecastRequest,
    ForecastResponse,
    ForecastOutput,
    FinancialMetrics,
    QualitativeInsights,
    MarketData
)
from app.models.database import init_db
from app.services import AgentPipeline, DatabaseService
from app.utils.scraper import extract_company_symbol

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize database
try:
    init_db()
    logger.info("Database initialized successfully")
except Exception as e:
    logger.error(f"Error initializing database: {e}")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered financial forecasting agent for generating business outlook forecasts",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"LLM Provider: {settings.LLM_PROVIDER}")
    logger.info(f"Database: {settings.MYSQL_DATABASE}")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Shutting down application")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint - API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "endpoints": {
            "forecast": "/forecast/tcs",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connection
        db = next(get_db())
        db.execute("SELECT 1")

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "connected",
            "llm_provider": settings.LLM_PROVIDER
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.post(
    "/forecast/tcs",
    response_model=ForecastResponse,
    tags=["Forecast"],
    summary="Generate Financial Forecast",
    description="Generate a comprehensive financial forecast for TCS based on historical data and AI analysis"
)
async def generate_forecast(
    request: ForecastRequest,
    db: Session = Depends(get_db)
):
    """
    Generate financial forecast for TCS.

    This endpoint:
    1. Downloads and processes financial documents
    2. Extracts financial metrics using AI
    3. Analyzes earnings call transcripts using RAG
    4. Fetches live market data
    5. Orchestrates all tools via LangChain agent
    6. Returns structured JSON forecast
    7. Logs everything to MySQL database

    Args:
        request: ForecastRequest with company_url and quarters
        db: Database session (injected)

    Returns:
        ForecastResponse with complete forecast and supporting data
    """
    start_time = time.time()

    # Extract company symbol from URL
    company_symbol = extract_company_symbol(request.company_url)

    # Create initial request log
    try:
        request_log = DatabaseService.create_request_log(
            db=db,
            company_symbol=company_symbol,
            company_url=request.company_url,
            quarters=request.quarters,
            request_payload=request.dict()
        )
        request_id = request_log.request_id

    except Exception as e:
        logger.error(f"Error creating request log: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )

    try:
        # Initialize agent pipeline
        logger.info(f"Processing forecast request {request_id}")
        pipeline = AgentPipeline()

        # Generate forecast
        forecast_result = pipeline.generate_forecast(
            company_url=request.company_url,
            quarters=request.quarters,
            include_market_data=request.include_market_data
        )

        # Check for errors
        if forecast_result.get("status") == "error":
            raise Exception(forecast_result.get("error", "Unknown error"))

        # Extract forecast data
        forecast_data = forecast_result.get("forecast", {})

        # Build response
        forecast_output = ForecastOutput(**forecast_data)

        response = ForecastResponse(
            request_id=request_id,
            company_symbol=company_symbol,
            forecast=forecast_output,
            quarters_analyzed=request.quarters,
            tools_used=["FinancialDataExtractor", "QualitativeAnalysis", "MarketData"],
            execution_time_seconds=round(time.time() - start_time, 2)
        )

        # Update request log with success
        DatabaseService.update_request_log(
            db=db,
            request_id=request_id,
            status="success",
            response_payload=response.dict(),
            execution_time_seconds=int(response.execution_time_seconds),
            tools_used=response.tools_used
        )

        logger.info(f"Forecast generated successfully: {request_id}")
        return response

    except Exception as e:
        logger.error(f"Error generating forecast: {e}")

        # Update request log with error
        DatabaseService.update_request_log(
            db=db,
            request_id=request_id,
            status="failed",
            error_message=str(e),
            execution_time_seconds=int(time.time() - start_time)
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating forecast: {str(e)}"
        )


@app.get(
    "/requests/{request_id}",
    tags=["Requests"],
    summary="Get Request Details",
    description="Retrieve details of a specific forecast request by ID"
)
async def get_request_details(
    request_id: str,
    db: Session = Depends(get_db)
):
    """Get details of a specific request."""
    request_log = DatabaseService.get_request_log(db, request_id)

    if not request_log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Request {request_id} not found"
        )

    return {
        "request_id": request_log.request_id,
        "company_symbol": request_log.company_symbol,
        "status": request_log.status,
        "request_timestamp": request_log.request_timestamp,
        "response_timestamp": request_log.response_timestamp,
        "execution_time_seconds": request_log.execution_time_seconds,
        "tools_used": request_log.tools_used,
        "error_message": request_log.error_message
    }


@app.get(
    "/requests",
    tags=["Requests"],
    summary="Get Recent Requests",
    description="Retrieve recent forecast requests"
)
async def get_recent_requests(
    company_symbol: Optional[str] = None,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get recent requests."""
    requests = DatabaseService.get_recent_requests(
        db=db,
        company_symbol=company_symbol,
        limit=limit
    )

    return {
        "count": len(requests),
        "requests": [
            {
                "request_id": req.request_id,
                "company_symbol": req.company_symbol,
                "status": req.status,
                "timestamp": req.created_at,
                "execution_time_seconds": req.execution_time_seconds
            }
            for req in requests
        ]
    }


@app.post(
    "/tools/financial-extract",
    tags=["Tools"],
    summary="Test Financial Data Extractor",
    description="Directly test the FinancialDataExtractorTool"
)
async def test_financial_extractor(company_url: str, quarters: int = 2):
    """Test financial data extraction tool."""
    try:
        from app.tools import FinancialDataExtractorTool

        tool = FinancialDataExtractorTool()
        result = tool.run(f"{company_url},{quarters}")

        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/tools/qualitative-analysis",
    tags=["Tools"],
    summary="Test Qualitative Analysis Tool",
    description="Directly test the QualitativeAnalysisTool"
)
async def test_qualitative_analysis(query: str):
    """Test qualitative analysis tool."""
    try:
        from app.tools import QualitativeAnalysisTool

        # Initialize LLM
        pipeline = AgentPipeline()

        tool = QualitativeAnalysisTool(llm=pipeline.llm)
        result = tool.run(query)

        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/tools/market-data",
    tags=["Tools"],
    summary="Test Market Data Tool",
    description="Directly test the MarketDataTool"
)
async def test_market_data(symbol: str):
    """Test market data tool."""
    try:
        from app.tools import MarketDataTool

        tool = MarketDataTool()
        result = tool.run(symbol)

        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
