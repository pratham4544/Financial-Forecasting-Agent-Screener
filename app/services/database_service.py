"""
Database service for logging requests and responses.
Handles all database operations for the application.
"""
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from app.models.database import RequestLog

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service for database operations."""

    @staticmethod
    def generate_request_id() -> str:
        """Generate unique request ID."""
        return f"req_{uuid.uuid4().hex[:16]}"

    @staticmethod
    def create_request_log(
        db: Session,
        company_symbol: str,
        company_url: str,
        quarters: int,
        request_payload: Dict[str, Any]
    ) -> RequestLog:
        """
        Create initial request log entry.

        Args:
            db: Database session
            company_symbol: Company stock symbol
            company_url: URL to company page
            quarters: Number of quarters to analyze
            request_payload: Request data

        Returns:
            RequestLog instance
        """
        try:
            request_id = DatabaseService.generate_request_id()

            log = RequestLog(
                request_id=request_id,
                company_symbol=company_symbol,
                company_url=company_url,
                quarters=quarters,
                request_payload=request_payload,
                request_timestamp=datetime.utcnow(),
                status="pending"
            )

            db.add(log)
            db.commit()
            db.refresh(log)

            logger.info(f"Created request log: {request_id}")
            return log

        except Exception as e:
            logger.error(f"Error creating request log: {e}")
            db.rollback()
            raise

    @staticmethod
    def update_request_log(
        db: Session,
        request_id: str,
        status: str,
        response_payload: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        execution_time_seconds: Optional[int] = None,
        tools_used: Optional[list] = None
    ) -> Optional[RequestLog]:
        """
        Update request log with results.

        Args:
            db: Database session
            request_id: Request identifier
            status: Status (success/failed)
            response_payload: Response data
            error_message: Error message if failed
            execution_time_seconds: Execution time
            tools_used: List of tools invoked

        Returns:
            Updated RequestLog instance or None
        """
        try:
            log = db.query(RequestLog).filter(
                RequestLog.request_id == request_id
            ).first()

            if not log:
                logger.error(f"Request log not found: {request_id}")
                return None

            log.status = status
            log.response_timestamp = datetime.utcnow()

            if response_payload:
                log.response_payload = response_payload

            if error_message:
                log.error_message = error_message

            if execution_time_seconds is not None:
                log.execution_time_seconds = execution_time_seconds

            if tools_used:
                log.tools_used = tools_used

            db.commit()
            db.refresh(log)

            logger.info(f"Updated request log: {request_id} with status: {status}")
            return log

        except Exception as e:
            logger.error(f"Error updating request log: {e}")
            db.rollback()
            return None

    @staticmethod
    def get_request_log(db: Session, request_id: str) -> Optional[RequestLog]:
        """
        Retrieve request log by ID.

        Args:
            db: Database session
            request_id: Request identifier

        Returns:
            RequestLog instance or None
        """
        try:
            log = db.query(RequestLog).filter(
                RequestLog.request_id == request_id
            ).first()
            return log

        except Exception as e:
            logger.error(f"Error retrieving request log: {e}")
            return None

    @staticmethod
    def get_recent_requests(
        db: Session,
        company_symbol: Optional[str] = None,
        limit: int = 10
    ) -> list:
        """
        Get recent request logs.

        Args:
            db: Database session
            company_symbol: Filter by company (optional)
            limit: Number of records to return

        Returns:
            List of RequestLog instances
        """
        try:
            query = db.query(RequestLog)

            if company_symbol:
                query = query.filter(RequestLog.company_symbol == company_symbol)

            logs = query.order_by(
                RequestLog.created_at.desc()
            ).limit(limit).all()

            return logs

        except Exception as e:
            logger.error(f"Error retrieving recent requests: {e}")
            return []
