"""
SQLAlchemy database models and connection setup.
Handles MySQL database operations and request logging.
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=settings.DEBUG
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


class RequestLog(Base):
    """
    Model for logging all forecast requests and responses.
    Tracks the complete lifecycle of each forecast generation.
    """
    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    request_id = Column(String(100), unique=True, index=True, nullable=False)
    company_symbol = Column(String(50), index=True, nullable=False)
    company_url = Column(String(500), nullable=True)
    quarters = Column(Integer, default=2)

    # Request data
    request_payload = Column(JSON, nullable=True)
    request_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Response data
    response_payload = Column(JSON, nullable=True)
    response_timestamp = Column(DateTime, nullable=True)

    # Execution metadata
    status = Column(String(50), default="pending")  # pending, success, failed
    error_message = Column(Text, nullable=True)
    execution_time_seconds = Column(Integer, nullable=True)

    # Tool execution logs
    tools_used = Column(JSON, nullable=True)  # List of tools invoked

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<RequestLog(id={self.id}, company={self.company_symbol}, status={self.status})>"


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    Dependency function to get database session.
    Used in FastAPI endpoints for dependency injection.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
