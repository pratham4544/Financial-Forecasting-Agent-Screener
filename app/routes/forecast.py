from fastapi import APIRouter, Depends, HTTPException
from app.models import ForecastRequest, ForecastResponse, RequestLog
from app.db import SessionLocal
from app.services.pipeline import run_pipeline_sync
import json
from sqlalchemy.orm import Session

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/tcs", response_model=ForecastResponse)
def generate_tcs_forecast(req: ForecastRequest, db: Session = Depends(get_db)):
    # log request
    log = RequestLog(company=req.company_url, request_payload=json.dumps(req.dict()))
    db.add(log)
    db.commit()
    db.refresh(log)

    try:
        result = run_pipeline_sync(req.company_url, quarters=req.quarters)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # log response
    log.response_payload = json.dumps(result)
    db.add(log)
    db.commit()

    return result
