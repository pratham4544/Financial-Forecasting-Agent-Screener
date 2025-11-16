from fastapi import FastAPI
from app.routers import forecast
from app.db import Base, engine

app = FastAPI(title="TCS Financial Forecast Agent")

# create DB tables
Base.metadata.create_all(bind=engine)

app.include_router(forecast.router, prefix="/forecast")
