# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging

from research.raw_code import (
    download_pdfs,
    create_vector_store,
    user_query_answer,
    current_market_price,
    
)
import research.raw_code as pipeline


logger = logging.getLogger(__name__)
app = FastAPI(title="TCS Financial Forecasting Agent")

class LoadRequest(BaseModel):
    url: str

class AskRequest(BaseModel):
    question: str

@app.post("/load")
def load_endpoint(req: LoadRequest):
    msg = pipeline.download_pdfs(req.url)
    pipeline.create_vector_store(req.url)

    return {
        "status": "success",
        "message": msg,
        "vector_store_ready": pipeline.vector_store is not None
    }



@app.post("/ask")
def ask_endpoint(req: AskRequest):
    if pipeline.vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="Vector store not initialized. Please load documents first."
        )

    answer, docs = pipeline.user_query_answer(req.question)
    return {"answer": answer, "chunks_used": len(docs)}


@app.get("/price")
def price_endpoint(url: str):
    price = pipeline.current_market_price(url)
    return {"url": url, "price": price}

@app.on_event("startup")
def load_index():
    pipeline.load_existing_vector_store()



@app.get("/")
def root():
    return {"message": "TCS Financial Forecasting Agent running."}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
