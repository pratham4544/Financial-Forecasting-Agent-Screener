from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


# Import your entire raw pipeline
from research.raw_code import (
    download_pdfs,
    create_vector_store,
    user_query_answer,
    current_market_price,
    vector_store
)
# ---------------------------
app = FastAPI()

# ---------------------------
# Request models
# ---------------------------

class LoadRequest(BaseModel):
    url: str

class QuestionRequest(BaseModel):
    question: str


# ---------------------------
# API ENDPOINTS
# ---------------------------

@app.post("/load")
def load_data(req: LoadRequest):
    """
    1. Downloads PDFs
    2. Creates vector store
    """
    global vector_store

    msg = download_pdfs(req.url)
    vector_store = create_vector_store(req.url)

    return {
        "status": "success",
        "message": msg,
        "vector_store_ready": vector_store is not None
    }


@app.post("/ask")
def ask_question(req: QuestionRequest):
    """
    Runs RAG + LLM chain and returns JSON output.
    """
    if vector_store is None:
        return {"error": "Vector store not ready. Call /load first."}

    response, chunks = user_query_answer(req.question)

    return {
        "answer": response,
        "chunks_used": len(chunks)
    }


@app.get("/price")
def get_price(url: str):
    """
    Extract stock symbol from URL and return live market price.
    """
    try:
        price = current_market_price(url)
        return {"stock_url": url, "price": price}
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def home():
    return {"message": "TCS AI Financial Analysis Agent is running!"}


# ---------------------------
# RUN API
# ---------------------------

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
