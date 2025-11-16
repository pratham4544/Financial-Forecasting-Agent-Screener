import os
from app.config import settings
from langchain import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings

def get_llm():
    # set env var for OpenAI compatibility
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
    return OpenAI(model_name=settings.OPENAI_MODEL, temperature=0)

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDINGS_MODEL)
