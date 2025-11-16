from typing import List, Dict, Any
from app.utils.html_scraper import find_document_links
from app.utils.pdf_utils import download_pdfs, extract_pdf_text
from app.utils.text_chunker import chunk_text
from app.services.llm_client import get_embeddings, get_llm
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

def extract_earnings_transcripts(company_url: str, quarters: int = 2) -> List[str]:
    links = find_document_links(company_url)
    # filter names likely to be earnings / concall / results
    filtered = [l for l in links if any(s in l.lower() for s in ("concall", "results", "quarter", "q1", "q2", "q3", "q4", "qtr", "earnings", "transcript"))]
    filtered = filtered[: max(quarters * 2, 4)]
    pdfs = download_pdfs(filtered)
    texts = []
    for p in pdfs:
        txt = extract_pdf_text(p)
        if txt and len(txt.strip()) > 50:
            texts.append(txt)
    return texts

def build_rag_index(texts: List[str]):
    embeddings = get_embeddings()
    docs = []
    for t in texts:
        for chunk in chunk_text(t):
            docs.append(Document(page_content=chunk))
    if not docs:
        return None
    store = FAISS.from_documents(docs, embeddings)
    return store

def qualitative_analysis(company_url: str, quarters: int = 2) -> Dict[str, Any]:
    transcripts = extract_earnings_transcripts(company_url, quarters=quarters)
    if not transcripts:
        return {"summary": "No transcripts found", "themes": [], "sentiment": {}}

    store = build_rag_index(transcripts)
    if store is None:
        return {"summary": "No textual content to analyze"}

    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = get_llm()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    queries = {
        "management_outlook": "Summarize management's forward-looking statements and guidance in a short paragraph (use evidence).",
        "recurring_themes": "List recurring themes mentioned by management (hiring, demand, margin, pricing).",
        "risks": "List explicit risks management highlighted.",
        "opportunities": "List opportunities management mentioned.",
        "sentiment": "Provide a one-line sentiment (positive/neutral/negative) with reasoning."
    }

    results = {}
    for k, q in queries.items():
        try:
            answer = qa_chain.run(q)
        except Exception:
            answer = "Error in retrieval/LLM"
        results[k] = answer

    return {
        "summary": results.get("management_outlook"),
        "themes": results.get("recurring_themes"),
        "risks": results.get("risks"),
        "opportunities": results.get("opportunities"),
        "sentiment": results.get("sentiment"),
    }
