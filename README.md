# TCS Financial Forecast Agent

## Overview
This FastAPI service implements an AI agent that generates a reasoned business outlook (forecast) for Tata Consultancy Services (TCS). The agent:
- Scrapes financial tables from Screener (or downloads linked PDFs).
- Downloads recent earnings call transcripts (PDFs), extracts text, builds a FAISS RAG index and runs semantic queries.
- Fetches live market price for TCS using `nse_live_stocks`.
- Orchestrates the three specialized tools as LangChain tools inside an agent that returns a structured JSON forecast.
- Logs incoming requests and final outputs to MySQL.

## Architecture
- FastAPI backend (`app/main.py`).
- Tools:
  - `FinancialDataExtractorTool` — scrapes numeric metrics from Screener.
  - `QualitativeAnalysisTool` — RAG-based analysis over transcripts (FAISS + embeddings).
  - `MarketDataTool` — live NSE price via `nse_live_stocks`.
- LangChain agent (`app/services/pipeline.py`) orchestrates tools and synthesizes the forecast.
- MySQL logging via SQLAlchemy.

## AI Stack & Tools
- LLM: OpenAI (configurable in `.env`).
- Embeddings: sentence-transformers (HuggingFace).
- Vector DB: FAISS (local).
- RAG via LangChain `RetrievalQA`.
- PDF extraction: `pdfminer.six`.
- Live market price: `nse_live_stocks`.

## Setup (local)
1. Clone the repo.
2. Create `.env` file at project root:
3.OPENAI_API_KEY=your_openai_key_here
4.MYSQL_DSN=mysql+pymysql://user:password@localhost:3306/tcs_forecast

3. Create Python venv and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt```


4. Ensure MySQL is running and database exists:
'''
CREATE DATABASE tcs_forecast CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
'''

Start the API:

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Usage

POST to:

POST http://localhost:8000/forecast/tcs
Content-Type: application/json

{
  "company_url":"https://www.screener.in/company/TCS/consolidated/#documents",
  "quarters": 2
}

Output

The API returns a structured JSON matching ForecastResponse model, and logs request & response to request_logs MySQL table.

Notes, Guardrails & Recommendations

Screener scraping is heuristic — production-grade scraping should prefer official IR PDFs when possible.

PDFs that are scanned require OCR (Tesseract) — currently pdfminer.six is used for text PDFs.

LLM calls: set temperature=0 to reduce hallucination. Keep RAG retrieval size (k) small (3-5).

Agent grounding: the pipeline instructs agent to only use tool outputs for factual numbers; still validate outputs manually.

Costs: LLM usage (OpenAI) can be costly — consider smaller models or batching queries.

Extending

Replace FAISS with a hosted vector DB for scale (Pinecone, Chroma Cloud).

Add Celery + Redis for background job execution if endpoints should be async.

Add unit tests and test fixtures (sample transcripts).

Troubleshooting

If pdfminer fails on a PDF, try OCR fallback (pytesseract).

If scraping fails due to layout changes on screener.in, update app/utils/html_scraper.py selectors.


---

# How this agent uses LangChain tools (notes)
- Each specialized tool is wrapped as a `langchain.tools.Tool` with a short description.
- `pipeline.run_pipeline_sync` builds an agent via `initialize_agent(...)` and asks it to orchestrate the tools and return a JSON.
- If the agent returns non-JSON text or fails, the pipeline falls back to deterministic direct tool calls and constructs a safe JSON.

---

# How to run (quick checklist)

1. Insert your OpenAI key and MySQL DSN in `.env`.
2. Install requirements.
3. Ensure MySQL database `tcs_forecast` exists.
4. `uvicorn app.main:app --reload`
5. `curl` or Postman to POST to `/forecast/tcs`.

---

# Final notes & offers
- If you want I can:
  - package this repo into a zip and give you a download link,
  - create a `docker-compose.yml` that runs MySQL + app,
  - extend the qualitative tool with structured evidence citations (include exact chunks),
  - add an OCR fallback using `pytesseract`.

Which of those should I do next?