# raw_code_clean.py
import os
import re
import logging
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import datetime
from typing import List, Optional, Tuple

# LangChain imports (modern API)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

# Fallbacks (in case OpenAI not configured)
from langchain_community.embeddings import FakeEmbeddings

# NSE price (keep your existing package usage)
from nse_live_stocks import Nse

# Configuration
import os
import shutil

DOWNLOAD_DIR = "pdf_downloads"

def reset_download_folder():
    if os.path.exists(DOWNLOAD_DIR):
        shutil.rmtree(DOWNLOAD_DIR)  # removes folder AND all files
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
reset_download_folder()

GROQ_API_KEY=os.getenv("GROQ_API_KEY")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prompt (same as your updated base_prompt)
BASE_PROMPT = """
You are a Financial Forecasting Analyst Agent designed to analyze quarterly financial reports and earnings call transcripts for Tata Consultancy Services (TCS) or any provided company.
Your role is to extract factual financial metrics, identify qualitative themes, and produce a coherent forward-looking business forecast.

You will receive:
- {context}: Extracted text chunks from financial reports, presentations, and earnings call transcripts.
- {question}: A specific analytical request related to forecasting, trend analysis, risk assessment, or qualitative synthesis.

----------------------------------------------
## CORE BEHAVIOR & CONSTRAINTS
----------------------------------------------

### 1. Grounded Analysis Only
- Use only the information provided in {context}.
- Never hallucinate numbers, facts, or statements.
- If a detail is not present in the context, explicitly state that the information is missing.

### 2. Financial Reasoning Requirements
In your analysis, prioritize:
- Revenue trends and drivers.
- Margin behavior (EBIT, EBITDA, operating margins).
- Profitability trends (PAT, EPS, YoY and QoQ changes).
- Deal wins, pipeline visibility, demand commentary.
- Geographic or segment performance (BFSI, Retail, North America, Europe, etc.).
- Management’s sentiment and tone.
- Forward-looking statements from earnings calls.
- Risks, opportunities, macro headwinds, deal visibility, attrition, hiring trends.

### 3. JSON Output (Mandatory)
Your entire response must be a **valid machine-readable JSON object**, following the structure defined below.
Never include informal text, notes, explanations, or commentary outside the JSON.

### 4. No Investment Advice
- Do not provide buy/sell/hold recommendations.
- Do not discuss stock price predictions.
- Only analyze operational & business performance.



----------------------------------------------
## RESPONSE EXPECTATIONS
----------------------------------------------

- Ensure all statements are supported directly by the provided context.
- If context is repetitive, synthesize insights concisely.
- If context contains conflicting information, mention this in 'limitations'.
- Maintain professional, concise, executive-level clarity.
- Always return valid JSON — no trailing commas, no commentary outside the JSON.

----------------------------------------------
## END OF PROMPT
----------------------------------------------
"""

# Replace with your full prompt string from earlier (base_prompt in raw_code)

# Globals (shared vector store)
vector_store: Optional[FAISS] = None

# Text splitter & default chunk sizes
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Embeddings / LLM factory functions
def get_embeddings():
    """
    Return embeddings object. If OPENAI_API_KEY present, use OpenAIEmbeddings.
    Otherwise fall back to FakeEmbeddings for local testing.
    """
    # if os.getenv("OPENAI_API_KEY"):
    #     return OpenAIEmbeddings()
    logger.warning("OPENAI_API_KEY not found: using FakeEmbeddings (development only).")
    return FakeEmbeddings(size=1352)

def get_llm():
    """
    Loads Groq LLM if GROQ_API_KEY is set. Otherwise raises an error.
    """
    groq_key = os.getenv("GROQ_API_KEY")

    if groq_key:
        return ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            api_key=groq_key
        )
    
    raise RuntimeError("GROQ_API_KEY not set. Please export GROQ_API_KEY.")

def load_existing_vector_store():
    global vector_store
    try:
        index_dir = "faiss_index"
        vector_store = FAISS.load_local(index_dir, get_embeddings(), allow_dangerous_deserialization=True)
    except:
        vector_store = None


# -------------------------
# PDF Download + classification
# -------------------------
def scrape_screener_pdfs(company_url: str) -> List[Tuple[str, str]]:
    """Scrape screener.in page and return list of (pdf_url, filename)."""
    logger.info(f"Scraping: {company_url}")
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(company_url, headers=headers).text
    soup = BeautifulSoup(html, "html.parser")
    links = soup.select(".documents a")
    pdf_links = []
    for a in links:
        href = a.get("href", "")
        if href.endswith(".pdf"):
            text = a.text.strip().replace("\n", "_").replace(" ", "_")
            if not text:
                text = "Document"
            pdf_links.append((href, text + ".pdf"))
    logger.info("Found %d PDF links.", len(pdf_links))
    return pdf_links

def clean_filename(name: str) -> str:
    name = re.sub(r'[\/:*?"<>|;,]', '_', name)
    name = re.sub(r"_+", "_", name)
    name = name.strip(" .")
    if len(name) > 180:
        name = name[:180]
    return name

def download_direct_pdf(url: str, download_dir: str, filename: str) -> str:
    logger.info("[DIRECT] Downloading: %s", url)
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    savepath = os.path.join(download_dir, filename)
    with open(savepath, "wb") as f:
        f.write(r.content)
    logger.info("Saved direct PDF: %s", savepath)
    return savepath

def download_pdf_generic(url: str, filename: str) -> Optional[str]:
    filename = clean_filename(filename)
    try:
        # Add special cases here if needed (BSE iframe etc.)
        saved = download_direct_pdf(url, DOWNLOAD_DIR, filename)
        return saved
    except Exception as e:
        logger.error("Error downloading %s -> %s", url, e)
        return None

def classify_transcript_or_ppt(pdf_path: str) -> Optional[str]:
    try:
        reader = PdfReader(pdf_path)
        text = reader.pages[0].extract_text()[:800].lower()
    except Exception:
        return None
    if "transcript" in text or "earnings call" in text:
        return "transcript"
    if "presentation" in text or "investor presentation" in text:
        return "presentation"
    return None

def maybe_rename_transcript_or_ppt(saved_path: str) -> str:
    doc_type = classify_transcript_or_ppt(saved_path)
    if not doc_type:
        return saved_path
    folder = os.path.dirname(saved_path)
    base = os.path.basename(saved_path)
    name, ext = os.path.splitext(base)
    counter = 1
    new_name = f"{name}_{counter}{ext}"
    new_path = os.path.join(folder, new_name)
    while os.path.exists(new_path):
        counter += 1
        new_name = f"{name}_{counter}{ext}"
        new_path = os.path.join(folder, new_name)
    try:
        os.rename(saved_path, new_path)
        logger.info("Renamed to %s", new_name)
        return new_path
    except Exception as e:
        logger.warning("Rename failed: %s", e)
        return saved_path

def download_pdfs(company_url: str) -> str:
    pdfs = scrape_screener_pdfs(company_url)
    for url, filename in pdfs:
        logger.info("Downloading: %s", filename)
        path = download_pdf_generic(url, filename)
        
        if path:
            maybe_rename_transcript_or_ppt(path)
            
        delete_old_pdfs()
        
        
    return "Finished downloading PDFs."

# -------------------------
# Cleanup old PDFs (optional)
# -------------------------
def delete_old_pdfs(folder: str = DOWNLOAD_DIR, max_age_days: int = 365) -> int:
    now = datetime.datetime.now()
    deleted = 0
    for filename in os.listdir(folder):
        if not filename.lower().endswith(".pdf"):
            continue
        file_path = os.path.join(folder, filename)
        try:
            reader = PdfReader(file_path)
            metadata = reader.metadata or {}
            date_str = metadata.get("/CreationDate") or metadata.get("/ModDate")
            if not date_str:
                continue
            # try a simplified parse (best-effort)
            try:
                dt = datetime.datetime.strptime(date_str[2:16], "%Y%m%d%H%M%S")
            except Exception:
                continue
            if (now - dt).days > max_age_days:
                os.remove(file_path)
                deleted += 1
        except Exception as e:
            logger.warning("Error deleting/reading %s: %s", filename, e)
    return deleted

# -------------------------
# Vector store creation
# -------------------------
def create_vector_store(url: Optional[str] = None) -> FAISS:
    """
    Ingest all PDFs in pdf_downloads/, then optionally ingest URL content.
    Builds a FAISS index and assigns it to global vector_store.
    """
    global vector_store
    vector_store = None  # reset
    embeddings = get_embeddings()

    # ingest PDFs
    files_processed = 0
    for filename in os.listdir(DOWNLOAD_DIR):
        if not filename.lower().endswith(".pdf"):
            continue
        file_path = os.path.join(DOWNLOAD_DIR, filename)
        logger.info("Processing PDF: %s", filename)
        try:
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            except Exception:
                logger.info("PyPDFLoader failed, trying PDFMinerLoader")
                loader = PDFMinerLoader(file_path)
                docs = loader.load()
        except Exception as e:
            logger.error("Skipping %s: %s", filename, e)
            continue

        splits = text_splitter.split_documents(docs)
        if not splits:
            logger.warning("No text extracted from %s", filename)
            continue

        if vector_store is None:
            vector_store = FAISS.from_documents(splits, embeddings)
        else:
            vector_store.add_documents(splits)
        files_processed += 1
        logger.info("Added %d chunks from %s", len(splits), filename)

    # ingest web page if given
    if url:
        try:
            loader = WebBaseLoader(url)
            url_docs = loader.load()
            url_splits = text_splitter.split_documents(url_docs)
            if vector_store is None and url_splits:
                vector_store = FAISS.from_documents(url_splits, embeddings)
            elif url_splits:
                vector_store.add_documents(url_splits)
            logger.info("Added %d chunks from URL", len(url_splits))
        except Exception as e:
            logger.error("Error ingesting URL %s: %s", url, e)

    if vector_store:
        # persist the index locally for reuse
        index_dir = "faiss_index"
        vector_store.save_local(index_dir)
        logger.info("FAISS index saved to %s", index_dir)
    else:
        logger.error("No documents indexed; vector_store is None after ingestion.")

    return vector_store

# -------------------------
# RAG / question answering
# -------------------------
PROMPT_TEMPLATE = BASE_PROMPT
prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

def user_query_answer(query: str, k: int = 5):
    """
    Run retrieval + LLM. Returns the LLM response and the retrieved chunks.
    """
    global vector_store
    if vector_store is None:
        raise RuntimeError("vector_store is not initialized. Call create_vector_store() first.")

    # retrieve
    docs = vector_store.similarity_search(query, k=k)
    # convert docs into a single context string (avoid sending Document objects)
    # context_text = "\n\n".join([d.page_content for d in docs])
    
    # run LLM
    llm = get_llm()
    # Using a simple prompt invocation pattern - you can use Chains if preferred
    chain = prompt | llm | JsonOutputParser()
    response = chain.invoke({"context": docs, "question": query})
    # If response is a BaseMessage, extract text
    try:
        answer_text = response.content if hasattr(response, "content") else str(response)
    except Exception:
        answer_text = str(response)
    return answer_text, docs

# -------------------------
# Stock price helper
# -------------------------
def current_market_price(url: str) -> Optional[float]:
    """
    Extract stock symbol from screener url and query NSE.
    Example URL: https://www.screener.in/company/TCS/consolidated/
    """
    m = re.search(r'/company/([^/]+)/', url)
    if not m:
        raise ValueError("Could not parse stock symbol from URL.")
    symbol = m.group(1)
    stock = Nse()
    result = stock.get_current_price(symbol)
    # your nse lib may have different field names; adjust accordingly
    return result.get('current_value') if isinstance(result, dict) else None
