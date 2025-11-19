import os
import re
import requests
from langchain_groq import ChatGroq
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import datetime
from sentence_transformers import SentenceTransformer
from langchain_unstructured import UnstructuredLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from dotenv import load_dotenv
load_dotenv()
from nse_live_stocks import Nse
import shutil


## Building Basic Variables

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

GROQ_API_KEY=os.environ['GROQ_API_KEY']

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3,api_key=GROQ_API_KEY)

DOWNLOAD_DIR='pdf_downloads'

vector_store = None



## System Prompt Variable

base_prompt = '''
You are an intelligent financial analysis agent specialized in reviewing company quarterly reports and earnings call transcripts.
Your primary function is to extract accurate financial metrics, analyze qualitative management commentary, and generate clear, structured insights.

During execution, you will receive:

{context}: Extracted text chunks from financial reports and transcripts

{question}: A specific analytical task (e.g., trends, risks, sentiment, outlook)

Guidelines for responding to {question} using {context}:

Accurate & Grounded: Use only the information found in the provided context—no guessing or fabricating data.

Financially Insightful: Provide concise explanations about revenue trends, margin movements, demand commentary, risks, and opportunities.

Forecast-Oriented: Highlight management sentiment and forward-looking statements relevant to future performance.

Structured JSON: Always respond in a predictable JSON format suitable for downstream processing.

No Investment Advice: Do not provide stock buy/sell/hold recommendations or personal financial advice.

Example JSON Output:
{{
  "reply": "Revenue grew 5% YoY driven by cloud and BFSI demand, while margins remained stable. Management highlighted healthy deal wins and improving client budgets.",
  "guidance_caution": "This summary is based solely on the provided financial context and does not constitute investment advice.",
  "follow_up_prompt": "Would you like insights on risks, opportunities, or the outlook for the next quarter?",
}}

'''

# Prompt 
prompt = PromptTemplate(template=base_prompt, input_variables=["context", "question"])


## Scrape Documents

def reset_download_folder():
    if os.path.exists(DOWNLOAD_DIR):
        shutil.rmtree(DOWNLOAD_DIR)  # removes folder AND all files
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def scrape_screener_pdfs(company_url):
    print(f"Scraping: {company_url}")
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

    print(f"Found {len(pdf_links)} PDF links.\n")
    return pdf_links

def classify_transcript_or_ppt(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = reader.pages[0].extract_text()[:800].lower()
    except:
        return None

    if "transcript" in text or "earnings call" in text:
        return "transcript"
    if "presentation" in text or "investor presentation" in text:
        return "presentation"

    return None

def maybe_rename_transcript_or_ppt(saved_path):
    doc_type = classify_transcript_or_ppt(saved_path)

    if not doc_type:
        print("   → Not transcript/presentation. Keeping original filename.\n")
        return saved_path

    # Original filename
    folder = os.path.dirname(saved_path)
    base = os.path.basename(saved_path)
    name, ext = os.path.splitext(base)

    # Make new filename with incremental number
    counter = 1
    new_name = f"{name}_{counter}{ext}"
    new_path = os.path.join(folder, new_name)

    while os.path.exists(new_path):
        counter += 1
        new_name = f"{name}_{counter}{ext}"
        new_path = os.path.join(folder, new_name)

    try:
        os.rename(saved_path, new_path)
        print(f"   ✔ Transcript/PPT detected → renamed to {new_name}\n")
        return new_path
    except:
        print("   ⚠ Rename failed. Keeping original.\n")
        return saved_path

def download_bse_annpdf(url, download_dir, filename):
    print(f"URL: {url}")
    print(f"[BSE-ANNPDF] Requesting: {url}")

    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    savepath = os.path.join(download_dir, filename)

    with open(savepath, "wb") as f:
        f.write(r.content)

    print(f"✔ Saved BSE AnnPdf: {savepath}")
    return savepath

def download_bse_iframe_pdf(url, download_dir, filename):
    print(f"URL: {url}")
    print("   [BSE-IFRAME] Requesting main page…")

    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
    soup = BeautifulSoup(html, "html.parser")
    iframe = soup.find("iframe")

    if not iframe:
        print("   ❌ No iframe found. Cannot download.")
        return None

    real_pdf = iframe.get("src")
    if not real_pdf.startswith("http"):
        real_pdf = "https://www.bseindia.com" + real_pdf

    print(f"   → PDF Source: {real_pdf}")

    r = requests.get(real_pdf, headers={"User-Agent": "Mozilla/5.0"})
    savepath = os.path.join(download_dir, filename)

    with open(savepath, "wb") as f:
        f.write(r.content)

    print(f"   ✔ Saved BSE iframe PDF: {savepath}")
    return savepath


def download_direct_pdf(url, download_dir, filename):
    print(f"[DIRECT] Downloading: {url}")
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    savepath = os.path.join(download_dir, filename)

    with open(savepath, "wb") as f:
        f.write(r.content)

    print(f"✔ Saved direct PDF: {savepath}")
    return savepath


def clean_filename(name):
    name = re.sub(r'[\/:*?"<>|;,]', '_', name)
    name = re.sub(r"_+", "_", name)
    name = name.strip(" .")
    if len(name) > 180:
        name = name[:180]
    return name

def download_pdf(url, filename):

    filename = clean_filename(filename)

    if "AnnPdfOpen.aspx" in url:
        saved = download_bse_annpdf(url, DOWNLOAD_DIR, filename)
        return maybe_rename_transcript_or_ppt(saved)

    if "xml-data/corpfiling" in url:
        saved = download_bse_iframe_pdf(url, DOWNLOAD_DIR, filename)
        return maybe_rename_transcript_or_ppt(saved)

    saved = download_direct_pdf(url, DOWNLOAD_DIR, filename)
    return maybe_rename_transcript_or_ppt(saved)

def run(company_url):
    pdfs = scrape_screener_pdfs(company_url)

    for url, filename in pdfs:
        print("-----------------------------------------")
        print(f"Downloading: {filename}")
        download_pdf(url, filename)

def delete_old_pdfs():

    FOLDER = "pdf_downloads/"
    ONE_YEAR_DAYS = 365

    def parse_pdf_date(date_str):
        # Format type 1: D:20181012055359+05'30'
        if date_str.startswith("D:"):
            try:
                dt = datetime.datetime.strptime(date_str[2:16], "%Y%m%d%H%M%S")
                return dt
            except:
                pass

        # Format type 2: Fri 12 Oct 2018 05:53:59 PM +05:30
        try:
            dt = datetime.datetime.strptime(date_str, "%a %d %b %Y %I:%M:%S %p %z")
            return dt.replace(tzinfo=None)
        except:
            return None


    now = datetime.datetime.now()
    deleted_files = 0

    for filename in os.listdir(FOLDER):
        if not filename.lower().endswith(".pdf"):
            continue
        
        file_path = os.path.join(FOLDER, filename)

        try:
            reader = PdfReader(file_path)
            metadata = reader.metadata
            
            if "/CreationDate" in metadata:
                pdf_date = parse_pdf_date(metadata["/CreationDate"])
            elif "/ModDate" in metadata:
                pdf_date = parse_pdf_date(metadata["/ModDate"])
            else:
                print(f"⚠ No metadata date found for {filename}, skipping.")
                continue

            if not pdf_date:
                print(f"⚠ Could not parse date for {filename}, skipping.")
                continue

            age_days = (now - pdf_date).days

            if age_days > ONE_YEAR_DAYS:
                print(f"🗑 Deleting: {filename} (Age: {age_days} days)")
                os.remove(file_path)
                deleted_files += 1
            else:
                print(f"✔ Keeping: {filename} (Age: {age_days} days)")

        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")

    print(f"\n✅ Done. Deleted {deleted_files} old PDF(s).")

    return f'Deleted {deleted_files} old PDF(s).'

def create_url_vector_store(url,vector_store):

    loader = UnstructuredURLLoader(urls=[url])
    data = loader.load()
    chunks = text_splitter.split_documents(data)
    vector_store=FAISS.from_documents(chunks,embeddings_model)
    
    return vector_store


def user_query_answer(query,vector_store):
    
    extracted_chunks = vector_store.similarity_search_with_score(query,k=5)
    chain = prompt | llm | JsonOutputParser()
    response = chain.invoke({"context": extracted_chunks, "question": query})
    return response, extracted_chunks

def create_chunks(DOWNLOAD_DIR):

    file_path = []

    # collect full paths to PDFs in DOWNLOAD_DIR
    for fname in os.listdir(DOWNLOAD_DIR):
        if fname.lower().endswith(".pdf"):
            file_path.append(os.path.join(DOWNLOAD_DIR, fname))
        
    print(f'Length of Files in Folder is {len(file_path)}')

    try:
        if not file_path:
            raise ValueError(f"No PDF files found in {DOWNLOAD_DIR}")

        loader = UnstructuredLoader(file_path)
        docs = loader.load()

        chunks = text_splitter.split_documents(docs)
        print(f'Total length of chunks stored into db is {len(chunks)}')

    except Exception as e:
        print(f'Error processing files: {e}')
        
    return chunks


def create_pdf_vector_stores(chunks):
    
    vector_store = None
    
    if vector_store is None:
        vector_store = FAISS.from_documents(chunks, embeddings_model)
        print("Created new FAISS vector store from chunks.")
    else:
        vector_store.add_documents(chunks)
        print("Added chunks to existing vector store.")

    return vector_store

def current_market_price(url):
    
    m = re.search(r'/company/([^/]+)/', url)
    
    if not m:
        raise ValueError("Could not parse stock symbol from URL.")
    
    symbol = m.group(1)
    stock = Nse()
    result = stock.get_current_price(symbol)
    
    return result.get('current_value') if isinstance(result, dict) else None