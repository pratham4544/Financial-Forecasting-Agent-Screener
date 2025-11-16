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
from nse_live_stocks import Nse

DOWNLOAD_DIR = "pdf_downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def current_market_price(stock_symbol):
    stock = Nse()
    result = stock.get_current_price('TCS')
    return result['current_value']

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

url = 'https://www.screener.in/company/JIOFIN/consolidated/'

run(url)

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


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
embeddings = FakeEmbeddings(size=1352)


vector_store = None
count = 0

def create_vector_store():
    for filename in os.listdir("pdf_downloads/"):
        file_path = f"pdf_downloads/{filename}"
        print(f"\n📄 Processing: {filename}")

        # Robust PDF loading
        try:
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            except:
                print("⚠ PyPDF failed, trying PDFMiner...")
                loader = PDFMinerLoader(file_path)
                docs = loader.load()
        except Exception as e:
            print(f"❌ Skipping {filename}: cannot load PDF -> {e}")
            continue

        # Ensure text exists
        splits = text_splitter.split_documents(docs)
        if not splits:
            print(f"⚠ No extractable text in {filename}, skipping.")
            continue

        # Build or update FAISS index
        if vector_store is None:
            vector_store = FAISS.from_documents(splits, embedding=embeddings)
        else:
            vector_store.add_documents(splits)

        count += 1
        print(f"✔ Added {len(splits)} chunks from {filename}")

    # Save final index
    if vector_store:
        vector_store.save_local("faiss_index")
        print("\n🎉 FAISS index saved successfully.")
    else:
        print("\n❌ No valid PDFs processed; FAISS index not created.")

    return vector_store
 
 
def create_url_vector_store(url):

    url_loader = WebBaseLoader(url)

    url_docs = url_loader.load()

    url_split_documents = text_splitter.split_documents(url_docs)

    vector_store.add_documents(url_split_documents)

    vector_store.save_local("faiss_index")

    vector_store = vector_store.load_local("faiss_index", embeddings=embeddings,allow_dangerous_deserialization=True)
    
    return vector_store


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    api_key='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'
    # other params...
)


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
  "follow_up_prompt": "Would you like insights on risks, opportunities, or the outlook for the next quarter?"
}}

'''


prompt = PromptTemplate(template=base_prompt, input_variables=["context", "question"])

def user_query_answer(query):
    vectorstore_retreiver = vector_store.as_retriever(search_kwargs={"k": 3})
    extracted_chunks = vectorstore_retreiver.invoke(query)
    chain = prompt | llm | JsonOutputParser()
    response = chain.invoke({"context": extracted_chunks, "question": query})
    return response, extracted_chunks


