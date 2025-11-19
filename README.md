# 📊 Financial Document Analysis Agent – Screener.in Intelligence

An AI-powered financial analysis system built using **Streamlit**, **LangChain**, **Groq LLM**, and **RAG (Retrieval-Augmented Generation)**.
The agent automatically downloads quarterly financial reports & earnings transcripts from Screener.in, extracts insights, and generates **structured, analyst-grade qualitative forecasts** for any Indian publicly traded company.

Built for intelligent financial document analysis with semantic search and conversational AI.

---

## 🔍 Key Features

### ✅ **Automatic Document Retrieval**

* Scrapes **financial reports, earnings call transcripts, investor presentations** directly from Screener.in
* Intelligent PDF classification (transcripts vs presentations)
* Auto-removes outdated documents (>1 year old)
* Downloads into managed `pdf_downloads/` directory
* Extracts live market data from NSE India

### ✅ **RAG-Based Retrieval & Analysis**

* Splits documents using `RecursiveCharacterTextSplitter` (500 chars, 50 overlap)
* Builds FAISS vector store with HuggingFace embeddings (`all-mpnet-base-v2`)
* Semantic search with relevance scoring
* Combines PDF data + live URL scraping for comprehensive context
* Supports persistent vector database (save/load functionality)

### ✅ **Intelligent Forecasting Agent**

* Powered by Groq's **gpt-oss-120b** LLM
* Analyzes both **quantitative metrics** and **qualitative insights**
* Produces structured JSON outputs with:
  - Financial analysis
  - Management sentiment
  - Forward-looking guidance
  - Risks and opportunities
  - Investment considerations
* No hallucination – all responses grounded in retrieved documents

### ✅ **Streamlit Interface**

* 🎨 Modern, intuitive UI with tabs and progress tracking
* 📥 **Document Processing Pipeline** – automated multi-step workflow
* 💬 **Interactive Query System** – ask unlimited questions
* 📜 **Chat History** – track all analyses
* 📈 **Live Market Data** – real-time NSE stock prices
* 💾 **Vector DB Management** – save/load/reset capabilities

---

## 🧱 Architecture Overview

```
User Input (Screener URL)
         ↓
┌────────────────────────────────────┐
│   Document Scraper & Processor     │
│  • Web scraping (BeautifulSoup)    │
│  • PDF downloads & classification  │
│  • Age-based filtering             │
└────────────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│   Knowledge Pool (Vector Store)    │
│  • Unstructured PDF loading        │
│  • Text chunking (500/50)          │
│  • HuggingFace embeddings          │
│  • FAISS indexing                  │
│  • URL content integration         │
└────────────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│   RAG Query Pipeline               │
│  • Semantic search (top-5)         │
│  • Context retrieval               │
│  • LLM analysis (Groq)             │
│  • JSON output parsing             │
└────────────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│   Market Data Integration          │
│  • NSE live price fetching         │
│  • Symbol extraction from URL      │
└────────────────────────────────────┘
         ↓
    Structured Response
```

---

## 📂 Project Structure

```
.
├── app.py                          # Streamlit Frontend Application
├── test_1.py                       # CLI testing script
├── research/
│   └── raw.py                      # Core RAG + Scraper + LLM pipeline
├── pdf_downloads/                  # Auto-managed PDF storage
├── faiss_index/                    # Persistent vector store
├── .env                            # Environment variables (GROQ_API_KEY)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## ⚙️ Technologies Used

| Component              | Technology                                      |
| ---------------------- | ----------------------------------------------- |
| **Frontend**           | Streamlit                                       |
| **LLM Provider**       | Groq (openai/gpt-oss-120b)                      |
| **Embeddings**         | HuggingFace (all-mpnet-base-v2)                 |
| **Vector Store**       | FAISS (Facebook AI Similarity Search)           |
| **Document Loading**   | Unstructured, PyPDF2, UnstructuredURLLoader     |
| **Web Scraping**       | BeautifulSoup4, Requests                        |
| **Text Processing**    | LangChain (RecursiveCharacterTextSplitter)      |
| **Market Data**        | nse-live-stocks (NSE India API)                 |
| **Orchestration**      | LangChain Chains & Prompts                      |
| **Environment**        | python-dotenv                                   |

---

## 🔥 Installation Guide

### **1. Clone the Repository**

```bash
git clone https://github.com/pratham4544/Financial-Forecasting-Agent-Screener
cd Financial-Forecasting-Agent-Screener
```

### **2. Create Virtual Environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Set Up Environment Variables**

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your Groq API key from: https://console.groq.com/

---

## 🚀 Running the Application

### Launch Streamlit Interface

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Alternative: CLI Mode

For quick testing without the UI:

```bash
python test_1.py
```

---

## 📖 How to Use

### Step 1: Enter Company URL
Navigate to the **Document Processing** tab and enter a Screener.in URL:

```
https://www.screener.in/company/TCS/consolidated/
https://www.screener.in/company/INFY/consolidated/
https://www.screener.in/company/GROWW/consolidated/
```

### Step 2: Process Documents
Click **🚀 Process Documents** to:
1. Reset download folder
2. Download all PDFs from Screener.in
3. Remove documents older than 1 year
4. Create document chunks
5. Build FAISS vector database
6. Add URL content to knowledge pool

### Step 3: Ask Questions
Switch to the **Query Analysis** tab and ask questions like:

**Financial Metrics:**
```
What is the revenue growth trend over the last two quarters?
Extract total revenue, net profit, and EBIT margin for Q3 and Q4.
```

**Profitability Analysis:**
```
Explain the key drivers behind margin expansion or contraction.
What factors affected operating margins?
```

**Management Commentary:**
```
What was the management sentiment in the latest earnings call?
What forward-looking statements did management make?
```

**Risk Assessment:**
```
List all major risks and opportunities highlighted by management.
What external challenges does the company face?
```

**Comprehensive Forecast:**
```
Provide a qualitative business forecast for the upcoming quarter based on the last two quarters' reports and transcripts.
```

### Step 4: View Results
- Get structured JSON responses with financial insights
- See current market price from NSE
- View source document chunks with relevance scores
- Track all queries in Chat History tab

---

## 🧠 Example JSON Output

```json
{
  "reply": "Revenue grew 5.2% YoY to ₹62,613 crores driven by strong BFSI and healthcare demand. EBIT margin expanded to 26.1% from 25.8% due to operational efficiency and favorable currency movements. Net profit increased 8.1% to ₹12,434 crores. Management highlighted robust deal pipeline worth $10.2B and improving client budgets across North America and Europe.",
  "guidance_caution": "This summary is based solely on the provided financial context and does not constitute investment advice.",
  "follow_up_prompt": "Would you like insights on segment-wise performance, risks, or detailed margin analysis?"
}
```

---

## 🎯 Key Capabilities

### 📊 Quantitative Analysis
- Revenue trends and growth rates
- Profitability metrics (EBIT, PAT, margins)
- Cash flow analysis
- Segment-wise performance
- Year-over-year comparisons

### 💬 Qualitative Insights
- Management sentiment analysis
- Strategic initiatives and priorities
- Forward-looking guidance
- Market commentary
- Competitive positioning

### ⚠️ Risk & Opportunity Assessment
- External market risks
- Operational challenges
- Growth opportunities
- Strategic pivots
- Regulatory considerations

### 🔮 Forecasting
- Expected business trajectory
- Key growth drivers
- Potential headwinds
- Strategic recommendations

---

## 🛠️ Configuration Options

### Sidebar Controls

**Document Management:**
- 🗑️ Reset Download Folder – clears all downloaded PDFs
- 💾 Save Vector DB – persists FAISS index to disk
- 📂 Load Vector DB – loads existing FAISS index

### Customizable Parameters (in raw.py)

```python
# Chunking settings
chunk_size = 500
chunk_overlap = 50

# Retrieval settings
k = 5  # Number of similar chunks to retrieve

# LLM settings
model = "openai/gpt-oss-120b"
temperature = 0.3

# PDF age filter
ONE_YEAR_DAYS = 365
```

---

## 📊 Features in Detail

### 🎯 Intelligent PDF Classification
Automatically detects and labels:
- Earnings call transcripts
- Investor presentations
- Annual reports
- Quarterly results

### 🧹 Smart Document Cleanup
- Parses PDF metadata (creation date, modification date)
- Removes documents older than 1 year
- Keeps storage lean and relevant

### 🔍 Advanced Retrieval
- Semantic search using dense embeddings
- Relevance scoring for transparency
- Multi-document context aggregation
- URL content integration

### 💾 Persistent Storage
- Save vector database for quick reloading
- No need to reprocess documents
- Resume analysis across sessions

### 📈 Live Market Integration
- Fetches current NSE stock prices
- Extracts symbol from Screener URL
- Real-time data for context

---

## 🧪 Sample Questions Library

### Financial Performance
```
What was the revenue for the last quarter?
How did net profit change compared to the previous quarter?
What is the current debt-to-equity ratio?
```

### Operational Metrics
```
What were the key operational highlights?
How many new clients were added?
What is the employee headcount trend?
```

### Management Insights
```
What strategic priorities did management emphasize?
What concerns did management address?
What is the outlook for the next quarter?
```

### Industry Analysis
```
How is the company positioned versus competitors?
What industry trends are affecting the business?
What regulatory changes were discussed?
```

---

## 🔐 Security & Privacy

- API keys stored in `.env` (never committed to Git)
- Local vector database (no data sent to external services except LLM)
- PDF files stored locally
- No personal data collection

---

## 🚧 Known Limitations

1. **Embeddings**: Using HuggingFace open-source model (good but not OpenAI-level)
2. **PDF Parsing**: Scanned documents may not extract cleanly
3. **Table Recognition**: Complex financial tables may lose structure
4. **BSE PDFs**: Some BSE announcements require iframe handling
5. **Rate Limits**: Groq API has rate limits on free tier

---

## 🔮 Future Enhancements

- [ ] Add MySQL/PostgreSQL logging for query history
- [ ] Support multiple companies in single session
- [ ] Export analysis reports to PDF
- [ ] Add chart generation from financial data
- [ ] Implement caching for faster repeat queries
- [ ] Add comparison mode (Company A vs Company B)
- [ ] OCR support for scanned documents
- [ ] Email alerts for new quarterly results
- [ ] API endpoint version (FastAPI)
- [ ] Multi-language support

---

## 🐛 Troubleshooting

### GROQ_API_KEY not found
```bash
# Make sure .env file exists with:
GROQ_API_KEY=your_actual_key_here
```

### FAISS index not loading
```bash
# Delete and rebuild:
rm -rf faiss_index/
# Then reprocess documents in the app
```

### PDF download errors
- Check internet connection
- Verify Screener.in URL is correct
- Some PDFs may be behind authentication

### Module not found errors
```bash
pip install -r requirements.txt --upgrade
```

---

## 📚 Documentation & Resources

- [LangChain Docs](https://python.langchain.com/)
- [Groq API Docs](https://console.groq.com/docs)
- [FAISS Documentation](https://faiss.ai/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Screener.in](https://www.screener.in/)

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👨‍💻 Author

Built with ❤️ for intelligent financial analysis

---

## 💬 Support

For issues or questions:
- Open a GitHub Issue
- Email: prathameshshete609@gmail.com

---

## ⭐ Star This Repo!

If you find this project useful, please consider giving it a star ⭐

---

**Disclaimer**: This tool is for educational and research purposes only. It does not provide investment advice. Always consult with qualified financial advisors before making investment decisions.