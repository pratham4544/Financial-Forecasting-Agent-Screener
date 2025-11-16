# TCS Financial Forecasting Agent

An AI-powered FastAPI application that generates comprehensive business outlook forecasts for Tata Consultancy Services (TCS) by autonomously analyzing financial documents, earnings call transcripts, and market data.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [AI Stack & Reasoning Approach](#ai-stack--reasoning-approach)
- [Agent & Tool Design](#agent--tool-design)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
- [API Usage](#api-usage)
- [Guardrails & Evaluation](#guardrails--evaluation)
- [Limits & Tradeoffs](#limits--tradeoffs)
- [Extending the System](#extending-the-system)

## 🎯 Overview

This project implements an **AI agent** (not just a Q&A system) that goes beyond simple question answering. It autonomously:

1. **Discovers and downloads** financial documents (quarterly reports, earnings call transcripts) from Screener.in
2. **Extracts** key financial metrics using intelligent parsing and pattern matching
3. **Analyzes** qualitative insights from earnings calls using RAG (Retrieval-Augmented Generation)
4. **Fetches** live market data to provide current context
5. **Synthesizes** all information into a reasoned, structured forecast for the upcoming quarter

The agent uses **LangChain's ReAct (Reasoning + Acting) framework** to orchestrate multiple specialized tools and generate coherent, data-grounded forecasts.

## 🏗 Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Application                      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │         POST /forecast/tcs (Main Endpoint)             │  │
│  └───────────────────┬────────────────────────────────────┘  │
│                      │                                        │
│  ┌───────────────────▼────────────────────────────────────┐  │
│  │              AgentPipeline (Orchestrator)              │  │
│  │  • Data Preparation (Download PDFs)                    │  │
│  │  • Tool Initialization                                 │  │
│  │  • LangChain ReAct Agent Execution                     │  │
│  │  • Result Synthesis & Structuring                      │  │
│  └───────────────────┬────────────────────────────────────┘  │
│                      │                                        │
│  ┌───────────────────▼────────────────────────────────────┐  │
│  │              Specialized Tools (LangChain)             │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │  FinancialDataExtractorTool                      │  │  │
│  │  │  • Web scraping (Screener.in tables)             │  │  │
│  │  │  • PDF text extraction (pdfminer, PyPDF2)        │  │  │
│  │  │  • Pattern matching for metrics                  │  │  │
│  │  │  • Growth rate calculations                      │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │  QualitativeAnalysisTool (RAG-based)             │  │  │
│  │  │  • PDF loading & chunking                        │  │  │
│  │  │  • Sentence-Transformers embeddings              │  │  │
│  │  │  • FAISS vector store                            │  │  │
│  │  │  • Semantic search & LLM synthesis               │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │  MarketDataTool                                  │  │  │
│  │  │  • NSE live stock prices (nse-live-stocks)       │  │  │
│  │  │  • Current market context                        │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────┘  │
│                      │                                        │
│  ┌───────────────────▼────────────────────────────────────┐  │
│  │            MySQL Database (SQLAlchemy)                 │  │
│  │  • Request logging (incoming requests)                 │  │
│  │  • Response logging (JSON outputs)                     │  │
│  │  • Execution metadata (time, tools used, status)       │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Component Breakdown

1. **FastAPI Layer** (`app/main.py`)
   - RESTful API endpoints
   - Request validation with Pydantic
   - Database session management
   - Error handling and logging

2. **Service Layer** (`app/services/`)
   - `agent_pipeline.py`: Main orchestration logic
   - `database_service.py`: Database operations

3. **Tool Layer** (`app/tools/`)
   - Each tool is a self-contained module
   - Wrapped as LangChain Tools for agent use
   - Implements specific domain expertise

4. **Model Layer** (`app/models/`)
   - `database.py`: SQLAlchemy ORM models
   - `schemas.py`: Pydantic models for validation

5. **Utility Layer** (`app/utils/`)
   - Reusable helper functions
   - Web scraping, PDF processing

## 🤖 AI Stack & Reasoning Approach

### AI Components Used

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM Provider** | OpenAI (GPT-4) / Groq | Core reasoning and synthesis |
| **Agent Framework** | LangChain (ReAct) | Tool orchestration and multi-step reasoning |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Text embeddings for RAG |
| **Vector Database** | FAISS (local) | Semantic search over transcripts |
| **PDF Extraction** | pdfminer.six, PyPDF2 | Extract text from PDFs |
| **Web Scraping** | BeautifulSoup4 | Parse HTML tables from Screener.in |
| **Market Data** | nse-live-stocks | Live NSE stock prices |

### Reasoning Approach

The agent uses **ReAct (Reasoning + Acting)** prompting:

```
Thought: I need financial metrics for the last 2 quarters
Action: FinancialDataExtractor
Action Input: https://www.screener.in/company/TCS/consolidated/,2
Observation: [Financial metrics extracted...]

Thought: Now I need management sentiment from transcripts
Action: QualitativeAnalysis
Action Input: comprehensive overall analysis
Observation: [Management outlook and themes identified...]

Thought: I should get current market price for context
Action: MarketData
Action Input: TCS
Observation: [Current price: ₹3,850...]

Thought: I now have all information to generate forecast
Final Answer: {JSON forecast with all insights synthesized}
```

### What the AI Achieves End-to-End

1. **Autonomous Data Discovery**
   - Scrapes Screener.in to find available PDFs
   - Downloads relevant quarterly reports and transcripts
   - Classifies documents by type (transcript, presentation, financial report)

2. **Intelligent Metric Extraction**
   - Parses unstructured PDF text
   - Identifies financial metrics using regex patterns
   - Calculates derived metrics (YoY growth, margins)
   - Handles multiple PDF formats robustly

3. **Deep Qualitative Analysis**
   - Builds semantic search index from transcripts
   - Identifies recurring themes across quarters
   - Extracts management sentiment (positive/neutral/negative)
   - Finds forward-looking statements
   - Lists risks and opportunities mentioned

4. **Synthesis Quality**
   - Combines quantitative + qualitative + market data
   - Generates coherent narrative forecast
   - Provides confidence levels
   - Includes appropriate disclaimers

## 🛠 Agent & Tool Design

### Master Agent Prompt

The agent is guided by a comprehensive system prompt (see `app/services/agent_pipeline.py`):

```python
You are a financial forecasting agent specialized in analyzing company performance
and generating business outlook forecasts.

**Your Task:**
Analyze the company's financial performance and generate a structured forecast.

**Process:**
1. Use FinancialDataExtractor to get quantitative metrics
2. Use QualitativeAnalysis to understand management sentiment
3. Use MarketData to get current market context
4. Synthesize all information into a coherent forecast

**Output Format:**
Your final answer MUST be a valid JSON object with:
- summary: Executive summary
- financial_trends: List of key trends
- qualitative_assessment: Management outlook
- outlook_next_quarter: Future forecast
- key_risks: List of risks
- key_opportunities: List of opportunities
- confidence_level: high/medium/low

**Guidelines:**
- Base all statements on factual data from tools
- Be specific with numbers and metrics
- Distinguish facts from forward-looking statements
- Provide balanced view of risks and opportunities
```

### Tool 1: FinancialDataExtractorTool

**Purpose:** Extract numerical financial metrics from quarterly reports and web tables.

**Capabilities:**
- Web scraping: Parses HTML tables from Screener.in
- PDF extraction: Uses multiple strategies (PyPDF2, pdfminer) for robustness
- Pattern matching: Regex-based extraction of Revenue, Profit, Margins, EPS
- Metric calculation: Computes YoY growth rates
- Structured output: Returns normalized JSON format

**Input:** `"company_url,quarters"` (e.g., `"https://screener.in/company/TCS,2"`)

**Output:** Structured text with metrics per quarter

**Key Design Decisions:**
- Multiple fallback strategies for PDF reading
- Defensive parsing (handles missing data gracefully)
- Focus on most critical metrics (extensible for more)

### Tool 2: QualitativeAnalysisTool

**Purpose:** Perform RAG-based semantic analysis of earnings call transcripts.

**Capabilities:**
- Document loading: Processes multiple PDF transcripts
- Text chunking: RecursiveCharacterTextSplitter (1000 char chunks, 200 overlap)
- Embedding generation: Uses sentence-transformers for semantic representation
- Vector storage: FAISS for efficient similarity search
- Semantic queries: Retrieves most relevant context (top-k=5)
- LLM synthesis: Generates insights from retrieved passages

**Input:** Natural language query (e.g., `"What is management's outlook?"`)

**Output:** Synthesized answer with source citations

**RAG Pipeline:**
```
Transcripts → Chunking → Embeddings → FAISS Index
                                           ↓
Query → Embedding → Similarity Search → Top-K Chunks
                                           ↓
Chunks + Query → LLM Prompt → Synthesized Answer
```

**Key Design Decisions:**
- Persistent FAISS index (saved to disk, reusable)
- Comprehensive analysis mode (runs multiple queries)
- Source attribution (tracks which PDF each insight came from)

### Tool 3: MarketDataTool

**Purpose:** Fetch live market data for real-time context.

**Capabilities:**
- Real-time price: Current NSE stock price via nse-live-stocks
- Market context: Price changes, timestamps
- Graceful degradation: Fallback message if data unavailable

**Input:** Stock symbol (e.g., `"TCS"`)

**Output:** Current price and market metrics

**Key Design Decisions:**
- Optional integration (system works without it)
- Fallback handling for API failures
- Extensible for additional market metrics

## ✨ Features

- ✅ **Autonomous Agent**: Multi-step reasoning with LangChain ReAct
- ✅ **Multi-source Analysis**: Combines financial reports + transcripts + market data
- ✅ **Structured JSON Output**: Predictable, machine-readable forecasts
- ✅ **MySQL Logging**: Complete audit trail of requests and responses
- ✅ **RAG Implementation**: Semantic search over earnings call transcripts
- ✅ **Robust PDF Processing**: Multiple extraction strategies
- ✅ **Live Market Data**: Real-time stock prices (optional)
- ✅ **RESTful API**: FastAPI with automatic OpenAPI docs
- ✅ **Type Safety**: Pydantic models throughout
- ✅ **Configurable LLM**: Supports OpenAI and Groq

## 📁 Project Structure

```
Financial-Forecasting-Agent-Screener/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application
│   ├── config.py                  # Configuration management
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py            # SQLAlchemy models
│   │   └── schemas.py             # Pydantic schemas
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── financial_extractor.py # Tool 1: Financial data extraction
│   │   ├── qualitative_analysis.py# Tool 2: RAG-based analysis
│   │   └── market_data.py         # Tool 3: Live market data
│   ├── services/
│   │   ├── __init__.py
│   │   ├── agent_pipeline.py      # Main agent orchestration
│   │   └── database_service.py    # Database operations
│   └── utils/
│       ├── __init__.py
│       ├── scraper.py             # Web scraping utilities
│       └── pdf_downloader.py      # PDF download utilities
├── data/
│   ├── pdfs/                      # Downloaded PDFs (gitignored)
│   └── faiss_index/               # FAISS vector store (gitignored)
├── research/                      # Initial research code
│   └── raw_code.py
├── .env.example                   # Environment variable template
├── .gitignore
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── LICENSE
```

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.10+** (tested on 3.10, 3.11)
- **MySQL 8.0+** (running and accessible)
- **API Key** for OpenAI or Groq

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Financial-Forecasting-Agent-Screener
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** If you encounter issues with `faiss-cpu`, you may need to install it separately:

```bash
# For CPU-only FAISS
pip install faiss-cpu

# For GPU support (if available)
pip install faiss-gpu
```

### Step 4: Set Up MySQL Database

1. **Create the database:**

```sql
CREATE DATABASE tcs_forecast CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

2. **Create a user (optional but recommended):**

```sql
CREATE USER 'tcs_user'@'localhost' IDENTIFIED BY 'secure_password';
GRANT ALL PRIVILEGES ON tcs_forecast.* TO 'tcs_user'@'localhost';
FLUSH PRIVILEGES;
```

3. **Verify connection:**

```bash
mysql -u tcs_user -p tcs_forecast
```

### Step 5: Configure Environment Variables

1. **Copy the example environment file:**

```bash
cp .env.example .env
```

2. **Edit `.env` with your settings:**

```bash
# Database Configuration
MYSQL_USER=tcs_user
MYSQL_PASSWORD=your_secure_password
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=tcs_forecast

# LLM Provider (choose one)
LLM_PROVIDER=openai  # or "groq"
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.0

# API Keys (provide at least one)
OPENAI_API_KEY=sk-your-openai-key-here
# GROQ_API_KEY=your-groq-key-here

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# RAG Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RAG_K=5
```

**Important:**
- Ensure you have a valid API key for your chosen LLM provider
- The database credentials must match your MySQL setup
- Never commit `.env` to version control (already in `.gitignore`)

### Step 6: Initialize Database Tables

The database tables will be created automatically when you first run the application, but you can verify:

```python
# Test database initialization
python -c "from app.models.database import init_db; init_db(); print('Database initialized successfully')"
```

## 🏃 How to Run

### Option 1: Using Uvicorn Directly

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Using Python

```bash
python -m app.main
```

### Option 3: Production Mode (No Reload)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Verify the Server is Running

Open your browser and navigate to:

- **API Documentation:** http://localhost:8000/docs
- **Alternative Docs:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health

You should see the interactive Swagger UI documentation.

## 📡 API Usage

### Main Endpoint: Generate Forecast

**POST** `/forecast/tcs`

**Request Body:**

```json
{
  "company_url": "https://www.screener.in/company/TCS/consolidated/",
  "quarters": 2,
  "include_market_data": true
}
```

**Response:**

```json
{
  "request_id": "req_abc123def456",
  "company_symbol": "TCS",
  "timestamp": "2025-01-15T10:30:00.000Z",
  "forecast": {
    "summary": "TCS demonstrated strong performance with 5.2% YoY revenue growth in Q3 FY24, driven by cloud transformation deals and BFSI sector demand. Operating margins improved to 25.1%, reflecting operational efficiency gains.",
    "financial_trends": [
      "Revenue growth of 5.2% YoY in Q3 FY24",
      "Operating margins expanded to 25.1% from 24.8%",
      "Net profit increased by 6.1% YoY",
      "Strong deal wins totaling $8.1B in TCV"
    ],
    "qualitative_assessment": "Management expressed confidence in the deal pipeline, highlighting improved client spending in BFSI and manufacturing verticals. Strategic focus on cloud, data, and AI capabilities is gaining traction with large enterprises.",
    "outlook_next_quarter": "Expect continued revenue growth in the range of 4-6% driven by strong deal conversion, cloud migration projects, and digital transformation initiatives. Margins likely to remain stable with potential upside from automation benefits.",
    "key_risks": [
      "Macroeconomic headwinds in key markets (US, Europe)",
      "Currency fluctuations impacting revenue",
      "Wage inflation pressures",
      "Slower client decision-making cycles"
    ],
    "key_opportunities": [
      "Growing deal pipeline in cloud and AI",
      "Market share gains in digital transformation",
      "Expansion in emerging markets",
      "Strategic acquisitions for capability building"
    ],
    "confidence_level": "high",
    "disclaimer": "This forecast is generated by an AI agent based on historical data and does not constitute investment advice."
  },
  "financial_metrics": [
    {
      "quarter": "Q3 FY24",
      "total_revenue": 59381,
      "net_profit": 11342,
      "operating_margin": 25.1,
      "revenue_growth_yoy": 5.2,
      "profit_growth_yoy": 6.1,
      "eps": 30.76
    },
    {
      "quarter": "Q2 FY24",
      "total_revenue": 58229,
      "net_profit": 11077,
      "operating_margin": 24.8,
      "eps": 30.02
    }
  ],
  "qualitative_insights": {
    "key_themes": [
      "Cloud transformation and migration",
      "AI and automation adoption",
      "BFSI sector recovery",
      "Operational excellence initiatives"
    ],
    "management_sentiment": "positive",
    "forward_looking_statements": [
      "Expecting improved client budgets in H2 FY24",
      "Strong deal conversion pipeline for next quarter",
      "Continued investment in AI capabilities"
    ],
    "risks_identified": [
      "Macro headwinds",
      "Currency volatility"
    ],
    "opportunities_identified": [
      "Cloud deals accelerating",
      "AI demand increasing"
    ]
  },
  "market_data": {
    "symbol": "TCS",
    "current_price": 3850.50,
    "timestamp": "2025-01-15T10:30:00.000Z"
  },
  "quarters_analyzed": 2,
  "tools_used": [
    "FinancialDataExtractor",
    "QualitativeAnalysis",
    "MarketData"
  ],
  "execution_time_seconds": 45.3
}
```

### Using cURL

```bash
curl -X POST "http://localhost:8000/forecast/tcs" \
  -H "Content-Type: application/json" \
  -d '{
    "company_url": "https://www.screener.in/company/TCS/consolidated/",
    "quarters": 2,
    "include_market_data": true
  }'
```

### Using Python Requests

```python
import requests

response = requests.post(
    "http://localhost:8000/forecast/tcs",
    json={
        "company_url": "https://www.screener.in/company/TCS/consolidated/",
        "quarters": 2,
        "include_market_data": True
    }
)

forecast = response.json()
print(forecast["forecast"]["summary"])
```

### Additional Endpoints

#### Get Request Details

```bash
GET /requests/{request_id}
```

#### Get Recent Requests

```bash
GET /requests?company_symbol=TCS&limit=10
```

#### Test Individual Tools

```bash
# Test Financial Data Extractor
POST /tools/financial-extract?company_url=https://www.screener.in/company/TCS&quarters=2

# Test Qualitative Analysis
POST /tools/qualitative-analysis?query=What is management's outlook?

# Test Market Data
POST /tools/market-data?symbol=TCS
```

## 🛡 Guardrails & Evaluation

### Prompting Strategy

1. **System Prompt Design:**
   - Clear role definition ("financial forecasting agent")
   - Explicit instructions for tool usage
   - Strict JSON output format requirement
   - Emphasis on grounding in factual data

2. **ReAct Framework:**
   - Thought/Action/Observation loop enforces structured reasoning
   - Agent must justify each tool invocation
   - Reduces hallucination by requiring explicit data retrieval

3. **Temperature Setting:**
   - Set to 0.0 for financial analysis (deterministic outputs)
   - Reduces creative hallucination
   - Ensures consistent formatting

### Retries and Error Handling

1. **PDF Extraction:**
   - Multiple fallback strategies (PyPDF2 → pdfminer)
   - Graceful degradation if extraction fails
   - Logs warnings for manual review

2. **Agent Execution:**
   - Fallback to direct tool execution if agent fails
   - Handles parsing errors gracefully
   - Maximum iteration limit (10) prevents infinite loops

3. **API Calls:**
   - Retry logic for web scraping (connection errors)
   - Timeout settings prevent hanging requests
   - Fallback messages for market data unavailability

### Grounding Checks

1. **Data Provenance:**
   - All metrics include source attribution
   - RAG responses cite source PDF files
   - Clear separation of facts vs. synthesis

2. **Validation:**
   - Pydantic models validate all structured outputs
   - Type checking prevents malformed data
   - Required fields ensure completeness

3. **Disclaimers:**
   - Every forecast includes legal disclaimer
   - Confidence levels indicate certainty
   - Distinguishes between factual data and forward-looking statements

### Evaluation Approach

**Manual Review Checklist:**
- [ ] Are financial metrics accurate compared to source PDFs?
- [ ] Do qualitative insights appear in actual transcripts?
- [ ] Is the forecast coherent and well-reasoned?
- [ ] Are risks and opportunities balanced?
- [ ] Is the JSON structure valid?

**Automated Checks:**
- Database logging enables audit trails
- Response times tracked for performance monitoring
- Tool usage logged for debugging

## ⚖️ Limits & Tradeoffs

### Current Limitations

1. **PDF Extraction:**
   - **Scanned PDFs:** pdfminer fails on image-based PDFs
   - **Mitigation:** Would require OCR (Tesseract) integration
   - **Impact:** ~10-20% of PDFs may fail extraction

2. **Web Scraping Fragility:**
   - **HTML Changes:** Screener.in layout changes break selectors
   - **Mitigation:** Multiple selector strategies, graceful degradation
   - **Impact:** Manual selector updates needed if site changes

3. **LLM Costs:**
   - **OpenAI GPT-4:** Expensive for repeated use (~$0.03 per request)
   - **Mitigation:** Option to use Groq (faster, cheaper but less capable)
   - **Impact:** Production use requires cost monitoring

4. **RAG Limitations:**
   - **Local FAISS:** Not suitable for multi-user scale
   - **Mitigation:** Could migrate to Pinecone/Weaviate
   - **Impact:** Single-user deployment recommended

5. **Market Data:**
   - **API Reliability:** nse-live-stocks occasionally fails
   - **Mitigation:** Graceful fallback, optional inclusion
   - **Impact:** Forecast generated without market data

### Design Tradeoffs

| Choice | Alternative | Reasoning |
|--------|-------------|-----------|
| **FAISS (local)** | Pinecone/Weaviate | Simplicity, no external deps, sufficient for demo |
| **sentence-transformers** | OpenAI embeddings | Free, runs locally, good quality for financial text |
| **ReAct Agent** | Function calling | More transparent reasoning, easier to debug |
| **Direct PDF download** | API integration | Screener.in has no official API, scraping necessary |
| **Temperature 0** | Temperature 0.3 | Prioritize accuracy over creativity for finance |

### Mitigation Strategies

1. **OCR Fallback (Future):**
   ```python
   # Add to PDF extractor
   if text_extraction_fails():
       use_tesseract_ocr(pdf_path)
   ```

2. **Caching:**
   - Cache PDF downloads (avoid re-downloading)
   - Cache FAISS index (reuse across requests)
   - Implemented: FAISS index persists to disk

3. **Rate Limiting:**
   - Add to FastAPI for production use
   - Prevents abuse and manages costs

4. **Monitoring:**
   - All requests logged to MySQL
   - Execution times tracked
   - Tool usage recorded for analytics

## 🔧 Extending the System

### Adding New Tools

1. Create new tool in `app/tools/new_tool.py`:

```python
from langchain_core.tools import Tool

class NewAnalysisTool:
    def __init__(self):
        self.name = "NewAnalysis"
        self.description = "Performs new type of analysis"

    def run(self, input_str: str) -> str:
        # Your logic here
        return result

    def as_langchain_tool(self) -> Tool:
        return Tool(
            name=self.name,
            description=self.description,
            func=self.run
        )
```

2. Register in `app/services/agent_pipeline.py`:

```python
from app.tools import NewAnalysisTool

def _initialize_tools(self):
    new_tool = NewAnalysisTool()
    self.tools.append(new_tool.as_langchain_tool())
```

### Supporting Other Companies

The system is designed for TCS but can be adapted:

1. Change URL pattern in requests
2. Adjust metric extraction patterns if needed
3. Symbol mapping for market data

### Scaling for Production

**Recommendations:**

1. **Async Processing:**
   - Use Celery + Redis for background jobs
   - Return request_id immediately, poll for results

2. **Vector DB Migration:**
   - Replace FAISS with Pinecone/Weaviate
   - Enable multi-user concurrent access

3. **Docker Deployment:**
   ```dockerfile
   FROM python:3.11-slim
   COPY . /app
   RUN pip install -r requirements.txt
   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]
   ```

4. **Monitoring:**
   - Add Prometheus metrics
   - Integrate with Sentry for error tracking

## 📝 License

MIT License - see LICENSE file

## 🤝 Contributing

Contributions welcome! Please open an issue or pull request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ for Elevation AI Assessment**

*This project demonstrates AI-first engineering with autonomous agents, RAG, and structured reasoning for financial analysis.*
