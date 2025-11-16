"""
QualitativeAnalysisTool: RAG-based semantic analysis of earnings call transcripts.

This tool:
1. Builds a FAISS vector store from earnings call transcripts
2. Performs semantic search to find relevant passages
3. Analyzes management sentiment and forward-looking statements
4. Identifies recurring themes, risks, and opportunities
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from app.config import settings

logger = logging.getLogger(__name__)


class QualitativeAnalysisTool:
    """
    RAG-based tool for analyzing earnings call transcripts.

    Uses FAISS vector store for semantic search and LLM for synthesis.
    """

    def __init__(self, llm=None):
        self.name = "QualitativeAnalysis"
        self.description = """
        Analyzes earnings call transcripts using RAG (Retrieval-Augmented Generation).
        Identifies management sentiment, key themes, forward-looking statements,
        risks, and opportunities from earnings calls.
        Input should be a query like "What is management's outlook?" or "Identify key risks".
        """
        self.llm = llm
        self.vector_store = None
        self.embeddings = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    def initialize_embeddings(self):
        """Initialize embedding model (lazy loading)."""
        if self.embeddings is None:
            logger.info("Initializing embedding model...")
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=settings.EMBEDDING_MODEL
                )
                logger.info("Embeddings initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing embeddings: {e}")
                # Fallback to fake embeddings for testing
                from langchain_community.embeddings import FakeEmbeddings
                self.embeddings = FakeEmbeddings(size=384)
                logger.warning("Using FakeEmbeddings as fallback")

    def build_vector_store(self, pdf_directory: Optional[str] = None) -> bool:
        """
        Build FAISS vector store from transcript PDFs.

        Args:
            pdf_directory: Directory containing PDFs (defaults to settings.PDF_DIR)

        Returns:
            True if successful, False otherwise
        """
        logger.info("Building vector store from transcripts...")

        # Initialize embeddings
        self.initialize_embeddings()

        pdf_dir = Path(pdf_directory) if pdf_directory else settings.PDF_DIR

        if not pdf_dir.exists():
            logger.error(f"PDF directory does not exist: {pdf_dir}")
            return False

        # Find transcript PDFs
        transcript_files = []
        for pdf_file in pdf_dir.glob("*.pdf"):
            # Prioritize files classified as transcripts
            if "transcript" in pdf_file.name.lower():
                transcript_files.append(pdf_file)

        if not transcript_files:
            # Fallback: use all PDFs if no transcripts found
            logger.warning("No transcript PDFs found, using all PDFs")
            transcript_files = list(pdf_dir.glob("*.pdf"))

        if not transcript_files:
            logger.error("No PDF files found in directory")
            return False

        logger.info(f"Processing {len(transcript_files)} PDF files")

        all_documents = []

        for pdf_file in transcript_files:
            try:
                logger.info(f"Loading: {pdf_file.name}")

                # Try multiple PDF loaders
                docs = self._load_pdf_robust(str(pdf_file))

                if docs:
                    # Add metadata
                    for doc in docs:
                        doc.metadata["source_file"] = pdf_file.name

                    all_documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} pages from {pdf_file.name}")
                else:
                    logger.warning(f"No content extracted from {pdf_file.name}")

            except Exception as e:
                logger.error(f"Error loading {pdf_file.name}: {e}")
                continue

        if not all_documents:
            logger.error("No documents loaded successfully")
            return False

        # Split documents into chunks
        logger.info("Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(all_documents)
        logger.info(f"Created {len(chunks)} chunks")

        # Build FAISS index
        logger.info("Building FAISS vector store...")
        try:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)

            # Save to disk
            settings.FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(settings.FAISS_INDEX_PATH))

            logger.info("Vector store built and saved successfully")
            return True

        except Exception as e:
            logger.error(f"Error building vector store: {e}")
            return False

    def load_vector_store(self) -> bool:
        """
        Load existing FAISS vector store from disk.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.initialize_embeddings()

            if not settings.FAISS_INDEX_PATH.exists():
                logger.warning("FAISS index not found on disk")
                return False

            logger.info("Loading FAISS vector store from disk...")
            self.vector_store = FAISS.load_local(
                str(settings.FAISS_INDEX_PATH),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Vector store loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False

    def _load_pdf_robust(self, pdf_path: str) -> List:
        """
        Load PDF with fallback strategies.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of Document objects
        """
        # Strategy 1: PyPDFLoader
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            if docs and any(doc.page_content.strip() for doc in docs):
                return docs
        except Exception as e:
            logger.debug(f"PyPDFLoader failed: {e}")

        # Strategy 2: PDFMinerLoader
        try:
            loader = PDFMinerLoader(pdf_path)
            docs = loader.load()
            if docs and any(doc.page_content.strip() for doc in docs):
                return docs
        except Exception as e:
            logger.debug(f"PDFMinerLoader failed: {e}")

        return []

    def query(self, question: str, k: int = None) -> Dict[str, Any]:
        """
        Query the vector store with semantic search.

        Args:
            question: Query string
            k: Number of documents to retrieve

        Returns:
            Dictionary with retrieved chunks and synthesis
        """
        if self.vector_store is None:
            # Try to load existing vector store
            if not self.load_vector_store():
                # Build new vector store
                if not self.build_vector_store():
                    return {
                        "error": "Unable to load or build vector store",
                        "chunks": [],
                        "answer": "No transcript data available for analysis."
                    }

        k = k or settings.RAG_K

        try:
            # Retrieve relevant documents
            retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
            chunks = retriever.invoke(question)

            # If LLM is available, synthesize answer
            if self.llm:
                answer = self._synthesize_answer(question, chunks)
            else:
                # Return raw chunks
                answer = self._format_chunks(chunks)

            return {
                "chunks": chunks,
                "answer": answer,
                "num_chunks": len(chunks)
            }

        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            return {
                "error": str(e),
                "chunks": [],
                "answer": f"Error during analysis: {str(e)}"
            }

    def _synthesize_answer(self, question: str, chunks: List) -> str:
        """Synthesize answer from retrieved chunks using LLM."""
        try:
            # Create context from chunks
            context = "\n\n".join([
                f"[Source: {chunk.metadata.get('source_file', 'Unknown')}]\n{chunk.page_content}"
                for chunk in chunks
            ])

            # Prompt for synthesis
            prompt = PromptTemplate(
                template="""You are analyzing earnings call transcripts for financial forecasting.

Context from transcripts:
{context}

Question: {question}

Provide a clear, concise analysis based ONLY on the information in the context above.
Focus on:
- Management's stated outlook and sentiment
- Forward-looking statements
- Risks and challenges mentioned
- Opportunities and growth drivers
- Strategic initiatives

Answer:""",
                input_variables=["context", "question"]
            )

            chain = prompt | self.llm
            response = chain.invoke({"context": context, "question": question})

            # Extract text from response
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)

        except Exception as e:
            logger.error(f"Error synthesizing answer: {e}")
            return self._format_chunks(chunks)

    def _format_chunks(self, chunks: List) -> str:
        """Format chunks into readable text."""
        if not chunks:
            return "No relevant information found in transcripts."

        output = "Relevant excerpts from earnings call transcripts:\n\n"
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get('source_file', 'Unknown')
            output += f"{i}. [From: {source}]\n"
            output += f"{chunk.page_content[:300]}...\n\n"

        return output

    def analyze_comprehensive(self) -> Dict[str, Any]:
        """
        Perform comprehensive qualitative analysis.
        Runs multiple queries to extract different insights.

        Returns:
            Dictionary with comprehensive analysis results
        """
        queries = {
            "sentiment": "What is management's overall sentiment and outlook for the business?",
            "themes": "What are the key recurring themes and focus areas mentioned?",
            "risks": "What risks, challenges, or concerns has management identified?",
            "opportunities": "What growth opportunities and positive developments were discussed?",
            "forward_statements": "What specific forward-looking statements did management make?"
        }

        results = {}

        for key, query in queries.items():
            logger.info(f"Running query: {key}")
            result = self.query(query, k=3)
            results[key] = result.get("answer", "")

        return results

    def run(self, input_str: str) -> str:
        """
        Run the tool with given input.
        Used by LangChain agent.

        Args:
            input_str: Query string

        Returns:
            String with analysis results
        """
        try:
            # If input suggests comprehensive analysis
            if "comprehensive" in input_str.lower() or "overall" in input_str.lower():
                results = self.analyze_comprehensive()

                output = "Comprehensive Qualitative Analysis:\n\n"
                output += f"Management Sentiment:\n{results.get('sentiment', 'N/A')}\n\n"
                output += f"Key Themes:\n{results.get('themes', 'N/A')}\n\n"
                output += f"Risks & Challenges:\n{results.get('risks', 'N/A')}\n\n"
                output += f"Opportunities:\n{results.get('opportunities', 'N/A')}\n\n"
                output += f"Forward-Looking Statements:\n{results.get('forward_statements', 'N/A')}\n"

                return output
            else:
                # Single query
                result = self.query(input_str)
                return result.get("answer", "No results found.")

        except Exception as e:
            logger.error(f"Error in QualitativeAnalysisTool.run: {e}")
            return f"Error performing qualitative analysis: {str(e)}"

    def as_langchain_tool(self) -> Tool:
        """Convert to LangChain Tool for use in agent."""
        return Tool(
            name=self.name,
            description=self.description,
            func=self.run
        )
