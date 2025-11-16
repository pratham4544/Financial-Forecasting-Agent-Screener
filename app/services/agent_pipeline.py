"""
Agent Pipeline: Orchestrates all tools to generate financial forecast.

This module implements the core agentic reasoning that:
1. Coordinates multiple specialized tools
2. Synthesizes information from different sources
3. Generates structured forecast output
"""
import logging
import json
from typing import Dict, Any, List, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from app.config import settings
from app.tools import FinancialDataExtractorTool, QualitativeAnalysisTool, MarketDataTool
from app.utils.scraper import scrape_screener_pdfs, extract_company_symbol
from app.utils.pdf_downloader import download_pdfs
from app.models.schemas import ForecastOutput, FinancialMetrics, QualitativeInsights, MarketData

logger = logging.getLogger(__name__)


class AgentPipeline:
    """
    Main pipeline that orchestrates the forecasting process.

    This class manages the entire workflow from data collection to forecast generation.
    """

    def __init__(self):
        """Initialize the agent pipeline with LLM and tools."""
        self.llm = self._initialize_llm()
        self.tools = self._initialize_tools()
        self.agent = None

    def _initialize_llm(self):
        """Initialize LLM based on configuration."""
        logger.info(f"Initializing LLM: {settings.LLM_PROVIDER}")

        try:
            if settings.LLM_PROVIDER == "openai":
                if not settings.OPENAI_API_KEY:
                    raise ValueError("OPENAI_API_KEY not set in environment")

                return ChatOpenAI(
                    model=settings.LLM_MODEL,
                    temperature=settings.LLM_TEMPERATURE,
                    api_key=settings.OPENAI_API_KEY
                )

            elif settings.LLM_PROVIDER == "groq":
                if not settings.GROQ_API_KEY:
                    raise ValueError("GROQ_API_KEY not set in environment")

                return ChatGroq(
                    model=settings.LLM_MODEL,
                    temperature=settings.LLM_TEMPERATURE,
                    api_key=settings.GROQ_API_KEY
                )

            else:
                raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")

        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise

    def _initialize_tools(self) -> List:
        """Initialize all specialized tools."""
        logger.info("Initializing tools...")

        financial_tool = FinancialDataExtractorTool()
        qualitative_tool = QualitativeAnalysisTool(llm=self.llm)
        market_tool = MarketDataTool()

        tools = [
            financial_tool.as_langchain_tool(),
            qualitative_tool.as_langchain_tool(),
            market_tool.as_langchain_tool()
        ]

        logger.info(f"Initialized {len(tools)} tools")
        return tools

    def prepare_data(self, company_url: str, quarters: int = 2) -> Dict[str, Any]:
        """
        Prepare data by downloading and processing PDFs.

        Args:
            company_url: URL to company's Screener.in page
            quarters: Number of quarters to analyze

        Returns:
            Dictionary with preparation status
        """
        logger.info("Preparing data: downloading PDFs...")

        try:
            # Scrape PDF links
            pdf_links = scrape_screener_pdfs(company_url)

            if not pdf_links:
                logger.warning("No PDF links found")
                return {
                    "status": "warning",
                    "message": "No PDFs found to download",
                    "pdfs_downloaded": 0
                }

            # Download PDFs (limit based on quarters needed)
            max_pdfs = min(quarters * 3, len(pdf_links))  # ~3 docs per quarter
            downloaded = download_pdfs(pdf_links, max_pdfs=max_pdfs)

            logger.info(f"Downloaded {len(downloaded)} PDFs")

            return {
                "status": "success",
                "pdfs_downloaded": len(downloaded),
                "pdf_files": downloaded
            }

        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return {
                "status": "error",
                "message": str(e),
                "pdfs_downloaded": 0
            }

    def generate_forecast(
        self,
        company_url: str,
        quarters: int = 2,
        include_market_data: bool = True
    ) -> Dict[str, Any]:
        """
        Generate complete forecast using agent reasoning.

        Args:
            company_url: URL to company's Screener.in page
            quarters: Number of quarters to analyze
            include_market_data: Whether to include live market data

        Returns:
            Dictionary with forecast and supporting data
        """
        logger.info(f"Generating forecast for {company_url}")

        try:
            # Step 1: Prepare data
            prep_result = self.prepare_data(company_url, quarters)

            # Step 2: Extract company symbol
            company_symbol = extract_company_symbol(company_url)

            # Step 3: Use agent for intelligent orchestration
            forecast_result = self._run_agent_forecast(
                company_url,
                company_symbol,
                quarters,
                include_market_data
            )

            return forecast_result

        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _run_agent_forecast(
        self,
        company_url: str,
        company_symbol: str,
        quarters: int,
        include_market_data: bool
    ) -> Dict[str, Any]:
        """
        Run agent-based forecast generation.

        Args:
            company_url: Company URL
            company_symbol: Stock symbol
            quarters: Number of quarters
            include_market_data: Include market data flag

        Returns:
            Forecast results
        """
        logger.info("Running agent-based forecast...")

        # Create agent with ReAct prompting
        agent_prompt = self._create_agent_prompt()

        try:
            # Create agent
            agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=agent_prompt
            )

            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10
            )

            # Construct task for agent
            task = self._construct_agent_task(
                company_url,
                company_symbol,
                quarters,
                include_market_data
            )

            # Execute agent
            logger.info("Executing agent...")
            result = agent_executor.invoke({"input": task})

            # Parse and structure output
            forecast_data = self._parse_agent_output(result)

            return forecast_data

        except Exception as e:
            logger.error(f"Error in agent execution: {e}")

            # Fallback: Direct tool execution
            logger.info("Falling back to direct tool execution...")
            return self._fallback_direct_execution(
                company_url,
                company_symbol,
                quarters,
                include_market_data
            )

    def _create_agent_prompt(self) -> PromptTemplate:
        """Create the main agent prompt template."""
        template = """You are a financial forecasting agent specialized in analyzing company performance and generating business outlook forecasts.

You have access to the following tools:
{tools}

Use these tools to gather comprehensive information and generate a reasoned forecast.

**Your Task:**
Analyze the company's financial performance and generate a structured forecast.

**Process:**
1. Use FinancialDataExtractor to get quantitative metrics (revenue, profit, margins, growth)
2. Use QualitativeAnalysis to understand management sentiment, outlook, risks, and opportunities
3. Use MarketData to get current market context
4. Synthesize all information into a coherent forecast

**Output Format:**
Your final answer MUST be a valid JSON object with this structure:
{{
  "summary": "Brief executive summary",
  "financial_trends": ["Trend 1", "Trend 2", ...],
  "qualitative_assessment": "Management outlook and sentiment",
  "outlook_next_quarter": "Forecast for upcoming quarter",
  "key_risks": ["Risk 1", "Risk 2", ...],
  "key_opportunities": ["Opportunity 1", "Opportunity 2", ...],
  "confidence_level": "high/medium/low"
}}

**Important Guidelines:**
- Base all statements on factual data from tools
- Be specific with numbers and metrics
- Identify clear trends and patterns
- Distinguish between facts and forward-looking statements
- Provide a balanced view of risks and opportunities

Use the following format:

Question: {input}
Thought: Consider what information I need
Action: The tool to use
Action Input: The input to the tool
Observation: The tool's output
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I have all the information needed to provide the final answer
Final Answer: {{JSON object with forecast}}

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

        return PromptTemplate(
            template=template,
            input_variables=["input", "tools", "agent_scratchpad"]
        )

    def _construct_agent_task(
        self,
        company_url: str,
        company_symbol: str,
        quarters: int,
        include_market_data: bool
    ) -> str:
        """Construct the task description for the agent."""
        task = f"""Analyze {company_symbol} and generate a comprehensive business forecast.

Company URL: {company_url}
Quarters to analyze: {quarters}
Include market data: {include_market_data}

Steps:
1. Extract financial metrics for the last {quarters} quarters using FinancialDataExtractor with input "{company_url},{quarters}"
2. Perform comprehensive qualitative analysis using QualitativeAnalysis with input "comprehensive overall analysis"
3. {"Get current market price using MarketData with input '" + company_symbol + "'" if include_market_data else "Skip market data"}
4. Synthesize all findings into a structured JSON forecast

Generate the forecast now."""

        return task

    def _parse_agent_output(self, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse agent output into structured format."""
        try:
            output_text = agent_result.get("output", "")

            # Try to extract JSON from output
            if "{" in output_text and "}" in output_text:
                # Find JSON object
                start_idx = output_text.find("{")
                end_idx = output_text.rfind("}") + 1
                json_str = output_text[start_idx:end_idx]

                forecast_data = json.loads(json_str)

                return {
                    "status": "success",
                    "forecast": forecast_data,
                    "raw_output": output_text
                }

            else:
                # No JSON found, return text output
                return {
                    "status": "partial",
                    "forecast": {
                        "summary": output_text,
                        "financial_trends": [],
                        "qualitative_assessment": output_text,
                        "outlook_next_quarter": "See summary",
                        "key_risks": [],
                        "key_opportunities": [],
                        "confidence_level": "low"
                    },
                    "raw_output": output_text
                }

        except Exception as e:
            logger.error(f"Error parsing agent output: {e}")
            return {
                "status": "error",
                "error": str(e),
                "raw_output": str(agent_result)
            }

    def _fallback_direct_execution(
        self,
        company_url: str,
        company_symbol: str,
        quarters: int,
        include_market_data: bool
    ) -> Dict[str, Any]:
        """
        Fallback: Direct execution of tools without agent.
        Used when agent execution fails.
        """
        logger.info("Executing tools directly...")

        results = {}

        # Execute each tool directly
        try:
            # Financial data
            financial_tool = FinancialDataExtractorTool()
            financial_output = financial_tool.run(f"{company_url},{quarters}")
            results["financial_data"] = financial_output

            # Qualitative analysis
            qualitative_tool = QualitativeAnalysisTool(llm=self.llm)
            qualitative_output = qualitative_tool.run("comprehensive overall analysis")
            results["qualitative_analysis"] = qualitative_output

            # Market data
            if include_market_data:
                market_tool = MarketDataTool()
                market_output = market_tool.run(company_symbol)
                results["market_data"] = market_output

            # Synthesize into forecast format
            forecast = self._synthesize_fallback_forecast(results)

            return {
                "status": "success",
                "forecast": forecast,
                "method": "fallback_direct"
            }

        except Exception as e:
            logger.error(f"Error in fallback execution: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _synthesize_fallback_forecast(self, tool_results: Dict[str, str]) -> Dict[str, Any]:
        """Synthesize tool outputs into forecast format."""
        # Extract key information from tool outputs
        financial_data = tool_results.get("financial_data", "")
        qualitative_data = tool_results.get("qualitative_analysis", "")
        market_data = tool_results.get("market_data", "")

        # Create structured forecast
        forecast = {
            "summary": f"Analysis based on available data:\n{financial_data[:200]}...",
            "financial_trends": self._extract_trends(financial_data),
            "qualitative_assessment": qualitative_data[:500] if qualitative_data else "Limited qualitative data available",
            "outlook_next_quarter": "Based on current trends and management commentary",
            "key_risks": ["Market volatility", "Economic conditions"],
            "key_opportunities": ["Growth initiatives", "Market expansion"],
            "confidence_level": "medium"
        }

        return forecast

    def _extract_trends(self, financial_text: str) -> List[str]:
        """Extract financial trends from text."""
        trends = []

        # Simple pattern matching for trends
        if "growth" in financial_text.lower():
            trends.append("Revenue growth observed")
        if "margin" in financial_text.lower():
            trends.append("Margin performance noted")
        if "profit" in financial_text.lower():
            trends.append("Profitability metrics available")

        if not trends:
            trends.append("Financial data extracted")

        return trends


def create_forecast_agent(company_url: str, quarters: int = 2, include_market_data: bool = True) -> Dict[str, Any]:
    """
    Convenience function to create and run forecasting agent.

    Args:
        company_url: Company Screener.in URL
        quarters: Number of quarters to analyze
        include_market_data: Include live market data

    Returns:
        Forecast results dictionary
    """
    pipeline = AgentPipeline()
    return pipeline.generate_forecast(company_url, quarters, include_market_data)
