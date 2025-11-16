from app.services.financial_tool import extract_financials_from_screener
from app.services.qualitative_tool import qualitative_analysis
from app.services.market_tool import get_live_price
from app.services.knowledge_pool import KnowledgePool
from app.services.llm_client import get_llm
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
import json

# wrapper functions used by the tools (adapting tool interfaces)
def _financial_tool_fn(input_text: str):
    """
    input_text expected format: "<company_url>||<quarters>"
    """
    try:
        parts = input_text.split("||")
        url = parts[0].strip()
        quarters = int(parts[1]) if len(parts) > 1 else 2
    except Exception:
        url = input_text.strip()
        quarters = 2
    data = extract_financials_from_screener(url, quarters=quarters)
    return json.dumps(data)

def _qualitative_tool_fn(input_text: str):
    """
    input_text expected: "<company_url>||<quarters>"
    """
    try:
        parts = input_text.split("||")
        url = parts[0].strip()
        quarters = int(parts[1]) if len(parts) > 1 else 2
    except Exception:
        url = input_text.strip()
        quarters = 2
    data = qualitative_analysis(url, quarters=quarters)
    return json.dumps(data)

def _market_tool_fn(input_text: str):
    """
    input_text expected: NSE symbol or 'TCS'
    """
    sym = input_text.strip() or "TCS"
    data = get_live_price(sym)
    return json.dumps(data)

# Create LangChain tools
financial_tool = Tool(name="financial_tool",
                      func=_financial_tool_fn,
                      description="Extract key financial metrics from screener / company page. Input: '<company_url>||<quarters>'")

qualitative_tool = Tool(name="qualitative_tool",
                        func=_qualitative_tool_fn,
                        description="Perform RAG semantic analysis of earnings call transcripts. Input: '<company_url>||<quarters>'")

market_data_tool = Tool(name="market_tool",
                        func=_market_tool_fn,
                        description="Fetch live market price for NSE symbol. Input: 'TCS' or other symbol.")

tools = [financial_tool, qualitative_tool, market_data_tool]

def run_pipeline_sync(company_url: str, quarters: int = 2):
    """
    This function builds an agent with three tools and asks it to orchestrate analysis.
    It still returns a structured JSON (deterministic pieces come from the tools).
    """
    pool = KnowledgePool()

    # Build agent
    llm = get_llm()
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

    # Prepare prompt: instruct agent to use tools and return JSON
    system_instruct = (
        "You are an analysis agent. Use the provided tools to fetch numeric financials, qualitative "
        "analysis from transcripts, and live market price. Produce a structured JSON with keys: "
        "company, quarters_analyzed, financial_metrics, qualitative_summary, risks_and_opportunities, "
        "market_price, forecast. Do not invent numeric tables. Use only tool outputs for factual values. "
        "For forecast, synthesize a short reasoned qualitative forecast using the tool outputs."
    )

    task_prompt = (
        f"{system_instruct}\n\nCompany URL: {company_url}\nQuarters: {quarters}\n\n"
        "Steps:\n1) Call financial_tool with '<company_url>||<quarters>' to get numeric metrics.\n"
        "2) Call qualitative_tool with '<company_url>||<quarters>' to get management commentary analysis.\n"
        "3) Call market_tool with 'TCS' to get live price.\n"
        "Finally, synthesize the results and return a JSON object (no extra text)."
    )

    # Run the agent (it will call the tools as needed)
    raw_agent_result = agent.run(task_prompt)

    # The agent's output should be JSON text — attempt to parse. If the agent returned plain text, fallback to building JSON.
    result = None
    try:
        result = json.loads(raw_agent_result)
    except Exception:
        # fallback deterministic pipeline: call tools directly and synthesize
        fin_json = json.loads(_financial_tool_fn(f"{company_url}||{quarters}"))
        qual_json = json.loads(_qualitative_tool_fn(f"{company_url}||{quarters}"))
        market_json = json.loads(_market_tool_fn("TCS"))

        # simple forecast derivation (deterministic)
        revs = fin_json.get("revenue", [])
        growth_est = "insufficient data"
        try:
            if len(revs) >= 2 and revs[1] != 0:
                growth = (revs[0] - revs[1]) / abs(revs[1])
                growth_est = f"{growth:.2%}"
        except Exception:
            pass

        forecast = {
            "revenue_growth_estimate": growth_est,
            "margin_outlook": "based on qualitative signals",
            "confidence": "medium"
        }

        result = {
            "company": company_url,
            "quarters_analyzed": quarters,
            "financial_metrics": fin_json,
            "qualitative_summary": qual_json,
            "risks_and_opportunities": {
                "risks": qual_json.get("risks"),
                "opportunities": qual_json.get("opportunities")
            },
            "market_price": market_json,
            "forecast": forecast
        }

    return result
