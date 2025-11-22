from typing import Dict
from langchain_core.messages import HumanMessage, SystemMessage
from .state import AgentState
from .assistant import llm

import os
from tavily import TavilyClient

from firecrawl import Firecrawl
firecrawl = Firecrawl(api_key=os.environ["FIRECRAWL_API_KEY"])


tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

INTENT_SYSTEM_PROMPT = """
You are an intent extraction module for a research agent.
Your goal is to convert the user's message into a focused search query.

Guidelines:
- Capture the essential topic and subtopics.
- Keep details required for accurate search.
- Remove conversational phrasing and irrelevant context.
- Prefer a short descriptive sentence or compact clause.
- Do NOT answer the question.
- Return ONLY the search query text.
"""

def intent_node(state: AgentState) -> Dict:
    """
    Generate a clean search query from the latest user message.
    
    This node uses the shared LLM to interpret the user's intent and produce
    a concise search query suitable for downstream web search.
    """
    if not state.messages:
        raise ValueError("No messages found in state.")
    
    user_msg = state.messages[-1]

    messages = [
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=user_msg.content),
    ]

    result = llm.invoke(messages)
    search_query = result.content.strip()

    return {
        "search_query": search_query
    }


def search_node(state: AgentState) -> Dict:
    """
    Perform a web search using the extracted search_query and store the
    raw Tavily results and list of URLs in state.

    This node does not generate assistant messages; it updates structured
    state fields for downstream fetch and answer nodes.
    """
    if not state.search_query:
        raise ValueError("search_node requires state.search_query to be set.")

    # Call Tavily Search with default parameters
    response = tavily_client.search(
        query=state.search_query,
        include_answers=True
    )

    # Extract structured data from the response
    results = response.get("results", [])
    urls = [item.get("url") for item in results if item.get("url")]

    return {
        "search_results": results,
        "urls": urls
    }


def fetch_node(state: AgentState) -> dict:
    documents = []

    # Optional config knobs — customize anytime
    SCRAPE_CONFIG = {
        "formats": ["markdown"],
        "only_main_content": True,
        "exclude_tags": ["nav", "footer", "aside"],
        "remove_base64_images": True,
        "block_ads": True,
        "mobile": False,
        "proxy": "auto",
    }

    for url in state.urls:
        try:
            doc = firecrawl.scrape(
                url=url,
                **SCRAPE_CONFIG
            )

            markdown = getattr(doc, "markdown", "")
            documents.append({
                "url": url,
                "content": markdown,
                "error": None
            })

        except Exception as e:
            documents.append({
                "url": url,
                "content": "",
                "error": str(e)
            })

    return {"documents": documents}
    return {"documents": documents}