from typing import Dict
from langchain_core.messages import HumanMessage, SystemMessage
from .state import AgentState
from .assistant import llm

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