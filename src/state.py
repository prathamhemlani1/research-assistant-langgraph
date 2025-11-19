from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


class AgentState(BaseModel):
    """
    Global state for the research agent.

    This model tracks the evolving workflow state across nodes in the graph,
    including conversation messages, search queries, search results,
    extracted URLs, and fetched document content.
    """

    # Conversation history (assistant + user messages)
    messages: List[BaseMessage] = Field(default_factory=list)

    # Query to run in the search step (set by intent node)
    search_query: Optional[str] = None

    # Raw search results returned from the search node (list of dicts)
    search_results: Optional[List[Dict]] = None

    # URLs extracted from search results
    urls: Optional[List[str]] = None

    # Fetched documents (HTML/text content) for downstream reasoning
    documents: Optional[List[str]] = None
