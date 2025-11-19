from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .assistant import assistant_node

def create_graph():
    """
    Construct and return the core LangGraph for the research assistant.

    This graph currently contains a single assistant node responsible for
    generating responses. Additional tool nodes, memory components, or
    workflow steps can be added as the project evolves.
    """
    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant_node)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", END)
    return builder.compile()

