from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .state import AgentState

SYSTEM_PROMPT = (
    "You are a helpful research assistant. "
    "Use the provided tools when necessary and provide clear, concise answers."
)

llm = ChatOpenAI(model="gpt-4o-mini")

def assistant_node(state: AgentState):
    """
    LLM node for generating the assistant's next message.

    This node receives the current AgentState, constructs the LLM
    input (system prompt + conversation), invokes the language model,
    and returns an updated state containing the model's response.
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
