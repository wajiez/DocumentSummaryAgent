from state import AgentState
from config import llm

def chat(state: AgentState):
    response = llm.invoke(state["messages"])

    return {"messages": [response]}
