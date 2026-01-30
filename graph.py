from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Optional, List

from agents import (
    chat_agent,
    summarizer_agent,
    voice_agent,
    reference_agent,
    doubt_agent,  
)

# ✅ State
class StudyMateState(TypedDict):
    user_id: str
    document_id: str
    agent_type: str
    user_query: Optional[str]
    document_text: Optional[str]
    chat_history: List[str]
    output: Optional[object]


# ✅ Router
def route_agent(state: StudyMateState):
    agent_type = state.get("agent_type")

    allowed = ["chat", "summarize", "voice", "reference", "doubt"]  # ✅ ADD doubt

    if agent_type not in allowed:
        # ✅ return END with error output
        state["output"] = f"Invalid agent_type. Use one of: {allowed}"
        return END

    return agent_type


# ✅ Graph
graph = StateGraph(StudyMateState)

# ✅ Nodes
graph.add_node("chat", chat_agent)
graph.add_node("summarize", summarizer_agent)
graph.add_node("voice", voice_agent)
graph.add_node("reference", reference_agent)
graph.add_node("doubt", doubt_agent)  # ✅ NEW NODE

# ✅ Start routing
graph.add_conditional_edges(
    START,
    route_agent,
    {
        "chat": "chat",
        "summarize": "summarize",
        "voice": "voice",
        "reference": "reference",
        "doubt": "doubt",  # ✅ NEW EDGE
        END: END
    },
)

# ✅ End edges
graph.add_edge("chat", END)
graph.add_edge("summarize", END)
graph.add_edge("voice", END)
graph.add_edge("reference", END)
graph.add_edge("doubt", END)

# ✅ Compile
app_graph = graph.compile()
