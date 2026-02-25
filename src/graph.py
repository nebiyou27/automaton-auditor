# src/graph.py
# ============
# Partial graph orchestration for detective phase (Fan-Out → Fan-In).

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.state import AgentState
from src.nodes.detectives import doc_analyst, repo_investigator
from src.nodes.aggregator import evidence_aggregator


def _route_doc_analyst(state: AgentState) -> Literal["doc_analyst", "evidence_aggregator"]:
    """
    Conditional routing:
    If pdf_path is missing, skip doc_analyst and go straight to aggregator.
    """
    pdf_path = (state.get("pdf_path") or "").strip()
    if not pdf_path:
        return "evidence_aggregator"
    return "doc_analyst"


def build_graph():
    """
    Build a runnable detective-phase graph:
    - START fans out to repo_investigator (always)
    - doc_analyst is conditionally executed based on pdf_path
    - evidence_aggregator is the fan-in synchronization point
    """
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("repo_investigator", repo_investigator)
    builder.add_node("doc_analyst", doc_analyst)
    builder.add_node("evidence_aggregator", evidence_aggregator)

    # Fan-out
    builder.add_edge(START, "repo_investigator")

    # Conditional path for doc_analyst
    builder.add_conditional_edges(
        START,
        _route_doc_analyst,
        {
            "doc_analyst": "doc_analyst",
            "evidence_aggregator": "evidence_aggregator",
        },
    )

    # Fan-in to aggregator (both detectives converge)
    builder.add_edge("repo_investigator", "evidence_aggregator")
    builder.add_edge("doc_analyst", "evidence_aggregator")

    # End
    builder.add_edge("evidence_aggregator", END)

    return builder.compile()


if __name__ == "__main__":
    g = build_graph()
    print("✅ Detective graph compiled successfully")
    print("Nodes:", list(g.nodes))