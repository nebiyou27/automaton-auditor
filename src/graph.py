# src/graph.py
# ============
# Full graph orchestration:
#   Fan-Out detectives -> Fan-In evidence aggregator ->
#   Fan-Out judges -> Fan-In Chief Justice -> END

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.state import AgentState
from src.nodes.detectives import doc_analyst, repo_investigator
from src.nodes.aggregator import evidence_aggregator
from src.nodes.skip import skip_doc_analyst
from src.nodes.judges import prosecutor_judge, defense_judge, techlead_judge
from src.nodes.justice import chief_justice


def _route_doc_analyst(state: AgentState) -> Literal["doc_analyst", "skip_doc_analyst"]:
    """
    Conditional routing:
    - If pdf_path is missing/blank, route to skip_doc_analyst (no-op),
      preserving fan-in semantics at evidence_aggregator.
    """
    pdf_path = (state.get("pdf_path") or "").strip()
    return "doc_analyst" if pdf_path else "skip_doc_analyst"


def build_graph():
    """
    Digital Courtroom graph:
    1) Detectives run in parallel (fan-out)
    2) EvidenceAggregator synchronizes (fan-in)
    3) Judges run in parallel (fan-out)
    4) Chief Justice synthesizes deterministically (fan-in)
    """
    builder = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    # Detective layer
    builder.add_node("repo_investigator", repo_investigator)
    builder.add_node("doc_analyst", doc_analyst)
    builder.add_node("skip_doc_analyst", skip_doc_analyst)

    # Fan-in evidence sync
    builder.add_node("evidence_aggregator", evidence_aggregator)

    # Judicial bench
    builder.add_node("prosecutor", prosecutor_judge)
    builder.add_node("defense", defense_judge)
    builder.add_node("techlead", techlead_judge)

    # Deterministic synthesis
    builder.add_node("chief_justice", chief_justice)

    # ── Detective fan-out ─────────────────────────────────────────────────────
    builder.add_edge(START, "repo_investigator")

    builder.add_conditional_edges(
        START,
        _route_doc_analyst,
        {
            "doc_analyst": "doc_analyst",
            "skip_doc_analyst": "skip_doc_analyst",
        },
    )

    # ── Detective fan-in barrier ──────────────────────────────────────────────
    builder.add_edge("repo_investigator", "evidence_aggregator")
    builder.add_edge("doc_analyst", "evidence_aggregator")
    builder.add_edge("skip_doc_analyst", "evidence_aggregator")

    # ── Judicial fan-out ──────────────────────────────────────────────────────
    builder.add_edge("evidence_aggregator", "prosecutor")
    builder.add_edge("evidence_aggregator", "defense")
    builder.add_edge("evidence_aggregator", "techlead")

    # ── Judicial fan-in to Chief Justice ──────────────────────────────────────
    builder.add_edge("prosecutor", "chief_justice")
    builder.add_edge("defense", "chief_justice")
    builder.add_edge("techlead", "chief_justice")

    # ── End ───────────────────────────────────────────────────────────────────
    builder.add_edge("chief_justice", END)

    return builder.compile()


if __name__ == "__main__":
    g = build_graph()
    print("✅ Digital Courtroom graph compiled successfully")
    print("Nodes:", list(g.nodes))

    # Minimal end-to-end check (safe default: no PDF)
    base_state = {
        "repo_url": "https://github.com/nebiyou27/automaton-auditor.git",
        "pdf_path": "",  # leave blank unless you have a real local pdf path
        "rubric_path": "rubric/week2_rubric.json",
        "evidences": {},
        "opinions": [],
        "final_report": "",
    }

    print("\n--- Run: Detective + Judges + Chief Justice (no PDF) ---")
    out = g.invoke(base_state)

    buckets = {k: len(v) for k, v in out.get("evidences", {}).items()}
    print("Evidence buckets:", buckets)
    print("\nFinal report preview:\n")
    print(out.get("final_report", "")[:1200])