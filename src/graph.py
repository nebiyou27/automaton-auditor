# src/graph.py
# ============
# Full graph orchestration:
#   Fan-Out detectives -> Fan-In evidence aggregator ->
#   Planner -> Executor -> Reflector -> (loop or judges) -> Chief Justice -> END

from __future__ import annotations

from pathlib import Path
from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.state import AgentState
from src.nodes.detectives import (
    doc_analyst,
    repo_investigator,
    vision_inspector_node,
)
from src.nodes.aggregator import evidence_aggregator
from src.nodes.error_handler import error_handler_node
from src.nodes.judge_repair import judge_repair_node
from src.nodes.planner import planner_node
from src.nodes.executor import executor_node
from src.nodes.reflector import judge_gate_node, reflector_node
from src.nodes.skip import skip_doc_analyst, skip_vision_inspector
from src.nodes.judges import defense_judge, judge_barrier_node, prosecutor_judge, techlead_judge
from src.nodes.justice import chief_justice


def _route_doc_analyst(state: AgentState) -> Literal["doc_analyst", "skip_doc_analyst"]:
    """
    Conditional routing:
    - If pdf_path is missing/blank, route to skip_doc_analyst (no-op),
      preserving fan-in semantics at evidence_aggregator.
    """
    pdf_path = (state.get("pdf_path") or "").strip()
    return "doc_analyst" if pdf_path else "skip_doc_analyst"


def _route_reflector(state: AgentState) -> Literal["planner", "judge_gate", "error_handler"]:
    error_type = (state.get("error_type") or "").strip()
    if error_type == "missing_evidence":
        return "error_handler"
    stop_decision = state.get("stop_decision")
    if stop_decision is None:
        return "planner"
    stop = bool(stop_decision.get("stop", False) if isinstance(stop_decision, dict) else getattr(stop_decision, "stop", False))
    return "judge_gate" if stop else "planner"


def _route_vision_inspector(state: AgentState) -> Literal["vision_inspector", "skip_vision_inspector"]:
    enable_vision = bool(state.get("enable_vision", False))
    pdf_path = (state.get("pdf_path") or "").strip()
    return "vision_inspector" if enable_vision and pdf_path else "skip_vision_inspector"


def _route_planner(state: AgentState) -> Literal["executor", "judge_gate", "error_handler"]:
    error_type = (state.get("error_type") or "").strip()
    if error_type == "missing_rubric":
        return "error_handler"
    stop_decision = state.get("stop_decision")
    if stop_decision is not None:
        stop = bool(stop_decision.get("stop", False) if isinstance(stop_decision, dict) else getattr(stop_decision, "stop", False))
        if stop:
            return "judge_gate"
    return "executor"


def _route_executor(state: AgentState) -> Literal["reflector", "error_handler"]:
    error_type = (state.get("error_type") or "").strip()
    if error_type in {"clone_failure", "executor_exception"}:
        return "error_handler"
    return "reflector"


def _route_judge_barrier(state: AgentState) -> Literal["chief_justice", "judge_repair"]:
    failures = state.get("judge_schema_failures", []) or []
    return "judge_repair" if len(failures) > 0 else "chief_justice"


def build_graph():
    """
    Digital Courtroom graph:
    1) Detectives run in parallel (fan-out)
    2) EvidenceAggregator synchronizes (fan-in)
    3) Planner proposes next 1-3 tool calls
    4) Executor runs selected tools and records results
    5) Reflector scores coverage/confidence + deterministic stop rules
    6) Judges run in parallel (fan-out) after stop conditions are met
    7) Chief Justice synthesizes deterministically (fan-in)
    """
    builder = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    # Detective layer
    builder.add_node("repo_investigator", repo_investigator)
    builder.add_node("doc_analyst", doc_analyst)
    builder.add_node("skip_doc_analyst", skip_doc_analyst)
    builder.add_node("skip_vision_inspector", skip_vision_inspector)
    # FIXED: C3+O1
    builder.add_node("vision_inspector", vision_inspector_node)

    # Fan-in evidence sync
    builder.add_node("evidence_aggregator", evidence_aggregator)
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("reflector", reflector_node)
    builder.add_node("judge_gate", judge_gate_node)
    builder.add_node("judge_barrier", judge_barrier_node)
    builder.add_node("judge_repair", judge_repair_node)
    builder.add_node("error_handler", error_handler_node)

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
    builder.add_conditional_edges(
        START,
        _route_vision_inspector,
        {
            "vision_inspector": "vision_inspector",
            "skip_vision_inspector": "skip_vision_inspector",
        },
    )

    # ── Detective fan-in barrier ──────────────────────────────────────────────
    builder.add_edge("repo_investigator", "evidence_aggregator")
    builder.add_edge("doc_analyst", "evidence_aggregator")
    builder.add_edge("skip_doc_analyst", "evidence_aggregator")
    builder.add_edge("vision_inspector", "evidence_aggregator")
    builder.add_edge("skip_vision_inspector", "evidence_aggregator")

    # ── Judicial fan-out ──────────────────────────────────────────────────────
    builder.add_edge("evidence_aggregator", "planner")
    builder.add_conditional_edges(
        "planner",
        _route_planner,
        {
            "executor": "executor",
            "judge_gate": "judge_gate",
            "error_handler": "error_handler",
        },
    )
    builder.add_conditional_edges(
        "executor",
        _route_executor,
        {
            "reflector": "reflector",
            "error_handler": "error_handler",
        },
    )
    builder.add_conditional_edges(
        "reflector",
        _route_reflector,
        {
            "planner": "planner",
            "judge_gate": "judge_gate",
            "error_handler": "error_handler",
        },
    )
    builder.add_edge("judge_gate", "prosecutor")
    builder.add_edge("prosecutor", "judge_barrier")
    builder.add_edge("judge_gate", "defense")
    builder.add_edge("defense", "judge_barrier")
    builder.add_edge("judge_gate", "techlead")
    builder.add_edge("techlead", "judge_barrier")
    builder.add_conditional_edges(
        "judge_barrier",
        _route_judge_barrier,
        {
            "chief_justice": "chief_justice",
            "judge_repair": "judge_repair",
        },
    )
    builder.add_edge("judge_repair", "chief_justice")

    # ── Judicial fan-in to Chief Justice ──────────────────────────────────────

    # ── End ───────────────────────────────────────────────────────────────────
    builder.add_edge("error_handler", END)
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
        "enable_vision": False,
        "repo_path": "",
        "rubric_path": "rubric/week2_rubric.json",
        "evidences": {},
        "opinions": [],
        "tool_runs": [],
        "iteration": 0,
        "max_iters": 6,
        "tool_budget": 20,
        "judge_schema_failures": [],
        "final_report": "",
    }

    print("\n--- Run: Detective + Judges + Chief Justice (no PDF) ---")
    out = g.invoke(base_state)

    buckets = {k: len(v) for k, v in out.get("evidences", {}).items()}
    print("Evidence buckets:", buckets)
    print("\nFinal report preview:\n")
    print(out.get("final_report", "")[:1200])
