# src/nodes/aggregator.py
# =======================
# Fan-In synchronization node that consolidates evidence from all detectives.

from __future__ import annotations

from typing import Dict, List

from src.state import AgentState, Evidence


def evidence_aggregator(state: AgentState) -> Dict[str, object]:
    """
    EvidenceAggregator (Fan-In):
    - Does not generate new evidence
    - Ensures the state has a normalized evidences dict
    - Produces a simple summary useful for debugging or downstream nodes

    Returns a partial state update (safe for LangGraph merge).
    """
    evidences = state.get("evidences", {}) or {}

    # Normalize structure: ensure each bucket is a list
    normalized: Dict[str, List[Evidence]] = {}
    for bucket, items in evidences.items():
        if isinstance(items, list):
            normalized[bucket] = items
        else:
            # If someone accidentally wrote wrong shape, keep it but don't crash
            normalized[bucket] = []

    # (Optional) store a short summary in final_report for interim runs
    counts = {k: len(v) for k, v in normalized.items()}
    summary = f"Evidence aggregated. Buckets={list(normalized.keys())} Counts={counts}"

    return {
        "evidences": normalized,
        "final_report": summary,
    }