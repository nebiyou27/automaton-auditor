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
    - Does NOT overwrite final_report (Chief Justice owns final_report)

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

    return {"evidences": normalized}