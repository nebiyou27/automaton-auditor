# src/nodes/skip.py
# =================
# Utility node used for conditional routing while preserving fan-in semantics.

from __future__ import annotations

from typing import Dict

from src.state import AgentState


def skip_doc_analyst(state: AgentState) -> Dict[str, object]:
    """
    No-op node.
    Exists only to preserve graph structure: START -> skip -> aggregator,
    so aggregator doesn't run early when doc_analyst is skipped.
    """
    return {}