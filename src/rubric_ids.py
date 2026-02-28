from __future__ import annotations

from typing import Dict, List, Optional

CANONICAL_DIMENSION_IDS: List[str] = [
    "git_forensic_analysis",
    "state_management_rigor",
    "graph_orchestration",
    "safe_tool_engineering",
    "structured_output_enforcement",
    "judicial_nuance",
    "chief_justice_synthesis",
    "theoretical_depth",
    "report_accuracy",
    "swarm_visual",
]

CANONICAL_DIMENSION_ID_SET = set(CANONICAL_DIMENSION_IDS)

# Legacy -> v3 translation map
LEGACY_ID_ALIASES: Dict[str, str] = {
    "typed_state_definitions": "state_management_rigor",
    "forensic_tool_engineering": "safe_tool_engineering",
    "partial_graph_orchestration": "graph_orchestration",
    "judicial_nuance_dialectics": "judicial_nuance",
}


def normalize_dimension_id(raw_id: object) -> Optional[str]:
    if not isinstance(raw_id, str):
        return None
    cid = raw_id.strip()
    if not cid:
        return None
    if cid in CANONICAL_DIMENSION_ID_SET:
        return cid
    return LEGACY_ID_ALIASES.get(cid)
