from __future__ import annotations

import json
import os
from typing import Dict, List, Set, Tuple

from src.rubric_ids import CANONICAL_DIMENSION_ID_SET, normalize_dimension_id
from src.state import AgentState, Evidence, JudicialOpinion


def _load_rubric_ids(state: AgentState) -> List[str]:
    rubric_path = (state.get("rubric_path") or "rubric/week2_rubric.json").strip()
    if not os.path.isabs(rubric_path):
        rubric_path = os.path.join(os.getcwd(), rubric_path)
    if not os.path.exists(rubric_path):
        return []
    try:
        with open(rubric_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    dims = data.get("dimensions", []) if isinstance(data, dict) else []
    out: List[str] = []
    for d in dims if isinstance(dims, list) else []:
        if not isinstance(d, dict):
            continue
        cid = normalize_dimension_id(d.get("id"))
        if cid and cid in CANONICAL_DIMENSION_ID_SET and cid not in out:
            out.append(cid)
    return out


def _collect_locations(state: AgentState, criterion_id: str) -> List[str]:
    refs: List[str] = []
    evidences = state.get("evidences", {}) or {}
    kw = criterion_id.lower()
    for _, items in evidences.items():
        if not isinstance(items, list):
            continue
        for ev in items:
            location = (ev.get("location", "") if isinstance(ev, dict) else getattr(ev, "location", "")).strip()
            if not location:
                continue
            hay = " ".join(
                [
                    str(ev.get("dimension_id", "") if isinstance(ev, dict) else getattr(ev, "dimension_id", "")),
                    str(ev.get("goal", "") if isinstance(ev, dict) else getattr(ev, "goal", "")),
                    str(ev.get("rationale", "") if isinstance(ev, dict) else getattr(ev, "rationale", "")),
                    str(ev.get("content", "") if isinstance(ev, dict) else getattr(ev, "content", "")),
                    location,
                ]
            ).lower()
            if kw in hay and location not in refs:
                refs.append(location)
            if len(refs) >= 3:
                return refs
    return refs


def judge_repair_node(state: AgentState) -> Dict[str, object]:
    failures = list(state.get("judge_schema_failures", []) or [])
    existing = list(state.get("opinions", []) or [])
    rubric_ids = _load_rubric_ids(state)
    judges = ("Prosecutor", "Defense", "TechLead")

    seen: Set[Tuple[str, str]] = set()
    for op in existing:
        judge = op.get("judge") if isinstance(op, dict) else getattr(op, "judge", "")
        cid = op.get("criterion_id") if isinstance(op, dict) else getattr(op, "criterion_id", "")
        if judge and cid:
            seen.add((str(judge), str(cid)))

    added: List[JudicialOpinion] = []
    for cid in rubric_ids:
        refs = _collect_locations(state, cid)
        for judge in judges:
            if (judge, cid) in seen:
                continue
            added.append(
                JudicialOpinion(
                    judge=judge,  # type: ignore[arg-type]
                    criterion_id=cid,
                    score=3,
                    argument=(
                        "JudgeRepair fallback opinion inserted because structured output validation "
                        "failed after retries."
                    ),
                    cited_evidence=refs[:2],
                )
            )

    repair_evidence = Evidence(
        dimension_id="chief_justice_synthesis",
        goal="JudgeRepair synthesized fallback judicial opinions after schema failures",
        found=len(failures) > 0,
        location="src/nodes/judges.py",
        rationale=(
            f"Schema failures observed={len(failures)}; inserted_fallback_opinions={len(added)}."
        ),
        content=json.dumps({"failures": failures[:20]}),
        confidence=0.7 if failures else 1.0,
    )
    return {
        "opinions": added,
        "evidences": {"judge_repair": [repair_evidence]},
        "error_type": "",
        "error_message": "",
        "failed_node": "",
    }
