from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Tuple

from src.rubric_ids import CANONICAL_DIMENSION_ID_SET, normalize_dimension_id
from src.state import AgentState, DimensionReflection, StopDecision

PDF_ONLY_DIMENSIONS = {"theoretical_depth", "report_accuracy", "swarm_visual"}


def _to_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _load_dimensions(state: AgentState) -> List[dict]:
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
    out: List[dict] = []
    for d in dims if isinstance(dims, list) else []:
        if not isinstance(d, dict):
            continue
        nid = normalize_dimension_id(d.get("id"))
        if not nid or nid not in CANONICAL_DIMENSION_ID_SET:
            continue
        copy = dict(d)
        copy["id"] = nid
        out.append(copy)
    return out


def _is_found(ev: object) -> bool:
    if isinstance(ev, dict):
        return bool(ev.get("found", False))
    return bool(getattr(ev, "found", False))


def _get_confidence(ev: object) -> float:
    value = ev.get("confidence", 0.0) if isinstance(ev, dict) else getattr(ev, "confidence", 0.0)
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _evidence_text(ev: object, bucket: str) -> str:
    if isinstance(ev, dict):
        parts = [
            bucket,
            _to_text(ev.get("dimension_id")),
            _to_text(ev.get("goal")),
            _to_text(ev.get("location")),
            _to_text(ev.get("rationale")),
            _to_text(ev.get("content")),
        ]
    else:
        parts = [
            bucket,
            _to_text(getattr(ev, "dimension_id", "")),
            _to_text(getattr(ev, "goal", "")),
            _to_text(getattr(ev, "location", "")),
            _to_text(getattr(ev, "rationale", "")),
            _to_text(getattr(ev, "content", "")),
        ]
    return " ".join(parts).lower()


def _evidence_dimension_id(ev: object) -> str:
    if isinstance(ev, dict):
        return _to_text(ev.get("dimension_id", "")).strip()
    return _to_text(getattr(ev, "dimension_id", "")).strip()


def _tokenize_instruction(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9_]+", text.lower()) if len(w) >= 4]


def _missing_questions(dim: dict, related_text: str) -> List[str]:
    instruction = _to_text(dim.get("forensic_instruction", "")).strip()
    if not instruction:
        return ["No forensic instruction provided."]
    clauses = re.split(r"[.;]\s+|\n+", instruction)
    out: List[str] = []
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        tokens = _tokenize_instruction(clause)
        if not tokens:
            continue
        hits = sum(1 for t in tokens[:8] if t in related_text)
        if hits == 0:
            out.append(clause)
    return out[:6]


def _is_applicable(dim: dict, pdf_path: str) -> bool:
    dim_id = _to_text(dim.get("id", "")).strip()
    if dim_id in PDF_ONLY_DIMENSIONS and not pdf_path:
        return False
    enable_vision = bool(dim.get("_enable_vision", False))
    if dim_id == "swarm_visual" and (not enable_vision or not pdf_path):
        return False
    return True


def _collect_related_for_dimension(state: AgentState, dim: dict) -> Tuple[List[object], str]:
    dim_id = _to_text(dim.get("id", "")).strip()
    dim_name = _to_text(dim.get("name", "")).strip().lower()
    evidences = state.get("evidences", {}) or {}
    related: List[object] = []
    text_parts: List[str] = []
    for bucket, items in evidences.items():
        if not isinstance(items, list):
            continue
        for ev in items:
            ev_dim = _evidence_dimension_id(ev).lower()
            hay = _evidence_text(ev, str(bucket))
            if ev_dim == dim_id.lower() or dim_id.lower() in hay or (dim_name and dim_name in hay):
                related.append(ev)
                text_parts.append(hay)
    return related, " ".join(text_parts)


def _score_dimension(state: AgentState, dim: dict, pdf_path: str) -> DimensionReflection:
    dim_id = _to_text(dim.get("id", "")).strip()
    dim = dict(dim)
    dim["_enable_vision"] = bool(state.get("enable_vision", False))
    applicable = _is_applicable(dim, pdf_path)
    if not applicable:
        return DimensionReflection(
            dimension_id=dim_id,
            applicable=False,
            coverage=0.0,
            confidence=0.0,
            missing_questions=[],
        )

    related, related_text = _collect_related_for_dimension(state, dim)
    if not related:
        return DimensionReflection(
            dimension_id=dim_id,
            applicable=True,
            coverage=0.0,
            confidence=0.0,
            missing_questions=_missing_questions(dim, related_text),
        )

    found_count = sum(1 for ev in related if _is_found(ev))
    coverage = found_count / len(related)
    avg_conf = sum(_get_confidence(ev) for ev in related) / len(related)
    confidence = max(0.0, min(1.0, 0.6 * avg_conf + 0.4 * coverage))
    return DimensionReflection(
        dimension_id=dim_id,
        applicable=True,
        coverage=round(coverage, 4),
        confidence=round(confidence, 4),
        missing_questions=_missing_questions(dim, related_text),
    )


def reflector_node(state: AgentState) -> Dict[str, object]:
    dims = _load_dimensions(state)
    pdf_path = (state.get("pdf_path") or "").strip()
    reflections = [_score_dimension(state, d, pdf_path) for d in dims]

    applicable = [r for r in reflections if r.applicable]
    all_confident = bool(applicable) and all(r.confidence >= 0.85 for r in applicable)

    current_iter = int(state.get("iteration", 0) or 0)
    max_iters = int(state.get("max_iters", 6) or 6)
    next_iter = current_iter + 1

    tool_budget = int(state.get("tool_budget", 20) or 20)
    tools_used = len(state.get("tool_runs", []) or [])
    budget_exhausted = tools_used >= tool_budget
    max_iters_reached = next_iter >= max_iters

    stop = all_confident or budget_exhausted or max_iters_reached

    remaining_risks: List[str] = []
    for r in applicable:
        if r.confidence < 0.85:
            remaining_risks.append(
                f"{r.dimension_id}: confidence={r.confidence:.2f}, coverage={r.coverage:.2f}"
            )

    if all_confident:
        reason = "Stop: all applicable dimensions reached confidence >= 0.85."
    elif budget_exhausted:
        reason = f"Stop: tool budget exhausted ({tools_used}/{tool_budget})."
    elif max_iters_reached:
        reason = f"Stop: max iterations reached ({next_iter}/{max_iters})."
    else:
        reason = (
            f"Continue: {len(remaining_risks)} dimensions below confidence threshold; "
            f"iter={next_iter}/{max_iters}, budget={tools_used}/{tool_budget}."
        )

    # Explicit missing-evidence signal for graph conditional routing.
    severe_missing_evidence = bool(applicable) and all(r.coverage == 0.0 for r in applicable)
    error_type = "missing_evidence" if severe_missing_evidence and stop else ""
    error_message = (
        "No applicable rubric dimensions have supporting evidence coverage."
        if error_type
        else ""
    )

    return {
        "reflections": reflections,
        "iteration": next_iter,
        "stop_decision": StopDecision(
            stop=stop,
            reason=reason,
            remaining_risks=remaining_risks[:12],
        ),
        "error_type": error_type,
        "error_message": error_message,
        "failed_node": "reflector" if error_type else "",
    }


def judge_gate_node(state: AgentState) -> Dict[str, object]:
    """
    No-op gate used to fan out to judges only after reflector decides to stop iterating.
    """
    return {}
