from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Tuple

from src.rubric_ids import CANONICAL_DIMENSION_ID_SET, normalize_dimension_id
from src.state import AgentState, Evidence, ToolCall

PDF_ONLY_DIMENSIONS = {"theoretical_depth", "report_accuracy", "swarm_visual"}


def _to_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _load_rubric(state: AgentState) -> tuple[dict, str | None]:
    rubric_path = (state.get("rubric_path") or "rubric/week2_rubric.json").strip()
    if not os.path.isabs(rubric_path):
        rubric_path = os.path.join(os.getcwd(), rubric_path)
    if not os.path.exists(rubric_path):
        return {"dimensions": []}, f"Rubric file not found: {rubric_path}"
    try:
        with open(rubric_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"dimensions": []}, "Rubric JSON root must be an object."
        dims = data.get("dimensions", [])
        if not isinstance(dims, list):
            return {"dimensions": []}, "Rubric dimensions must be a list."
        return data, None
    except Exception as e:
        return {"dimensions": []}, f"Rubric JSON parse error: {e!r}"


def _flatten_evidence(state: AgentState) -> List[Tuple[str, object]]:
    evidences = state.get("evidences", {}) or {}
    rows: List[Tuple[str, object]] = []
    for bucket, items in evidences.items():
        if not isinstance(items, list):
            continue
        for item in items:
            rows.append((str(bucket), item))
    return rows


def _evidence_item_text(bucket: str, ev: object) -> str:
    if isinstance(ev, dict):
        parts = [
            bucket,
            _to_text(ev.get("goal")),
            _to_text(ev.get("location")),
            _to_text(ev.get("rationale")),
            _to_text(ev.get("content")),
        ]
    else:
        parts = [
            bucket,
            _to_text(getattr(ev, "goal", "")),
            _to_text(getattr(ev, "location", "")),
            _to_text(getattr(ev, "rationale", "")),
            _to_text(getattr(ev, "content", "")),
        ]
    return " ".join(parts).lower()


def _extract_paths(text: str) -> List[str]:
    path_like = re.findall(r"(?:src|rubric|audit|tests)/[A-Za-z0-9_./-]+", text)
    quoted = re.findall(r"[\"']([^\"']+)[\"']", text)
    out: List[str] = []
    for p in path_like + quoted:
        clean = p.strip()
        if clean and "/" in clean and len(clean) < 220:
            out.append(clean)
    return out


def _choose_tool_for_dimension(dimension: dict, seen_tools: set[str]) -> Tuple[str, Dict[str, object], str]:
    instruction = _to_text(dimension.get("forensic_instruction", "")).lower()
    dim_id = _to_text(dimension.get("id", ""))
    paths = _extract_paths(_to_text(dimension.get("forensic_instruction", "")))

    if "git log" in instruction or "commit" in instruction:
        return "git_log", {}, "Extract commit progression and timestamp evidence."

    if ("stategraph" in instruction or "add_edge" in instruction or "ast" in instruction) and "ast_scan" not in seen_tools:
        path = "src/graph.py"
        for p in paths:
            if p.endswith(".py"):
                path = p
                break
        return "ast_scan", {"path": path}, "Parse graph/state structure to verify orchestration claims."

    if "typeddict" in instruction or "basemodel" in instruction or "operator.add" in instruction:
        return "ast_scan", {"path": "src/state.py"}, "Validate typed state and reducer wiring."

    if "pdf" in instruction and ("image" in instruction or "diagram" in instruction):
        return (
            "vision_analyze",
            {"pdf_path": "$state.pdf_path"},
            "Analyze report visuals for architecture-diagram verification.",
        )

    if "pdf" in instruction:
        terms = []
        for key in ["Dialectical Synthesis", "Fan-In / Fan-Out", "Metacognition", "State Synchronization"]:
            if key.lower() in instruction:
                terms.append(key)
        query = terms[0] if terms else dim_id.replace("_", " ")
        return "pdf_query", {"query": query, "top_k": 3}, "Retrieve report text evidence for claimed concepts."

    if paths:
        return "ast_scan", {"path": paths[0]}, "Parse cited file structure for implementation verification."

    return "clone", {}, "Clone sandboxed target repo to enable additional forensic checks."


def _is_found(ev: object) -> bool:
    if isinstance(ev, dict):
        return bool(ev.get("found", False))
    return bool(getattr(ev, "found", False))


def _validation_evidence(dimension_id: str, found: bool, rationale: str, content: str) -> Evidence:
    return Evidence(
        dimension_id=dimension_id,
        goal=f"Planner dimension validation: {dimension_id or 'unsupported'}",
        found=found,
        location="rubric/week2_rubric.json",
        rationale=rationale,
        content=content[:700],
        confidence=1.0 if found else 0.6,
    )


def planner_node(state: AgentState) -> Dict[str, object]:
    rubric, rubric_err = _load_rubric(state)
    if rubric_err:
        return {
            "planned_tool_calls": [],
            "error_type": "missing_rubric",
            "error_message": rubric_err,
            "failed_node": "planner",
        }

    raw_dimensions = rubric.get("dimensions", [])
    dimensions = raw_dimensions if isinstance(raw_dimensions, list) else []

    evidence_rows = _flatten_evidence(state)
    ranked: List[Tuple[int, dict, str]] = []
    validation: List[Evidence] = []
    has_pdf = bool((state.get("pdf_path") or "").strip())

    for idx, dim in enumerate(dimensions):
        if not isinstance(dim, dict):
            continue
        raw_id = _to_text(dim.get("id", "")).strip()
        dim_id = normalize_dimension_id(raw_id)
        dim_name = _to_text(dim.get("name", "")).strip()
        if not raw_id:
            validation.append(
                _validation_evidence(
                    "",
                    False,
                    "Dimension has no id; cannot be handled by planner/toolset.",
                    f"dimension_index={idx}",
                )
            )
            continue
        if not dim_id or dim_id not in CANONICAL_DIMENSION_ID_SET:
            validation.append(
                _validation_evidence(
                    "",
                    False,
                    "Unsupported rubric dimension id; planner/toolset has no mapped handler.",
                    f"raw_id={raw_id}",
                )
            )
            continue
        if raw_id != dim_id:
            validation.append(
                _validation_evidence(
                    dim_id,
                    True,
                    "Legacy rubric id translated to canonical v3 id.",
                    f"raw_id={raw_id} -> canonical_id={dim_id}",
                )
            )
        else:
            validation.append(
                _validation_evidence(
                    dim_id,
                    True,
                    "Dimension id is canonical and handled by planner/toolset.",
                    f"id={dim_id}",
                )
            )

        dim = dict(dim)
        dim["id"] = dim_id
        target_artifact = _to_text(dim.get("target_artifact", "")).strip().lower()

        # Do not dispatch PDF-only work when no PDF is provided.
        if (not has_pdf) and (
            dim_id in PDF_ONLY_DIMENSIONS or target_artifact in {"pdf_report", "pdf_images"}
        ):
            validation.append(
                _validation_evidence(
                    dim_id,
                    True,
                    "Dimension marked not applicable for this run because pdf_path is empty.",
                    f"id={dim_id}, target_artifact={target_artifact or 'n/a'}",
                )
            )
            continue

        related = []
        for bucket, ev in evidence_rows:
            hay = _evidence_item_text(bucket, ev)
            if dim_id.lower() in hay or (dim_name and dim_name.lower() in hay):
                related.append(ev)

        if not related:
            risk = 100
            reason = f"{dim_id}: no evidence linked to this dimension."
        else:
            found_count = sum(1 for ev in related if _is_found(ev))
            missing_count = len(related) - found_count
            risk = missing_count * 20 + (0 if found_count else 40)
            reason = f"{dim_id}: found={found_count}, missing={missing_count}."

        ranked.append((risk, dim, reason))

    ranked.sort(key=lambda x: x[0], reverse=True)

    planned_calls: List[ToolCall] = []
    seen_tools: set[str] = set()
    for risk, dim, reason in ranked:
        if risk <= 0:
            continue
        if len(planned_calls) >= 3:
            break

        tool_name, args, expected = _choose_tool_for_dimension(dim, seen_tools)
        seen_tools.add(tool_name)
        priority = 5 if risk >= 100 else (4 if risk >= 60 else (3 if risk >= 30 else 2))

        planned_calls.append(
            ToolCall(
                dimension_id=_to_text(dim.get("id", "")),
                tool_name=tool_name,
                args=args,
                why=f"Highest remaining risk. {reason}",
                expected_evidence=expected,
                priority=priority,
            )
        )

    return {
        "planned_tool_calls": planned_calls[:3],
        "evidences": {"planner_validation": validation},
        "error_type": "",
        "error_message": "",
        "failed_node": "",
    }
