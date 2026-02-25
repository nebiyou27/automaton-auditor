# src/nodes/judges.py
# ===================
# Judicial layer: three distinct personas produce structured JudicialOpinion outputs.
# Phase 4.5 hardened:
# - deterministic rubric path default
# - fail-closed rubric loading (no silent fake criterion IDs)
# - safe fallback on JSON parse OR Pydantic validation failures

from __future__ import annotations

import json
import os
from typing import Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI

from src.state import AgentState, JudicialOpinion


_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
)


def _format_evidence_for_judges(state: AgentState, max_items_per_bucket: int = 25) -> str:
    evidences = state.get("evidences", {}) or {}
    lines: List[str] = []
    for bucket, items in evidences.items():
        lines.append(f"\n=== EVIDENCE BUCKET: {bucket.upper()} (showing up to {max_items_per_bucket}) ===")
        for ev in items[:max_items_per_bucket]:
            status = "FOUND" if ev.found else "NOT_FOUND"
            lines.append(f"- [{status}] goal={ev.goal}")
            lines.append(f"  location={ev.location}")
            lines.append(f"  rationale={ev.rationale}")
            if ev.content:
                lines.append(f"  content_preview={ev.content[:200]}")
    return "\n".join(lines)


# --- Distinct persona prompts (dialectical bench) ---

PROSECUTOR_SYSTEM = """
You are the Prosecutor in a Digital Courtroom.
Philosophy: "Assume nothing. Trust no claims without evidence."

Behavior:
- Be strict and skeptical.
- Penalize missing artifacts, weak traceability, and unsafe patterns.
- If you see contradictions (README claims vs files missing), call them out.
- Cite exact locations from the evidence (paths, chunk_ids, pages).
"""

DEFENSE_SYSTEM = """
You are the Defense Attorney in a Digital Courtroom.
Philosophy: "Reward intent, progress, and reasonable engineering tradeoffs."

Behavior:
- Look for what exists and what works.
- Give credit for partial implementations if the architecture is coherent.
- Highlight strengths (typing, reducers, sandboxing, RAG-lite PDF tools).
- Cite exact locations from the evidence (paths, chunk_ids, pages).
"""

TECHLEAD_SYSTEM = """
You are the Senior Tech Lead in a Digital Courtroom.
Philosophy: "Pragmatic truth: Does it work and is it maintainable?"

Behavior:
- Be the tiebreaker between Prosecutor and Defense.
- Focus on correctness, safety, reproducibility, and maintainability.
- Provide concrete remediation suggestions.
- Cite exact locations from the evidence (paths, chunk_ids, pages).
"""


def _extract_json_object(raw: str) -> dict:
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise ValueError("No valid JSON object found in model output.")


def _resolve_rubric_path(state: AgentState) -> str:
    """
    Deterministic rubric source:
    - Use state["rubric_path"] if set, otherwise default to rubric/week2_rubric.json.
    - Resolve relative to current working directory.
    """
    rubric_path = (state.get("rubric_path") or "rubric/week2_rubric.json").strip()
    if os.path.isabs(rubric_path):
        return rubric_path
    return os.path.join(os.getcwd(), rubric_path)


def _load_rubric_dimension_ids_or_error(state: AgentState) -> tuple[List[str], str | None]:
    """
    Fail-closed rubric loading:
    - If rubric file missing/invalid/empty -> return ([], error_message)
    - If valid -> return ([dimension_ids], None)
    """
    path = _resolve_rubric_path(state)
    if not os.path.exists(path):
        return [], f"Rubric file not found at: {path}"

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return [], f"Rubric file could not be parsed as JSON: {e!r}"

    dims = data.get("dimensions")
    if not isinstance(dims, list) or not dims:
        return [], "Rubric JSON has no dimensions[] list (or it is empty)."

    ids: List[str] = []
    for d in dims:
        if isinstance(d, dict) and isinstance(d.get("id"), str) and d["id"].strip():
            ids.append(d["id"].strip())

    if not ids:
        return [], "Rubric dimensions[] exist but contain no valid id fields."

    return ids, None


def _rubric_hint_for_criterion(state: AgentState, criterion_id: str) -> str | None:
    """
    Optional compact rubric hint for a criterion (kept short to avoid prompt bloat).
    If rubric cannot be read, return None (judges can still reason from evidence).
    """
    path = _resolve_rubric_path(state)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for d in data.get("dimensions", []):
            if isinstance(d, dict) and d.get("id") == criterion_id:
                name = d.get("name", "")
                instr = d.get("forensic_instruction", "")
                return f"- name: {name}\n- forensic_instruction: {instr}"
    except Exception:
        return None
    return None


def _fallback_opinion(persona: str, criterion_id: str, message: str) -> JudicialOpinion:
    return JudicialOpinion(
        judge=persona,  # type: ignore
        criterion_id=criterion_id,
        score=1,
        argument=message,
        cited_evidence=[],
    )


def _run_one_opinion(
    *,
    persona: str,
    persona_system: str,
    criterion_id: str,
    evidence_text: str,
    rubric_hint: str | None = None,
) -> JudicialOpinion:
    rubric_hint_text = f"\nRubric guidance:\n{rubric_hint}\n" if rubric_hint else ""

    prompt = f"""
{persona_system}

You must return ONLY valid JSON (no markdown, no commentary).
The JSON must match this schema exactly:

{{
  "judge": "{persona}",
  "criterion_id": "{criterion_id}",
  "score": <integer 1-5>,
  "argument": "<string>",
  "cited_evidence": ["<location1>", "<location2>"]
}}

Rules:
- score must be an integer 1-5
- cited_evidence entries must be concrete locations (file paths, chunk_ids/pages, evidence locations)
- argument must reference the evidence AND the criterion
- do NOT include extra keys
{rubric_hint_text}
EVIDENCE:
{evidence_text}
""".strip()

    raw = _llm.invoke(prompt).content

    # Parse JSON
    try:
        data = _extract_json_object(raw)
    except Exception:
        return _fallback_opinion(persona, criterion_id, "Model output could not be parsed into valid JSON; fallback opinion generated.")

    # Force labels (prevents cheating)
    data["judge"] = persona
    data["criterion_id"] = criterion_id

    # Validate with Pydantic (never crash)
    try:
        return JudicialOpinion(**data)
    except Exception:
        return _fallback_opinion(persona, criterion_id, "Invalid model JSON schema; fallback opinion generated.")


def _run_judge_for_all_criteria(persona: str, persona_system: str, state: AgentState) -> List[JudicialOpinion]:
    evidence_text = _format_evidence_for_judges(state)

    criterion_ids, err = _load_rubric_dimension_ids_or_error(state)
    if err:
        # Fail-closed: produce a single opinion indicating rubric load failure.
        return [_fallback_opinion(persona, "rubric_load_failed", err)]

    opinions: List[JudicialOpinion] = []
    for cid in criterion_ids:
        hint = _rubric_hint_for_criterion(state, cid)
        opinions.append(
            _run_one_opinion(
                persona=persona,
                persona_system=persona_system,
                criterion_id=cid,
                evidence_text=evidence_text,
                rubric_hint=hint,
            )
        )
    return opinions


def prosecutor_judge(state: AgentState) -> Dict[str, List[JudicialOpinion]]:
    return {"opinions": _run_judge_for_all_criteria("Prosecutor", PROSECUTOR_SYSTEM, state)}


def defense_judge(state: AgentState) -> Dict[str, List[JudicialOpinion]]:
    return {"opinions": _run_judge_for_all_criteria("Defense", DEFENSE_SYSTEM, state)}


def techlead_judge(state: AgentState) -> Dict[str, List[JudicialOpinion]]:
    return {"opinions": _run_judge_for_all_criteria("TechLead", TECHLEAD_SYSTEM, state)}