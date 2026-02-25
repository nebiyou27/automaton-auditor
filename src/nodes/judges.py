# src/nodes/judges.py
# ===================
# Judicial layer: three distinct personas produce structured JudicialOpinion outputs.
# Phase 4.5 (batched): one LLM call per judge returning one opinion per rubric criterion.

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

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


PROSECUTOR_SYSTEM = """
You are the Prosecutor in a Digital Courtroom.
Philosophy: "Assume nothing. Trust no claims without evidence."
Be strict, skeptical, and cite exact evidence locations.
"""

DEFENSE_SYSTEM = """
You are the Defense Attorney in a Digital Courtroom.
Philosophy: "Reward intent, progress, and reasonable engineering tradeoffs."
Be fair, highlight strengths, and cite exact evidence locations.
"""

TECHLEAD_SYSTEM = """
You are the Senior Tech Lead in a Digital Courtroom.
Philosophy: "Pragmatic truth: Does it work and is it maintainable?"
Be the tiebreaker, be concrete, and cite exact evidence locations.
"""


def _resolve_rubric_path(state: AgentState) -> str:
    rubric_path = (state.get("rubric_path") or "rubric/week2_rubric.json").strip()
    if os.path.isabs(rubric_path):
        return rubric_path
    return os.path.join(os.getcwd(), rubric_path)


def _load_rubric_ids_and_hints(state: AgentState) -> Tuple[List[str], Dict[str, str], str | None]:
    """
    Returns (ids, hints_by_id, error).
    Fail-closed: if rubric invalid, return ([], {}, error_msg).
    """
    path = _resolve_rubric_path(state)
    if not os.path.exists(path):
        return [], {}, f"Rubric file not found: {path}"

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return [], {}, f"Rubric JSON parse error: {e!r}"

    dims = data.get("dimensions")
    if not isinstance(dims, list) or not dims:
        return [], {}, "Rubric dimensions[] missing or empty."

    ids: List[str] = []
    hints: Dict[str, str] = {}

    for d in dims:
        if not isinstance(d, dict):
            continue
        cid = d.get("id")
        if isinstance(cid, str) and cid.strip():
            cid = cid.strip()
            ids.append(cid)
            name = d.get("name", "")
            instr = d.get("forensic_instruction", "")
            # compact hint
            hints[cid] = f"name={name} | forensic_instruction={instr}"

    if not ids:
        return [], {}, "Rubric dimensions[] contains no valid id fields."

    return ids, hints, None


def _extract_json_array(raw: str) -> list:
    """
    Best-effort extraction of a JSON array from model output.
    """
    raw = (raw or "").strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # salvage: find first '[' and last ']'
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        parsed = json.loads(raw[start : end + 1])
        if isinstance(parsed, list):
            return parsed

    raise ValueError("No valid JSON array found in model output.")


def _fallback_opinions(persona: str, criterion_ids: List[str], message: str) -> List[JudicialOpinion]:
    # Always return something so the graph does not crash
    if not criterion_ids:
        criterion_ids = ["rubric_load_failed"]
    return [
        JudicialOpinion(
            judge=persona,  # type: ignore
            criterion_id=cid,
            score=1,
            argument=message,
            cited_evidence=[],
        )
        for cid in criterion_ids
    ]


def _batched_prompt(persona: str, persona_system: str, ids: List[str], hints: Dict[str, str], evidence_text: str) -> str:
    # Keep hints compact to avoid token bloat
    rubric_lines = "\n".join([f"- {cid}: {hints.get(cid,'')}" for cid in ids])

    return f"""
{persona_system}

You must return ONLY valid JSON: a JSON array of objects.
Each object MUST match this schema exactly:

{{
  "judge": "{persona}",
  "criterion_id": "<one of the provided rubric ids>",
  "score": <integer 1-5>,
  "argument": "<string>",
  "cited_evidence": ["<location1>", "<location2>"]
}}

Rules:
- Return ONE object per rubric criterion id (same order as provided is preferred).
- score must be an integer 1-5
- cited_evidence must be concrete locations (file paths, chunk_ids/pages, evidence locations)
- Do not include extra keys
- Do not return markdown, code fences, or commentary

Rubric criterion ids + hints:
{rubric_lines}

EVIDENCE:
{evidence_text}
""".strip()


def _run_batched_judge(persona: str, persona_system: str, state: AgentState) -> List[JudicialOpinion]:
    ids, hints, err = _load_rubric_ids_and_hints(state)
    if err:
        return _fallback_opinions(persona, ["rubric_load_failed"], err)

    evidence_text = _format_evidence_for_judges(state)

    try:
        raw = _llm.invoke(_batched_prompt(persona, persona_system, ids, hints, evidence_text)).content
        arr = _extract_json_array(raw)
    except Exception as e:
        # Includes 429 quota errors and parsing errors
        return _fallback_opinions(persona, ids, f"Judge model call failed or was rate-limited; fallback opinions generated. ({e})")

    opinions: List[JudicialOpinion] = []
    seen = set()

    for cid, item in zip(ids, arr):
        if not isinstance(item, dict):
            opinions.append(JudicialOpinion(judge=persona, criterion_id=cid, score=1,
                                            argument="Invalid JSON item type; fallback.", cited_evidence=[]))  # type: ignore
            continue

        # force correct labels + criterion_id safety
        item["judge"] = persona
        item["criterion_id"] = item.get("criterion_id") if item.get("criterion_id") in ids else cid

        try:
            op = JudicialOpinion(**item)
        except Exception:
            op = JudicialOpinion(judge=persona, criterion_id=item["criterion_id"], score=1,
                                 argument="Invalid model JSON schema; fallback opinion generated.", cited_evidence=[])  # type: ignore

        opinions.append(op)
        seen.add(op.criterion_id)

    # Ensure we have one opinion per criterion (fill missing)
    for cid in ids:
        if cid not in seen:
            opinions.append(JudicialOpinion(
                judge=persona,  # type: ignore
                criterion_id=cid,
                score=1,
                argument="Missing opinion for criterion from model output; fallback generated.",
                cited_evidence=[],
            ))

    return opinions


def prosecutor_judge(state: AgentState) -> Dict[str, List[JudicialOpinion]]:
    return {"opinions": _run_batched_judge("Prosecutor", PROSECUTOR_SYSTEM, state)}


def defense_judge(state: AgentState) -> Dict[str, List[JudicialOpinion]]:
    return {"opinions": _run_batched_judge("Defense", DEFENSE_SYSTEM, state)}


def techlead_judge(state: AgentState) -> Dict[str, List[JudicialOpinion]]:
    return {"opinions": _run_batched_judge("TechLead", TECHLEAD_SYSTEM, state)}