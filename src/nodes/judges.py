# src/nodes/judges.py
# ===================
# Judicial layer: three distinct personas produce structured JudicialOpinion outputs.
# Uses Gemini via langchain-google-genai and enforces strict JSON (no parser import needed).

from __future__ import annotations

import json
from typing import Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI

from src.state import AgentState, JudicialOpinion


# Use Gemini for all agents (as per your decision)
_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
)


def _format_evidence_for_judges(state: AgentState, max_items_per_bucket: int = 25) -> str:
    """
    Compact evidence formatting (facts only) for the judges.
    Keeps prompts readable and prevents context bloat.
    """
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
    """
    Best-effort extraction of a JSON object from model output.
    Ensures robustness if the model adds extra text.
    """
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise ValueError("No valid JSON object found in model output.")


def _run_judge(persona: str, persona_system: str, state: AgentState, criterion_id: str) -> JudicialOpinion:
    evidence_text = _format_evidence_for_judges(state)

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
- score must be an integer 1–5
- cited_evidence entries must be concrete locations (file paths, chunk_ids/pages, or evidence locations)
- argument must directly reference the evidence
- do NOT include extra keys

EVIDENCE:
{evidence_text}
""".strip()

    raw = _llm.invoke(prompt).content

    try:
        data = _extract_json_object(raw)
    except Exception:
        # Hard fallback: keep system running deterministically
        data = {
            "judge": persona,
            "criterion_id": criterion_id,
            "score": 1,
            "argument": "Model output could not be parsed into valid JSON.",
            "cited_evidence": [],
        }

    # Force correct labels (prevents model cheating)
    data["judge"] = persona
    data["criterion_id"] = criterion_id

    # Validate into Pydantic model (strict schema)
    return JudicialOpinion(**data)


def prosecutor_judge(state: AgentState) -> Dict[str, List[JudicialOpinion]]:
    """
    Prosecutor node: strict critic.
    """
    criterion_id = "detective_phase"
    op = _run_judge("Prosecutor", PROSECUTOR_SYSTEM, state, criterion_id)
    return {"opinions": [op]}


def defense_judge(state: AgentState) -> Dict[str, List[JudicialOpinion]]:
    """
    Defense node: generous interpreter.
    """
    criterion_id = "detective_phase"
    op = _run_judge("Defense", DEFENSE_SYSTEM, state, criterion_id)
    return {"opinions": [op]}


def techlead_judge(state: AgentState) -> Dict[str, List[JudicialOpinion]]:
    """
    Tech Lead node: pragmatic tiebreaker.
    """
    criterion_id = "detective_phase"
    op = _run_judge("TechLead", TECHLEAD_SYSTEM, state, criterion_id)
    return {"opinions": [op]}