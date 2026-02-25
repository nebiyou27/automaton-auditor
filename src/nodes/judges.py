# src/nodes/judges.py
# ===================
# Judicial layer: three distinct personas produce structured JudicialOpinion outputs.
# Phase 4.5 (batched): one LLM call per judge returning one opinion per rubric criterion.
#
# Hardened for LOCAL Ollama (DeepSeek-R1):
# - No "rate limit sleep" logic (local model)
# - Strips <think>...</think> and other non-JSON wrappers
# - Enforces JSON-only output via strict prompt
# - Robust JSON array extraction + schema validation
# - Neutral fallback score=3 to avoid collapsing total score when LLM fails

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Tuple, Optional

from langchain_ollama import ChatOllama

from src.state import AgentState, JudicialOpinion


_llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "deepseek-r1:8b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    format="json",
    temperature=0.2,
)


# ------------------------------------------------------------------------------
# Evidence formatting (safe for Evidence objects OR dicts)
# ------------------------------------------------------------------------------

def _safe_get(ev, key: str, default=None):
    """Handle Evidence objects or dicts."""
    if isinstance(ev, dict):
        return ev.get(key, default)
    return getattr(ev, key, default)


def _format_evidence_for_judges(state: AgentState, max_items_per_bucket: int = 25) -> str:
    evidences = state.get("evidences", {}) or {}
    lines: List[str] = []
    for bucket, items in evidences.items():
        lines.append(f"\n=== EVIDENCE BUCKET: {str(bucket).upper()} (showing up to {max_items_per_bucket}) ===")
        if not isinstance(items, list):
            lines.append("- [WARN] bucket items are not a list (ignored).")
            continue

        for ev in items[:max_items_per_bucket]:
            found = bool(_safe_get(ev, "found", False))
            goal = str(_safe_get(ev, "goal", ""))
            location = str(_safe_get(ev, "location", ""))
            rationale = str(_safe_get(ev, "rationale", ""))
            content = _safe_get(ev, "content", None)

            status = "FOUND" if found else "NOT_FOUND"
            lines.append(f"- [{status}] goal={goal}")
            lines.append(f"  location={location}")
            lines.append(f"  rationale={rationale}")
            if content:
                lines.append(f"  content_preview={str(content)[:200]}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------
# Personas
# ------------------------------------------------------------------------------

PROSECUTOR_SYSTEM = """
You are the Prosecutor in a Digital Courtroom.
Philosophy: "Assume nothing. Trust no claims without evidence."
Be strict, skeptical, and cite exact evidence locations.
""".strip()

DEFENSE_SYSTEM = """
You are the Defense Attorney in a Digital Courtroom.
Philosophy: "Reward intent, progress, and reasonable engineering tradeoffs."
Be fair, highlight strengths, and cite exact evidence locations.
""".strip()

TECHLEAD_SYSTEM = """
You are the Senior Tech Lead in a Digital Courtroom.
Philosophy: "Pragmatic truth: Does it work and is it maintainable?"
Be the tiebreaker, be concrete, and cite exact evidence locations.
""".strip()


# ------------------------------------------------------------------------------
# Rubric loading
# ------------------------------------------------------------------------------

def _resolve_rubric_path(state: AgentState) -> str:
    rubric_path = (state.get("rubric_path") or "rubric/week2_rubric.json").strip()
    if os.path.isabs(rubric_path):
        return rubric_path
    return os.path.join(os.getcwd(), rubric_path)


def _load_rubric_ids_and_hints(state: AgentState) -> Tuple[List[str], Dict[str, str], Optional[str]]:
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
            name = str(d.get("name", "")).strip()
            instr = str(d.get("forensic_instruction", "")).strip()
            hints[cid] = f"name={name} | forensic_instruction={instr}"

    if not ids:
        return [], {}, "Rubric dimensions[] contains no valid id fields."

    return ids, hints, None


# ------------------------------------------------------------------------------
# Output parsing
# ------------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think_and_noise(raw: str) -> str:
    """
    DeepSeek-R1 often emits <think>...</think> or extra prose.
    Remove that before JSON extraction.
    """
    raw = (raw or "").strip()
    raw = _THINK_RE.sub("", raw).strip()

    # Also remove common markdown code fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()
    return raw


def _extract_json_array(raw: str) -> list:
    """
    Best-effort extraction of model output supporting:
    - direct JSON array
    - object wrapper: {"opinions": [...]}
    """
    raw = _strip_think_and_noise(raw)

    # 1) direct parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            opinions = parsed.get("opinions")
            if isinstance(opinions, list):
                return opinions
    except Exception:
        pass

    # 2) salvage: try object wrapper first
    start_obj = raw.find("{")
    end_obj = raw.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        candidate_obj = raw[start_obj : end_obj + 1]
        try:
            parsed_obj = json.loads(candidate_obj)
            if isinstance(parsed_obj, dict):
                opinions = parsed_obj.get("opinions")
                if isinstance(opinions, list):
                    return opinions
        except Exception:
            pass

    # 3) salvage: find first '[' and last ']'
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start : end + 1]
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            return parsed

    raise ValueError("No valid JSON array found in model output.")


# ------------------------------------------------------------------------------
# Fallback behavior (neutral score)
# ------------------------------------------------------------------------------

def _fallback_opinions(persona: str, criterion_ids: List[str], message: str, score: int = 3) -> List[JudicialOpinion]:
    """
    Always return valid opinions so the graph never crashes.
    Use neutral score=3 by default to avoid collapsing system score due to infra issues.
    """
    if not criterion_ids:
        criterion_ids = ["rubric_load_failed"]

    safe_score = max(1, min(5, int(score)))

    return [
        JudicialOpinion(
            judge=persona,  # type: ignore[arg-type]
            criterion_id=cid,
            score=safe_score,
            argument=message,
            cited_evidence=[],
        )
        for cid in criterion_ids
    ]


# ------------------------------------------------------------------------------
# Prompting
# ------------------------------------------------------------------------------

def _batched_prompt(
    persona: str,
    persona_system: str,
    ids: List[str],
    hints: Dict[str, str],
    evidence_text: str,
) -> str:
    rubric_lines = "\n".join([f"- {cid}: {hints.get(cid,'')}" for cid in ids])

    return f"""
{persona_system}

IMPORTANT OUTPUT RULES (MUST FOLLOW):
- Return ONLY a JSON array.
- Do NOT include any other text.
- Do NOT include markdown or code fences.
- Do NOT include <think> or explanations.
- If you cannot comply, return [].

You must return a JSON array of objects.
Each object MUST match this schema exactly:

{{
  "judge": "{persona}",
  "criterion_id": "<one of the provided rubric ids>",
  "score": <integer 1-5>,
  "argument": "<string>",
  "cited_evidence": ["<location1>", "<location2>"]
}}

Rules:
- Return EXACTLY ONE object per rubric criterion id.
- criterion_id MUST be one of the provided ids.
- score MUST be an integer 1-5.
- cited_evidence MUST be concrete locations (file paths, chunk_ids/pages, evidence locations).
- Do not include extra keys.

Rubric criterion ids + hints:
{rubric_lines}

EVIDENCE:
{evidence_text}
""".strip()


def _invoke(prompt: str) -> str:
    """
    Local Ollama call. No rate-limit retry.
    """
    return _llm.invoke(prompt).content


# ------------------------------------------------------------------------------
# Batched judge runner
# ------------------------------------------------------------------------------

def _run_batched_judge(persona: str, persona_system: str, state: AgentState) -> List[JudicialOpinion]:
    ids, hints, err = _load_rubric_ids_and_hints(state)
    if err:
        return _fallback_opinions(persona, ["rubric_load_failed"], err, score=3)

    evidence_text = _format_evidence_for_judges(state)
    prompt = _batched_prompt(persona, persona_system, ids, hints, evidence_text)

    try:
        raw = _invoke(prompt)
        arr = _extract_json_array(raw)
    except Exception as e:
        return _fallback_opinions(
            persona,
            ids,
            f"Judge model call failed. Neutral fallback opinions generated. ({e})",
            score=3,
        )

    # Build a map by criterion_id (more robust than zip)
    by_id: Dict[str, dict] = {}
    for item in arr:
        if not isinstance(item, dict):
            continue
        cid = item.get("criterion_id")
        if cid in ids and cid not in by_id:
            by_id[cid] = item

    opinions: List[JudicialOpinion] = []

    for cid in ids:
        item = by_id.get(cid)

        if not isinstance(item, dict):
            opinions.append(
                JudicialOpinion(
                    judge=persona,  # type: ignore[arg-type]
                    criterion_id=cid,
                    score=3,
                    argument="Missing/invalid opinion for criterion from model output; neutral fallback generated.",
                    cited_evidence=[],
                )
            )
            continue

        # Force correct judge + criterion_id
        item["judge"] = persona
        item["criterion_id"] = cid

        try:
            op = JudicialOpinion(**item)
        except Exception:
            op = JudicialOpinion(
                judge=persona,  # type: ignore[arg-type]
                criterion_id=cid,
                score=3,
                argument="Invalid model JSON schema; neutral fallback generated.",
                cited_evidence=[],
            )

        opinions.append(op)

    return opinions


# ------------------------------------------------------------------------------
# Node entrypoints
# ------------------------------------------------------------------------------

def prosecutor_judge(state: AgentState) -> Dict[str, List[JudicialOpinion]]:
    return {"opinions": _run_batched_judge("Prosecutor", PROSECUTOR_SYSTEM, state)}


def defense_judge(state: AgentState) -> Dict[str, List[JudicialOpinion]]:
    return {"opinions": _run_batched_judge("Defense", DEFENSE_SYSTEM, state)}


def techlead_judge(state: AgentState) -> Dict[str, List[JudicialOpinion]]:
    return {"opinions": _run_batched_judge("TechLead", TECHLEAD_SYSTEM, state)}
