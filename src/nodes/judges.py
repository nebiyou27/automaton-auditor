# src/nodes/judges.py
# ===================
# Judicial layer: three distinct personas produce structured JudicialOpinion outputs.

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from langchain_ollama import ChatOllama

from src.state import AgentState, JudicialOpinion


def _get_structured_llm():
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "deepseek-r1:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        format="json",
        temperature=0.2,
    )
    return llm, llm.with_structured_output(JudicialOpinion)


def _safe_get(ev, key: str, default=None):
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


def _resolve_rubric_path(state: AgentState) -> str:
    rubric_path = (state.get("rubric_path") or "rubric/week2_rubric.json").strip()
    if os.path.isabs(rubric_path):
        return rubric_path
    return os.path.join(os.getcwd(), rubric_path)


def _load_rubric_ids_and_hints(
    state: AgentState,
) -> Tuple[List[str], Dict[str, str], Dict[str, Dict[str, str]], Optional[str]]:
    path = _resolve_rubric_path(state)
    if not os.path.exists(path):
        return [], {}, {}, f"Rubric file not found: {path}"

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return [], {}, {}, f"Rubric JSON parse error: {e!r}"

    dims = data.get("dimensions")
    if not isinstance(dims, list) or not dims:
        return [], {}, {}, "Rubric dimensions[] missing or empty."

    ids: List[str] = []
    hints: Dict[str, str] = {}
    judicial_logic_by_id: Dict[str, Dict[str, str]] = {}
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
            jl = d.get("judicial_logic", {})
            if isinstance(jl, dict):
                judicial_logic_by_id[cid] = {
                    "prosecutor": str(jl.get("prosecutor", "")).strip(),
                    "defense": str(jl.get("defense", "")).strip(),
                    "tech_lead": str(jl.get("tech_lead", "")).strip(),
                }
            else:
                judicial_logic_by_id[cid] = {
                    "prosecutor": "",
                    "defense": "",
                    "tech_lead": "",
                }

    if not ids:
        return [], {}, {}, "Rubric dimensions[] contains no valid id fields."

    return ids, hints, judicial_logic_by_id, None


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think_and_noise(raw: str) -> str:
    raw = (raw or "").strip()
    raw = _THINK_RE.sub("", raw).strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return raw


def _fallback_opinions(persona: str, criterion_ids: List[str], message: str, score: int = 3) -> List[JudicialOpinion]:
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


def _single_criterion_prompt(
    persona: str,
    persona_system: str,
    criterion_id: str,
    hint: str,
    judicial_instruction: str,
    evidence_text: str,
) -> str:
    return f"""
{persona_system}

Your specific instruction for this criterion:
{judicial_instruction}

Criterion: {criterion_id}
Context: {hint}

Evidence:
{evidence_text}
""".strip()


def _invoke_structured(prompt: str, persona: str, criterion_id: str) -> JudicialOpinion:
    _llm, _structured_llm = _get_structured_llm()
    try:
        op = _structured_llm.invoke(prompt)
        if isinstance(op, JudicialOpinion):
            op.judge = persona  # type: ignore[assignment]
            op.criterion_id = criterion_id
            return op
        if isinstance(op, dict):
            op["judge"] = persona
            op["criterion_id"] = criterion_id
            return JudicialOpinion(**op)
    except Exception:
        try:
            _strip_think_and_noise(str(_llm.invoke(prompt).content))
        except Exception:
            pass

    return JudicialOpinion(
        judge=persona,  # type: ignore[arg-type]
        criterion_id=criterion_id,
        score=3,
        argument="Invalid model JSON schema; neutral fallback generated.",
        cited_evidence=[],
    )


def _run_batched_judge(persona: str, persona_system: str, state: AgentState) -> List[JudicialOpinion]:
    ids, hints, judicial_logic_by_id, err = _load_rubric_ids_and_hints(state)
    if err:
        return _fallback_opinions(persona, ["rubric_load_failed"], err, score=3)

    evidence_text = _format_evidence_for_judges(state)
    persona_key = {
        "Prosecutor": "prosecutor",
        "Defense": "defense",
        "TechLead": "tech_lead",
    }.get(persona, "")
    opinions: List[JudicialOpinion] = []
    for cid in ids:
        judicial_instruction = ""
        if persona_key:
            judicial_instruction = judicial_logic_by_id.get(cid, {}).get(persona_key, "")
        prompt = _single_criterion_prompt(
            persona,
            persona_system,
            cid,
            hints.get(cid, ""),
            judicial_instruction,
            evidence_text,
        )
        opinions.append(_invoke_structured(prompt, persona, cid))

    return opinions


def prosecutor_judge(state: AgentState) -> Dict[str, List[JudicialOpinion]]:
    return {"opinions": _run_batched_judge("Prosecutor", PROSECUTOR_SYSTEM, state)}


def defense_judge(state: AgentState) -> Dict[str, List[JudicialOpinion]]:
    return {"opinions": _run_batched_judge("Defense", DEFENSE_SYSTEM, state)}


def techlead_judge(state: AgentState) -> Dict[str, List[JudicialOpinion]]:
    return {"opinions": _run_batched_judge("TechLead", TECHLEAD_SYSTEM, state)}
