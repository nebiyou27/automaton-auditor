# src/nodes/judges.py
# ===================
# Judicial layer: three distinct personas produce structured JudicialOpinion outputs.

from __future__ import annotations

import json
import os
import re
from json import JSONDecodeError
from typing import Any, Dict, List, Optional, Tuple

from langchain_ollama import ChatOllama

from src.rubric_ids import CANONICAL_DIMENSION_ID_SET, normalize_dimension_id
from src.state import AgentState, JudicialOpinion


def _get_llm() -> ChatOllama:
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "deepseek-r1:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        format="json",
        temperature=0.2,
    )


def _normalize_judge(value: object, fallback: str) -> str:
    text = str(value or "").strip().lower()
    mapping = {
        "prosecutor": "Prosecutor",
        "defense": "Defense",
        "defence": "Defense",
        "techlead": "TechLead",
        "tech_lead": "TechLead",
        "tech lead": "TechLead",
    }
    return mapping.get(text, fallback)


def _coerce_score(value: object, default: int = 3) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(5, score))


def _coerce_cited_evidence(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        v = value.strip()
        return [v] if v else []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    s = str(value).strip()
    return [s] if s else []


def _safe_get(ev, key: str, default=None):
    if isinstance(ev, dict):
        return ev.get(key, default)
    return getattr(ev, key, default)


def _extract_evidence_locations(state: AgentState, criterion_id: str, max_refs: int = 10) -> List[str]:
    evidences = state.get("evidences", {}) or {}
    keywords = [k for k in re.split(r"[_\W]+", criterion_id.lower()) if k]
    refs: List[str] = []
    for _, items in evidences.items():
        if not isinstance(items, list):
            continue
        for ev in items:
            location = str(_safe_get(ev, "location", "")).strip()
            if not location:
                continue
            hay = " ".join(
                [
                    str(_safe_get(ev, "dimension_id", "")),
                    str(_safe_get(ev, "goal", "")),
                    str(_safe_get(ev, "rationale", "")),
                    str(_safe_get(ev, "content", "")),
                    location,
                ]
            ).lower()
            if keywords and not any(k in hay for k in keywords):
                continue
            if location not in refs:
                refs.append(location)
            if len(refs) >= max_refs:
                return refs

    if refs:
        return refs
    # Fallback: allow any known locations if criterion-specific match is empty.
    for _, items in evidences.items():
        if not isinstance(items, list):
            continue
        for ev in items:
            location = str(_safe_get(ev, "location", "")).strip()
            if location and location not in refs:
                refs.append(location)
            if len(refs) >= max_refs:
                return refs
    return refs


def _format_evidence_for_judges(
    state: AgentState,
    criterion_id: str = "",
    max_items_per_bucket: int = 25,
) -> str:
    evidences = state.get("evidences", {}) or {}
    keywords = [k for k in re.split(r"[_\W]+", criterion_id.lower()) if k]
    lines: List[str] = []
    for bucket, items in evidences.items():
        lines.append(f"\n=== EVIDENCE BUCKET: {str(bucket).upper()} (showing up to {max_items_per_bucket}) ===")
        if not isinstance(items, list):
            lines.append("- [WARN] bucket items are not a list (ignored).")
            continue

        selected = items
        if keywords:
            matched = []
            for ev in items:
                hay = " ".join(
                    [
                        str(_safe_get(ev, "goal", "")),
                        str(_safe_get(ev, "location", "")),
                        str(_safe_get(ev, "rationale", "")),
                        str(_safe_get(ev, "content", "")),
                    ]
                ).lower()
                if any(k in hay for k in keywords):
                    matched.append(ev)
            if matched:
                selected = matched

        for ev in selected[:max_items_per_bucket]:
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
Philosophy: Assume nothing. Trust no claims without evidence.

SCORING GUIDE - you MUST follow this:
- Score 1: Evidence is missing or contradicts the requirement
- Score 2: Evidence is weak or partially meets requirement
- Score 3: Evidence meets requirement but has minor gaps
- Score 4: Evidence clearly meets requirement
- Score 5: Evidence exceeds requirement with exceptional quality

Be strict but score accurately based on evidence.
Be strict, skeptical, and cite exact evidence locations.
""".strip()

DEFENSE_SYSTEM = """
You are the Defense Attorney in a Digital Courtroom.
Philosophy: "Reward intent, progress, and reasonable engineering tradeoffs."

SCORING GUIDE - you MUST follow this:
- Score 1: Evidence is missing or contradicts the requirement
- Score 2: Evidence is weak or partially meets requirement
- Score 3: Evidence meets requirement but has minor gaps
- Score 4: Evidence clearly meets requirement
- Score 5: Evidence exceeds requirement with exceptional quality

Be strict but score accurately based on evidence.
Be fair, highlight strengths, and cite exact evidence locations.
""".strip()

TECHLEAD_SYSTEM = """
You are the Senior Tech Lead in a Digital Courtroom.
Philosophy: "Pragmatic truth: Does it work and is it maintainable?"

SCORING GUIDE - you MUST follow this:
- Score 1: Evidence is missing or contradicts the requirement
- Score 2: Evidence is weak or partially meets requirement
- Score 3: Evidence meets requirement but has minor gaps
- Score 4: Evidence clearly meets requirement
- Score 5: Evidence exceeds requirement with exceptional quality

Be strict but score accurately based on evidence.
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
        cid = normalize_dimension_id(d.get("id"))
        if isinstance(cid, str) and cid.strip() and cid in CANONICAL_DIMENSION_ID_SET:
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


def _is_positive_argument(text: str) -> bool:
    lowered = (text or "").lower()
    positives = [
        "meets requirement",
        "clearly meets",
        "all requirements are met",
        "implemented",
        "present",
        "verified",
        "strong",
    ]
    return any(p in lowered for p in positives)


def _is_negative_argument(text: str) -> bool:
    lowered = (text or "").lower()
    negatives = [
        "missing",
        "not found",
        "does not meet",
        "failed",
        "cannot verify",
        "incomplete",
        "contradicts",
        "absent",
    ]
    return any(n in lowered for n in negatives)


def _is_score_argument_inconsistent(score: int, argument: str) -> bool:
    if _is_positive_argument(argument) and score <= 2:
        return True
    if _is_negative_argument(argument) and score >= 4:
        return True
    return False


def _coerce_opinion(op: object, persona: str, criterion_id: str) -> Optional[JudicialOpinion]:
    if isinstance(op, JudicialOpinion):
        op.judge = persona  # type: ignore[assignment]
        op.criterion_id = criterion_id
        return op
    if isinstance(op, dict):
        payload: Dict[str, Any] = dict(op)
        payload["judge"] = _normalize_judge(payload.get("judge"), persona)
        payload["criterion_id"] = criterion_id
        payload["score"] = _coerce_score(payload.get("score"))
        payload["argument"] = str(payload.get("argument", "")).strip()
        payload["cited_evidence"] = _coerce_cited_evidence(payload.get("cited_evidence"))
        if not payload["argument"]:
            return None
        return JudicialOpinion(**payload)
    return None


def _coerce_opinion_from_raw_text(raw_text: str, persona: str, criterion_id: str) -> Optional[JudicialOpinion]:
    cleaned = _strip_think_and_noise(raw_text)
    if not cleaned:
        return None

    try:
        parsed = json.loads(cleaned)
    except JSONDecodeError:
        # Try to recover JSON object if the model adds extra wrapper text.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except JSONDecodeError:
            return None

    if isinstance(parsed, dict):
        # Common wrapper shapes produced by local models.
        for key in ("opinion", "result", "output", "data", "judicial_opinion"):
            candidate = parsed.get(key)
            if isinstance(candidate, dict):
                parsed = candidate
                break

        if isinstance(parsed.get("reasoning"), str) and not parsed.get("argument"):
            parsed["argument"] = parsed["reasoning"]
        if isinstance(parsed.get("justification"), str) and not parsed.get("argument"):
            parsed["argument"] = parsed["justification"]
        if isinstance(parsed.get("evidence"), list) and not parsed.get("cited_evidence"):
            parsed["cited_evidence"] = parsed["evidence"]
        if isinstance(parsed.get("citations"), list) and not parsed.get("cited_evidence"):
            parsed["cited_evidence"] = parsed["citations"]

    return _coerce_opinion(parsed, persona, criterion_id)


def _fallback_opinions(persona: str, criterion_ids: List[str], message: str, score: int = 3) -> List[JudicialOpinion]:
    if not criterion_ids:
        return []

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
    allowed_refs: List[str],
) -> str:
    refs_text = ", ".join(allowed_refs[:8]) if allowed_refs else "(none available)"
    return f"""
{persona_system}

You must return EXACTLY one JSON object with this shape:
{{
  "judge": "{persona}",
  "criterion_id": "{criterion_id}",
  "score": <integer 1-5>,
  "argument": "<2-6 sentences grounded in evidence>",
  "cited_evidence": ["<file_or_evidence_ref>", "..."]
}}
Rules:
- Return JSON only (no markdown, no commentary).
- Use only the criterion and evidence shown below.
- Keep score aligned with argument sentiment and scoring guide.
- If evidence is missing, set lower score and explain that clearly.
- `cited_evidence` MUST contain concrete evidence locations from this allowlist: {refs_text}

Your specific instruction for this criterion:
{judicial_instruction}

Criterion: {criterion_id}
Context: {hint}

Evidence:
{evidence_text}
""".strip()


def _coerce_structured_output(payload: object, persona: str, criterion_id: str) -> Optional[JudicialOpinion]:
    if isinstance(payload, JudicialOpinion):
        payload.judge = persona  # type: ignore[assignment]
        payload.criterion_id = criterion_id
        return payload
    if isinstance(payload, dict):
        data = dict(payload)
        data["judge"] = _normalize_judge(data.get("judge"), persona)
        data["criterion_id"] = criterion_id
        data["score"] = _coerce_score(data.get("score"))
        data["argument"] = str(data.get("argument", "")).strip()
        data["cited_evidence"] = _coerce_cited_evidence(data.get("cited_evidence"))
        if not data["argument"]:
            return None
        return JudicialOpinion(**data)
    return None


def _normalize_citations(citations: List[str], allowed_refs: List[str]) -> List[str]:
    if not allowed_refs:
        return citations[:4]
    normalized: List[str] = []
    for c in citations:
        cand = c.strip()
        if not cand:
            continue
        # Accept exact match or containment overlap with known references.
        match = next((r for r in allowed_refs if cand == r or cand in r or r in cand), None)
        if match and match not in normalized:
            normalized.append(match)
    if not normalized:
        normalized = allowed_refs[:2]
    return normalized[:4]


def _invoke_structured(
    prompt: str,
    persona: str,
    criterion_id: str,
    allowed_refs: List[str],
    retries: int = 2,
) -> Tuple[Optional[JudicialOpinion], Optional[str]]:
    _llm = _get_llm().with_structured_output(JudicialOpinion)
    last_error: Optional[str] = None

    for attempt in range(retries + 1):
        try:
            op_raw = _llm.invoke(prompt)
            op = _coerce_structured_output(op_raw, persona, criterion_id)
            if op is None:
                raise ValueError("Schema validation failed: empty/invalid structured payload.")
            op.cited_evidence = _normalize_citations(op.cited_evidence, allowed_refs)
            if _is_score_argument_inconsistent(op.score, op.argument):
                raise ValueError("Schema validation failed: score/argument inconsistency.")
            return op, None
        except Exception as e:
            last_error = str(e)
            if attempt < retries:
                prompt = (
                    f"{prompt}\n\n"
                    "Validation retry: your previous output failed schema or consistency checks. "
                    "Return valid JudicialOpinion JSON matching all constraints."
                )
                continue
            break
    return None, last_error or "Unknown schema validation failure."


def _run_batched_judge(persona: str, persona_system: str, state: AgentState) -> List[JudicialOpinion]:
    ids, hints, judicial_logic_by_id, err = _load_rubric_ids_and_hints(state)
    if err:
        return []

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
        evidence_text = _format_evidence_for_judges(state, criterion_id=cid)
        allowed_refs = _extract_evidence_locations(state, cid)
        prompt = _single_criterion_prompt(
            persona,
            persona_system,
            cid,
            hints.get(cid, ""),
            judicial_instruction,
            evidence_text,
            allowed_refs,
        )
        op, _ = _invoke_structured(prompt, persona, cid, allowed_refs=allowed_refs, retries=2)
        if op is not None:
            opinions.append(op)

    return opinions


def _run_batched_judge_with_errors(
    persona: str,
    persona_system: str,
    state: AgentState,
) -> Tuple[List[JudicialOpinion], List[str]]:
    ids, hints, judicial_logic_by_id, err = _load_rubric_ids_and_hints(state)
    if err:
        return [], [f"{persona}|rubric_load|{err}"]

    persona_key = {
        "Prosecutor": "prosecutor",
        "Defense": "defense",
        "TechLead": "tech_lead",
    }.get(persona, "")

    opinions: List[JudicialOpinion] = []
    errors: List[str] = []
    for cid in ids:
        judicial_instruction = ""
        if persona_key:
            judicial_instruction = judicial_logic_by_id.get(cid, {}).get(persona_key, "")
        evidence_text = _format_evidence_for_judges(state, criterion_id=cid)
        allowed_refs = _extract_evidence_locations(state, cid)
        prompt = _single_criterion_prompt(
            persona,
            persona_system,
            cid,
            hints.get(cid, ""),
            judicial_instruction,
            evidence_text,
            allowed_refs,
        )
        op, op_err = _invoke_structured(prompt, persona, cid, allowed_refs=allowed_refs, retries=2)
        if op is not None:
            opinions.append(op)
        else:
            errors.append(f"{persona}|{cid}|{op_err or 'schema_validation_failed'}")
    return opinions, errors


def prosecutor_judge(state: AgentState) -> Dict[str, object]:
    opinions, errors = _run_batched_judge_with_errors("Prosecutor", PROSECUTOR_SYSTEM, state)
    return {"opinions": opinions, "judge_schema_failures": errors}


def defense_judge(state: AgentState) -> Dict[str, object]:
    opinions, errors = _run_batched_judge_with_errors("Defense", DEFENSE_SYSTEM, state)
    return {"opinions": opinions, "judge_schema_failures": errors}


def techlead_judge(state: AgentState) -> Dict[str, object]:
    opinions, errors = _run_batched_judge_with_errors("TechLead", TECHLEAD_SYSTEM, state)
    return {"opinions": opinions, "judge_schema_failures": errors}


def judge_barrier_node(state: AgentState) -> Dict[str, object]:
    """
    Fan-in barrier for judge outputs before routing to Chief Justice or JudgeRepair.
    """
    return {}
