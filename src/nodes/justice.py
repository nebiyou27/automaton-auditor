# src/nodes/justice.py
# ====================
# Chief Justice: deterministic synthesis (NO LLM).
# Converts judge opinions + evidence into final Markdown report.

from __future__ import annotations

import json
import os
import re
from typing import Dict, List

from src.state import AgentState, JudicialOpinion


def _group_by_criterion(opinions: List[JudicialOpinion]) -> Dict[str, List[JudicialOpinion]]:
    grouped: Dict[str, List[JudicialOpinion]] = {}
    for op in opinions:
        grouped.setdefault(op.criterion_id, []).append(op)
    return grouped


def _resolve_score(opinions: List[JudicialOpinion]) -> Dict[str, object]:
    scores = {op.judge: op.score for op in opinions}
    prosecutor = scores.get("Prosecutor", 3)
    defense = scores.get("Defense", 3)
    techlead = scores.get("TechLead", 3)

    variance = max(scores.values()) - min(scores.values()) if scores else 0
    if variance > 2:
        return {
            "final_score": techlead,
            "reason": f"High variance detected (P={prosecutor}, D={defense}, T={techlead}). TechLead used as tiebreaker.",
        }

    final_score = round((prosecutor + defense + 2 * techlead) / 4)
    return {
        "final_score": final_score,
        "reason": f"Weighted scoring: P={prosecutor}, D={defense}, T={techlead} (double).",
    }


def _extract_cap_from_rule(rule_text: str, default: int = 3) -> int:
    text = str(rule_text or "")
    patterns = [
        r"\bcap(?:ped)?\s*(?:at|to)?\s*(\d+)\b",
        r"\bmax(?:imum)?\s*(?:at|to)?\s*(\d+)\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            try:
                return max(0, min(5, int(m.group(1))))
            except Exception:
                break

    m = re.search(r"\b([0-5])\b", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return default
    return default


def _extract_security_terms_from_rule(rule_text: str) -> List[str]:
    text = str(rule_text or "")
    terms: List[str] = []

    quoted = re.findall(r"[\"'`]([^\"'`]+)[\"'`]", text)
    terms.extend([t.strip().lower() for t in quoted if t.strip()])

    for chunk in re.split(r"[(),;]", text):
        for token in re.split(r"\s*(?:/|,|\bor\b|\band\b)\s*", chunk, flags=re.IGNORECASE):
            token = token.strip(" .:-").lower()
            if not token:
                continue
            if len(token) < 3:
                continue
            if token in {"if", "then", "when", "score", "cap", "capped", "at", "to", "due"}:
                continue
            if token not in terms:
                terms.append(token)

    return terms


def _has_security_override(evidences: Dict[str, List[object]], rule_text: str) -> bool:
    terms = _extract_security_terms_from_rule(rule_text)
    if not terms:
        return False

    for items in evidences.values():
        for ev in items:
            content = ""
            if isinstance(ev, dict):
                content = str(ev.get("content", "") or "")
            else:
                content = str(getattr(ev, "content", "") or "")
            lowered = content.lower()
            if any(term in lowered for term in terms):
                return True
    return False


def _resolve_rubric_path(rubric_path: str) -> str:
    path = (rubric_path or "rubric/week2_rubric.json").strip()
    if os.path.isabs(path):
        return path
    return os.path.join(os.getcwd(), path)


def _load_synthesis_rules(rubric_path: str) -> dict:
    path = _resolve_rubric_path(rubric_path)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            rules = json.load(f).get("synthesis_rules", {})
            return rules if isinstance(rules, dict) else {}
    except Exception:
        return {}


def _is_found(ev: object) -> bool:
    if isinstance(ev, dict):
        return bool(ev.get("found", False))
    return bool(getattr(ev, "found", False))


def _get_goal(ev: object) -> str:
    if isinstance(ev, dict):
        return str(ev.get("goal", "") or "")
    return str(getattr(ev, "goal", "") or "")


def _audit_blockers(evidences: Dict[str, List[object]]) -> List[str]:
    blockers: List[str] = []

    repo_items = evidences.get("repo", []) or []
    clone_failed = any(
        ("clone repository in sandbox" in _get_goal(ev).lower()) and (not _is_found(ev))
        for ev in repo_items
    )
    if clone_failed:
        blockers.append("Repository clone/access failed, so repo-backed evidence is incomplete.")

    dynamic_items = evidences.get("dynamic_plan", []) or []
    has_plan = any("llm investigation plan for:" in _get_goal(ev).lower() for ev in dynamic_items)
    has_execution = any(_get_goal(ev).startswith("Dynamic check [") for ev in dynamic_items)
    if has_plan and not has_execution:
        blockers.append("Only plan evidence was produced for dynamic checks; execution evidence is missing.")

    return blockers


def generate_markdown_report(state: AgentState, rules: dict) -> str:
    grouped = _group_by_criterion(state.get("opinions", []) or [])
    evidences = state.get("evidences", {}) or {}
    blockers = _audit_blockers(evidences)
    incomplete_audit = len(blockers) > 0
    security_rule = str(rules.get("security_override", "") or "")
    security_cap = _extract_cap_from_rule(security_rule, default=3)
    security_override_active = bool(security_rule) and _has_security_override(evidences, security_rule)

    if not grouped:
        grouped = {"detective_phase": []}

    criterion_sections: List[str] = []
    remediation: List[str] = []
    total = 0
    count = 0

    for criterion_id, ops in grouped.items():
        if ops:
            resolution = _resolve_score(ops)
            final_score = int(resolution["final_score"])
            reason = str(resolution["reason"])
        else:
            final_score = 0
            reason = "No judge opinions available."

        dissent_lines = []
        for op in ops:
            dissent_lines.append(f"- **{op.judge}** (Score {op.score}/5): {op.argument}")
            if op.cited_evidence:
                dissent_lines.append(f"  - Cited: {', '.join(op.cited_evidence)}")

        if security_override_active and final_score > security_cap:
            final_score = security_cap
            dissent_lines.append(
                f"- **Security Override**: Score capped at {security_cap} due to synthesis rule `security_override`."
            )

        total += final_score
        count += 1

        if final_score < 4:
            remediation.append(f"- Improve `{criterion_id}` based on cited evidence and deterministic resolution notes.")

        criterion_sections.append(
            f"""### Criterion: {criterion_id}
**Final Score:** {final_score}/5

**Dissent (Judge Opinions):**
{chr(10).join(dissent_lines) if dissent_lines else "- No opinions."}

**Deterministic Resolution:**
{reason}
"""
        )

    avg = round(total / count) if count else 0

    evidence_summary = []
    for bucket, items in evidences.items():
        found = sum(1 for e in items if (e.get("found", False) if isinstance(e, dict) else getattr(e, "found", False)))
        evidence_summary.append(f"- **{bucket.upper()}**: {found}/{len(items)} found")

    if not remediation:
        remediation.append("- No urgent remediation required based on current scoring.")

    executive_extras: List[str] = []
    executive_extras.append(f"- **Audit Status:** {'INCOMPLETE_AUDIT' if incomplete_audit else 'COMPLETE'}")
    if "dissent_requirement" in rules:
        executive_extras.append(f"- **Dissent Requirement:** {rules.get('dissent_requirement')}")

    fact_supremacy_section = ""
    if "fact_supremacy" in rules:
        fact_supremacy_section = f"""
## Synthesis Notes
- **Fact Supremacy:** {rules.get("fact_supremacy")}
"""

    if incomplete_audit:
        blocker_lines = "\n".join([f"- {b}" for b in blockers])
        return f"""# AUTOMATON AUDITOR - FINAL VERDICT

## Executive Summary
- **Repository:** {state.get("repo_url", "N/A")}
- **PDF:** {state.get("pdf_path", "N/A") or "None"}
- **Overall Score:** N/A (incomplete audit)
- **Evidence Coverage:**
{chr(10).join(evidence_summary) if evidence_summary else "- No evidence collected."}
{chr(10).join(executive_extras) if executive_extras else ""}

## Blocking Issues
{blocker_lines}

## Remediation Plan
- Resolve blocking issues above, rerun detectives, then regenerate judicial verdict.
{fact_supremacy_section}
"""

    # FIXED: C11a
    return f"""# AUTOMATON AUDITOR - FINAL VERDICT

## Executive Summary
- **Repository:** {state.get("repo_url", "N/A")}
- **PDF:** {state.get("pdf_path", "N/A") or "None"}
- **Overall Score:** {avg}/5
- **Evidence Coverage:**
{chr(10).join(evidence_summary) if evidence_summary else "- No evidence collected."}
{chr(10).join(executive_extras) if executive_extras else ""}

## Criterion Breakdown
{chr(10).join(criterion_sections)}

## Remediation Plan
{chr(10).join(remediation)}
{fact_supremacy_section}
"""


def chief_justice(state: AgentState) -> Dict[str, str]:
    rubric_path = state.get("rubric_path", "rubric/week2_rubric.json")
    rules = _load_synthesis_rules(rubric_path)
    return {"final_report": generate_markdown_report(state, rules)}
