# src/nodes/justice.py
# ====================
# Chief Justice: deterministic synthesis (NO LLM).
# Converts judge opinions + evidence into final Markdown report.

from __future__ import annotations

from typing import Dict, List

from src.state import AgentState, JudicialOpinion


SECURITY_TERMS = ("os.system", "shell injection", "security negligence")


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


def _has_security_override(evidences: Dict[str, List[object]]) -> bool:
    for items in evidences.values():
        for ev in items:
            content = ""
            if isinstance(ev, dict):
                content = str(ev.get("content", "") or "")
            else:
                content = str(getattr(ev, "content", "") or "")
            lowered = content.lower()
            if any(term in lowered for term in SECURITY_TERMS):
                return True
    return False


def generate_markdown_report(state: AgentState) -> str:
    grouped = _group_by_criterion(state.get("opinions", []) or [])
    evidences = state.get("evidences", {}) or {}

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

        # FIXED: C10
        if _has_security_override(evidences) and final_score > 3:
            final_score = 3
            dissent_lines.append(
                "- **Security Override**: Score capped at 3 due to security risk indicators (os.system/shell injection/security negligence)."
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

    # FIXED: C11a
    return f"""# AUTOMATON AUDITOR - FINAL VERDICT

## Executive Summary
- **Repository:** {state.get("repo_url", "N/A")}
- **PDF:** {state.get("pdf_path", "N/A") or "None"}
- **Overall Score:** {avg}/5
- **Evidence Coverage:**
{chr(10).join(evidence_summary) if evidence_summary else "- No evidence collected."}

## Criterion Breakdown
{chr(10).join(criterion_sections)}

## Remediation Plan
{chr(10).join(remediation)}
"""


def chief_justice(state: AgentState) -> Dict[str, str]:
    return {"final_report": generate_markdown_report(state)}
