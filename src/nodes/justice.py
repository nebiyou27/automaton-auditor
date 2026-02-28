# src/nodes/justice.py
# ====================
# Chief Justice: deterministic synthesis (NO LLM).
# Converts judge opinions + evidence into final Markdown report.

from __future__ import annotations

import ast
import json
import os
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, List, Tuple

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


def _coerce_score(score: object, default: int = 3) -> int:
    try:
        value = int(score)
    except (TypeError, ValueError):
        return default
    return max(1, min(5, value))


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


def _safe_get(ev: object, key: str, default: object = "") -> object:
    if isinstance(ev, dict):
        return ev.get(key, default)
    return getattr(ev, key, default)


def _build_evidence_index(evidences: Dict[str, List[object]]) -> Dict[str, List[object]]:
    index: Dict[str, List[object]] = {}
    for items in evidences.values():
        for ev in items:
            location = str(_safe_get(ev, "location", "") or "").strip()
            if not location:
                continue
            index.setdefault(location, []).append(ev)
    return index


def _citation_supported(citation: str, evidence_index: Dict[str, List[object]]) -> bool:
    ref = (citation or "").strip()
    if not ref:
        return False
    for location, items in evidence_index.items():
        if ref == location or ref in location or location in ref:
            if any(_is_found(ev) for ev in items):
                return True
    return False


def _extract_text_from_evidence(ev: object) -> str:
    parts = [
        str(_safe_get(ev, "dimension_id", "") or ""),
        str(_safe_get(ev, "goal", "") or ""),
        str(_safe_get(ev, "rationale", "") or ""),
        str(_safe_get(ev, "location", "") or ""),
        str(_safe_get(ev, "content", "") or ""),
    ]
    return " ".join(parts).lower()


def _has_confirmed_unsafe_tool_engineering_evidence(
    evidences: Dict[str, List[object]],
    rule_text: str,
) -> bool:
    rule_terms = _extract_security_terms_from_rule(rule_text)
    unsafe_terms = {
        "shell injection",
        "os.system",
        "unsanitized",
        "unsafe",
        "command injection",
        "subprocess shell=true",
        "no input sanitization",
        "temp directory missing",
        "raw shell",
    }
    if rule_terms:
        unsafe_terms.update(t for t in rule_terms if len(t) > 2)

    for items in evidences.values():
        for ev in items:
            if not _is_found(ev):
                continue
            text = _extract_text_from_evidence(ev)
            dim = str(_safe_get(ev, "dimension_id", "") or "").lower()
            if "safe_tool_engineering" not in dim and "tool" not in text and "security" not in text:
                continue
            if any(term in text for term in unsafe_terms):
                return True
    return False


def _has_security_override(evidences: Dict[str, List[object]], rule_text: str) -> bool:
    # Backward-compatible wrapper used by existing detective checks.
    return _has_confirmed_unsafe_tool_engineering_evidence(evidences, rule_text)


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


def _is_score_argument_inconsistent(op: JudicialOpinion) -> bool:
    if _is_positive_argument(op.argument) and op.score <= 2:
        return True
    if _is_negative_argument(op.argument) and op.score >= 4:
        return True
    return False


def _is_positive_claim(op: JudicialOpinion) -> bool:
    return op.score >= 4 or _is_positive_argument(op.argument)


def _mentions_modular_workable(argument: str) -> bool:
    lowered = (argument or "").lower()
    checks = [
        "modular",
        "workable",
        "maintainable",
        "fan-out",
        "fan in",
        "fan-in",
        "orchestration",
        "stategraph",
    ]
    hits = sum(1 for c in checks if c in lowered)
    return hits >= 2


def _extract_file_path_from_citation(citation: str) -> str:
    raw = (citation or "").strip()
    if not raw:
        return ""
    no_line = raw.split("#", 1)[0].split(":", 1)[0]
    candidate = no_line.strip()
    if not candidate:
        return ""
    if candidate.lower().endswith((".py", ".md", ".json", ".txt", ".yaml", ".yml")):
        return candidate
    return ""


def _recheck_citation(citation: str, evidence_index: Dict[str, List[object]], state: AgentState) -> Tuple[bool, str]:
    if _citation_supported(citation, evidence_index):
        return True, f"`{citation}` found in collected evidence."

    file_path = _extract_file_path_from_citation(citation)
    if file_path:
        repo_path = (state.get("repo_path") or "").strip()
        candidate = Path(file_path)
        if not candidate.is_absolute() and repo_path:
            candidate = Path(repo_path) / file_path
        elif not candidate.is_absolute():
            candidate = Path.cwd() / file_path

        if candidate.exists() and candidate.is_file():
            try:
                snippet = candidate.read_text(encoding="utf-8", errors="ignore")[:1200]
                if snippet.strip():
                    return True, f"Re-opened file snippet from `{candidate}`."
            except Exception:
                pass
        return False, f"File snippet re-check failed for `{candidate}`."

    lowered = (citation or "").lower()
    if ".pdf" in lowered or "chunk" in lowered or "page" in lowered:
        return False, f"PDF chunk citation `{citation}` not found in indexed evidence."
    return False, f"Citation `{citation}` could not be matched to evidence."


def _maybe_rerun_ast(citations: List[str], state: AgentState) -> Tuple[bool, str]:
    repo_path = (state.get("repo_path") or "").strip()
    for citation in citations:
        path_hint = _extract_file_path_from_citation(citation)
        if not path_hint or not path_hint.lower().endswith(".py"):
            continue
        candidate = Path(path_hint)
        if not candidate.is_absolute() and repo_path:
            candidate = Path(repo_path) / path_hint
        elif not candidate.is_absolute():
            candidate = Path.cwd() / path_hint
        if not candidate.exists() or not candidate.is_file():
            continue
        try:
            source = candidate.read_text(encoding="utf-8", errors="ignore")
            ast.parse(source)
            return True, f"Re-ran AST parse for `{candidate}`."
        except Exception as e:
            return False, f"AST re-check failed for `{candidate}` ({e.__class__.__name__})."
    return False, "No Python citation available for AST re-check."


def _apply_fact_supremacy(
    op: JudicialOpinion,
    evidence_index: Dict[str, List[object]],
    rule_enabled: bool,
) -> Tuple[JudicialOpinion, List[str]]:
    notes: List[str] = []
    if not rule_enabled:
        return op, notes

    cited = op.cited_evidence or []
    supported = [c for c in cited if _citation_supported(c, evidence_index)]
    if _is_positive_claim(op) and len(supported) == 0:
        original = op.score
        op.score = min(op.score, 2)
        notes.append(
            f"Fact supremacy overruled unsupported {op.judge} claim (score {original} -> {op.score})."
        )
    return op, notes


def _resolve_weighted_score(
    criterion_id: str,
    opinions: List[JudicialOpinion],
    functionality_weight_rule_enabled: bool,
) -> Dict[str, object]:
    scores = {op.judge: _coerce_score(op.score) for op in opinions}
    prosecutor = scores.get("Prosecutor", 3)
    defense = scores.get("Defense", 3)
    techlead = scores.get("TechLead", 3)
    variance = max(scores.values()) - min(scores.values()) if scores else 0

    graph_criterion = criterion_id == "graph_orchestration"
    techlead_op = next((op for op in opinions if op.judge == "TechLead"), None)
    techlead_confirms_modular = bool(
        techlead_op and techlead_op.score >= 4 and _mentions_modular_workable(techlead_op.argument)
    )
    use_highest_techlead_weight = (
        functionality_weight_rule_enabled and graph_criterion and techlead_confirms_modular
    )

    if variance > 2:
        reason = (
            f"High variance detected (P={prosecutor}, D={defense}, T={techlead}); pending variance re-evaluation."
        )
        return {"final_score": techlead, "reason": reason, "variance": variance}

    if use_highest_techlead_weight:
        total = prosecutor + defense + (4 * techlead)
        final_score = round(total / 6)
        reason = (
            f"Functionality-weight applied for graph orchestration: P={prosecutor}, D={defense}, T={techlead} (x4)."
        )
    else:
        final_score = round((prosecutor + defense + 2 * techlead) / 4)
        reason = f"Weighted scoring: P={prosecutor}, D={defense}, T={techlead} (x2)."
    return {"final_score": final_score, "reason": reason, "variance": variance}


def _variance_re_evaluation(
    criterion_id: str,
    opinions: List[JudicialOpinion],
    evidence_index: Dict[str, List[object]],
    state: AgentState,
    enabled: bool,
) -> Tuple[int, str, List[str]]:
    base = _resolve_weighted_score(criterion_id, opinions, functionality_weight_rule_enabled=False)
    final_score = int(base["final_score"])
    reason = str(base["reason"])
    variance = int(base["variance"])
    notes: List[str] = []

    if not enabled or variance <= 2:
        return final_score, reason, notes

    notes.append(
        f"Variance re-evaluation triggered for `{criterion_id}` (variance={variance})."
    )
    reliability: Dict[str, float] = {"Prosecutor": 0.0, "Defense": 0.0, "TechLead": 0.0}
    for op in opinions:
        citations = op.cited_evidence or []
        if not citations:
            reliability[op.judge] = 0.0
            notes.append(f"{op.judge}: no citations supplied.")
            continue

        pass_count = 0
        for citation in citations:
            ok, detail = _recheck_citation(citation, evidence_index, state)
            notes.append(f"{op.judge}: {detail}")
            if ok:
                pass_count += 1

        ast_ok, ast_detail = _maybe_rerun_ast(citations, state)
        notes.append(f"{op.judge}: {ast_detail}")
        if ast_ok:
            pass_count += 1

        reliability[op.judge] = pass_count / max(1, len(citations) + 1)

    total_weight = 0.0
    weighted_sum = 0.0
    for op in opinions:
        weight = max(0.2, reliability.get(op.judge, 0.0))
        total_weight += weight
        weighted_sum += weight * _coerce_score(op.score)

    if total_weight > 0:
        final_score = round(weighted_sum / total_weight)
    reason = (
        f"Variance re-evaluation applied using citation reliability weights "
        f"(P={reliability.get('Prosecutor', 0.0):.2f}, "
        f"D={reliability.get('Defense', 0.0):.2f}, "
        f"T={reliability.get('TechLead', 0.0):.2f})."
    )
    return final_score, reason, notes


def _resolve_output_markdown_path(state: AgentState) -> Path:
    preferred = (state.get("report_path") or state.get("out_path") or "").strip()
    if preferred:
        path = Path(preferred)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.suffix.lower() != ".md":
            path = path.with_suffix(".md")
        return path

    repo_url = str(state.get("repo_url", "") or "")
    is_self = "nebiyou27" in repo_url
    folder = "report_onself_generated" if is_self else "report_onpeer_generated"
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return Path.cwd() / "audit" / folder / f"chief_justice_report_{stamp}.md"


def _write_markdown_report(state: AgentState, report: str) -> str:
    path = _resolve_output_markdown_path(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return str(path)


def generate_markdown_report(state: AgentState, rules: dict) -> str:
    grouped = _group_by_criterion(state.get("opinions", []) or [])
    evidences = state.get("evidences", {}) or {}
    evidence_index = _build_evidence_index(evidences)
    blockers = _audit_blockers(evidences)
    incomplete_audit = len(blockers) > 0
    security_rule = str(rules.get("security_override", "") or "")
    security_cap = _extract_cap_from_rule(security_rule, default=3)
    security_override_active = bool(security_rule) and _has_confirmed_unsafe_tool_engineering_evidence(
        evidences, security_rule
    )
    fact_supremacy_enabled = "fact_supremacy" in rules
    functionality_weight_enabled = "functionality_weight" in rules
    dissent_requirement_enabled = "dissent_requirement" in rules
    variance_re_eval_enabled = "variance_re_evaluation" in rules

    criterion_sections: List[str] = []
    remediation: List[str] = []
    total = 0
    count = 0
    contradiction_count = 0

    for criterion_id, ops in grouped.items():
        normalized_ops: List[JudicialOpinion] = []
        fact_notes: List[str] = []
        for op in ops:
            normalized_score = _coerce_score(op.score)
            if normalized_score != op.score:
                op.score = normalized_score
            op, notes = _apply_fact_supremacy(op, evidence_index, fact_supremacy_enabled)
            normalized_ops.append(op)
            fact_notes.extend(notes)

        ops = normalized_ops
        if ops:
            weighted_resolution = _resolve_weighted_score(
                criterion_id,
                ops,
                functionality_weight_rule_enabled=functionality_weight_enabled,
            )
            final_score = int(weighted_resolution["final_score"])
            reason = str(weighted_resolution["reason"])
            variance = int(weighted_resolution["variance"])
        else:
            final_score = 0
            reason = "No judge opinions available."
            variance = 0

        re_eval_notes: List[str] = []
        if variance > 2:
            v_score, v_reason, v_notes = _variance_re_evaluation(
                criterion_id=criterion_id,
                opinions=ops,
                evidence_index=evidence_index,
                state=state,
                enabled=variance_re_eval_enabled,
            )
            final_score = v_score
            reason = v_reason if variance_re_eval_enabled else reason
            re_eval_notes.extend(v_notes)

        dissent_lines = []
        for op in ops:
            dissent_lines.append(f"- **{op.judge}** (Score {op.score}/5): {op.argument}")
            if op.cited_evidence:
                dissent_lines.append(f"  - Cited: {', '.join(op.cited_evidence)}")
            if _is_score_argument_inconsistent(op):
                contradiction_count += 1
                dissent_lines.append(
                    "  - Consistency note: score appears misaligned with argument sentiment."
                )

        for note in fact_notes:
            dissent_lines.append(f"- **Fact Supremacy:** {note}")
        for note in re_eval_notes:
            dissent_lines.append(f"- **Variance Re-check:** {note}")

        if dissent_requirement_enabled and variance > 2:
            prosecutor_op = next((op for op in ops if op.judge == "Prosecutor"), None)
            defense_op = next((op for op in ops if op.judge == "Defense"), None)
            prosecutor_view = prosecutor_op.argument if prosecutor_op else "No Prosecutor argument."
            defense_view = defense_op.argument if defense_op else "No Defense argument."
            dissent_lines.append(
                f"- **Required Dissent Summary:** Prosecutor argued: \"{prosecutor_view}\" | "
                f"Defense argued: \"{defense_view}\"."
            )

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
    if contradiction_count:
        executive_extras.append(f"- **Judge Consistency Alerts:** {contradiction_count}")
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
    report = generate_markdown_report(state, rules)
    report_path = _write_markdown_report(state, report)
    report_with_path = (
        f"{report}\n\n## Output File\n- Markdown report written to: `{report_path}`\n"
    )
    return {"final_report": report_with_path}
