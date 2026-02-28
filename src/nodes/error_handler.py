from __future__ import annotations

from typing import Dict

from src.state import AgentState, StopDecision


def _summarize_evidence(state: AgentState) -> str:
    evidences = state.get("evidences", {}) or {}
    lines = []
    for bucket, items in evidences.items():
        if not isinstance(items, list):
            continue
        found = sum(
            1
            for ev in items
            if (ev.get("found", False) if isinstance(ev, dict) else getattr(ev, "found", False))
        )
        lines.append(f"- {bucket}: {found}/{len(items)} found")
    return "\n".join(lines) if lines else "- No evidence collected."


def error_handler_node(state: AgentState) -> Dict[str, object]:
    error_type = (state.get("error_type") or "unknown_error").strip()
    error_message = (state.get("error_message") or "No details provided.").strip()
    failed_node = (state.get("failed_node") or "unknown").strip()
    tools_used = len(state.get("tool_runs", []) or [])
    iteration = int(state.get("iteration", 0) or 0)
    max_iters = int(state.get("max_iters", 6) or 6)
    tool_budget = int(state.get("tool_budget", 20) or 20)

    report = f"""# AUTOMATON AUDITOR - PARTIAL REPORT (ERROR HANDLER)

## Status
- Result: PARTIAL
- Failed Node: {failed_node}
- Error Type: {error_type}
- Error Message: {error_message}

## Runtime Snapshot
- Iteration: {iteration}/{max_iters}
- Tools Used: {tools_used}/{tool_budget}

## Evidence Coverage
{_summarize_evidence(state)}

## Next Actions
- Resolve the error above.
- Re-run the pipeline to complete missing rubric dimensions.
"""
    stop_decision = StopDecision(
        stop=True,
        reason=f"Terminated via ErrorHandler due to {error_type}.",
        remaining_risks=[error_message],
    )
    return {"final_report": report, "stop_decision": stop_decision}
