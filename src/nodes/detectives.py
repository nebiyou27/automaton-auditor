# src/nodes/detectives.py
# =======================
# Detective layer: collect FACTS only.
# No scoring. No judgment. Only structured Evidence objects.

from __future__ import annotations

import base64
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from src.state import AgentState, Evidence
from src.tools.repo_tools import (
    ToolResult,
    analyze_langgraph_graph_py,
    clone_repo_sandboxed,
    extract_git_history,
    find_typed_state_definitions,
    ingest_pdf_for_query,
    query_pdf_chunks,
)
from src.tools.rubric_utils import get_dimensions_for, load_rubric


def _evidence(
    *,
    goal: str,
    found: bool,
    location: str,
    rationale: str,
    content: str | None = None,
    confidence: float = 1.0,
    dimension_id: str = "",
) -> Evidence:
    return Evidence(
        goal=goal,
        found=found,
        location=location,
        rationale=rationale,
        content=content,
        confidence=confidence,
        dimension_id=dimension_id,
    )


def _extract_reducer_snippet(state_file_path: str, max_lines: int = 25) -> str | None:
    try:
        with open(state_file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        keep: List[str] = []
        for ln in lines:
            s = ln.strip()
            if (
                "evidences:" in s
                or "opinions:" in s
                or "Annotated[" in s
                or "operator.ior" in s
                or "operator.add" in s
            ):
                keep.append(ln.rstrip("\n"))

        if not keep:
            return None
        return "\n".join(keep[:max_lines])
    except Exception:
        return None


def _read_text_file(path: str, max_chars: int = 2000) -> tuple[bool, str]:
    if not os.path.isfile(path):
        return False, f"File not found: {path}"
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return True, f.read(max_chars)
    except Exception as e:
        return False, f"Read failed: {e!r}"


def _parse_git_date(value: str) -> datetime | None:
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S %z")
    except Exception:
        return None


def _classify_git_progression(commits: List[Dict[str, str]]) -> Dict[str, object]:
    count = len(commits)
    if count == 0:
        return {
            "commit_count": 0,
            "progression_story": False,
            "init_only": False,
            "bulk_upload_suspected": False,
            "timestamps_clustered_minutes": None,
            "timestamps_clustered": False,
            "phase_hit_count": 0,
            "phase_hit_indices": {"setup": None, "tool_engineering": None, "graph_orchestration": None},
            "classification": "mixed",
            "phase_hits": {"setup": False, "tool_engineering": False, "graph_orchestration": False},
        }

    messages = [str(c.get("message", "") or "").lower() for c in commits]
    dates = [_parse_git_date(str(c.get("date", "") or "")) for c in commits]
    valid_dates = [d for d in dates if d is not None]

    setup_kw = ("setup", "environment", "env", "bootstrap", "requirements", "dependency", "venv")
    tool_kw = ("tool", "tools", "ast", "pdf", "repo", "executor", "planner", "detective")
    graph_kw = ("graph", "langgraph", "orchestration", "stategraph", "judge", "justice", "reflector")

    def _first_idx(keywords: tuple[str, ...]) -> int | None:
        for i, msg in enumerate(messages):
            if any(k in msg for k in keywords):
                return i
        return None

    idx_setup = _first_idx(setup_kw)
    idx_tool = _first_idx(tool_kw)
    idx_graph = _first_idx(graph_kw)

    phase_hits = {
        "setup": idx_setup is not None,
        "tool_engineering": idx_tool is not None,
        "graph_orchestration": idx_graph is not None,
    }
    progression_story = (
        idx_setup is not None
        and idx_tool is not None
        and idx_graph is not None
        and idx_setup < idx_tool < idx_graph
    )

    init_only = count == 1 and any(t in messages[0] for t in ("init", "initial", "first commit"))
    has_bulk_word = any(("bulk" in m) or ("upload" in m) for m in messages)
    clustered_minutes = None
    clustered = False
    if len(valid_dates) >= 2:
        span_s = (max(valid_dates) - min(valid_dates)).total_seconds()
        clustered_minutes = round(span_s / 60.0, 2)
        clustered = span_s <= 10 * 60
    bulk_upload_suspected = bool(has_bulk_word or init_only or (clustered and count <= 3))
    phase_hit_count = sum(1 for v in phase_hits.values() if v)
    if init_only:
        classification = "init_only"
    elif progression_story and not bulk_upload_suspected:
        classification = "progressive_build"
    elif bulk_upload_suspected and not progression_story:
        classification = "bulk_upload"
    else:
        classification = "mixed"

    return {
        "commit_count": count,
        "progression_story": progression_story,
        "init_only": init_only,
        "bulk_upload_suspected": bulk_upload_suspected,
        "timestamps_clustered_minutes": clustered_minutes,
        "timestamps_clustered": clustered,
        "phase_hit_count": phase_hit_count,
        "phase_hit_indices": {"setup": idx_setup, "tool_engineering": idx_tool, "graph_orchestration": idx_graph},
        "classification": classification,
        "phase_hits": phase_hits,
    }


def _git_confidence(classifier: Dict[str, object]) -> float:
    count = int(classifier.get("commit_count", 0) or 0)
    phase_hits = int(classifier.get("phase_hit_count", 0) or 0)
    clustered = bool(classifier.get("timestamps_clustered", False))
    classification = str(classifier.get("classification", "mixed"))

    if count >= 4 and phase_hits >= 2 and not clustered:
        return 0.93
    if classification == "init_only" or clustered or phase_hits <= 1:
        return 0.62
    return 0.80


def _fmt_commit(c: Dict[str, str]) -> str:
    return f"{c.get('hash', '')} {c.get('message', '')} ({c.get('date', '')})".strip()


def _select_representative_commits(commits: List[Dict[str, str]], classifier: Dict[str, object]) -> List[Dict[str, str]]:
    if not commits:
        return []

    idxs: List[int] = [0, len(commits) - 1]
    phase_idx = classifier.get("phase_hit_indices", {})
    if isinstance(phase_idx, dict):
        for k in ("setup", "tool_engineering", "graph_orchestration"):
            v = phase_idx.get(k)
            if isinstance(v, int):
                idxs.append(v)

    # Optional 4th phase-like anchor: first explicit bulk/upload marker if present.
    for i, c in enumerate(commits):
        msg = str(c.get("message", "")).lower()
        if "bulk" in msg or "upload" in msg:
            idxs.append(i)
            break

    unique_sorted = sorted({i for i in idxs if 0 <= i < len(commits)})
    reps = [commits[i] for i in unique_sorted]

    # Ensure 3+ representatives when possible.
    if len(reps) < 3 and len(commits) >= 3:
        mid = len(commits) // 2
        for i in (mid, max(1, mid - 1), min(len(commits) - 2, mid + 1)):
            if commits[i] not in reps:
                reps.append(commits[i])
            if len(reps) >= 3:
                break

    return reps[:6]


def _build_git_forensic_summary(commits: List[Dict[str, str]], classifier: Dict[str, object]) -> Dict[str, object]:
    count = int(classifier.get("commit_count", 0) or 0)
    first = commits[0].get("hash", "") if commits else ""
    last = commits[-1].get("hash", "") if commits else ""
    phase_hits = int(classifier.get("phase_hit_count", 0) or 0)
    clustered = bool(classifier.get("timestamps_clustered", False))
    label = str(classifier.get("classification", "mixed"))
    reps = _select_representative_commits(commits, classifier)
    reps_txt = "; ".join(_fmt_commit(c) for c in reps[:6]) if reps else "none"

    rationale = (
        f"Git progression classification: {label} "
        f"(commit_count={count}, phase_hits={phase_hits}, timestamps_clustered={'yes' if clustered else 'no'}). "
        f"Representative commits: {reps_txt}"
    )
    location = f"git log --oneline --reverse (first={first or 'none'} last={last or 'none'}, count={count})"
    confidence = _git_confidence(classifier)
    return {
        "classification": label,
        "rationale": rationale,
        "location": location,
        "confidence": confidence,
        "representative_commits": reps,
    }


REQUIRED_CHECKS_BY_CRITERION: Dict[str, List[Dict[str, str]]] = {
    "git_forensic_analysis": [
        {"tool": "git_log", "target": "", "reason": "Extract full commit progression and timestamps."},
    ],
    "state_management_rigor": [
        {"tool": "check_file_exists", "target": "src/state.py", "reason": "State typing source must exist."},
        {"tool": "grep_search", "target": "class Evidence(BaseModel)", "reason": "Verify BaseModel evidence type."},
        {"tool": "grep_search", "target": "class JudicialOpinion(BaseModel)", "reason": "Verify BaseModel opinion type."},
        {"tool": "grep_search", "target": "operator.ior", "reason": "Verify parallel-safe evidence reducer."},
        {"tool": "grep_search", "target": "operator.add", "reason": "Verify parallel-safe opinion reducer."},
    ],
    "safe_tool_engineering": [
        {"tool": "check_file_exists", "target": "src/tools/repo_tools.py", "reason": "Tool implementation file must exist."},
        {"tool": "grep_search", "target": "TemporaryDirectory(", "reason": "Verify sandboxed clone lifecycle."},
        {"tool": "grep_search", "target": "query_pdf_chunks", "reason": "Verify PDF query interface exists."},
        {"tool": "grep_search", "target": "os.system(", "reason": "Detect unsafe shell usage if present."},
        {"tool": "grep_search", "target": "try:", "reason": "Find explicit exception handling in tooling."},
    ],
    "graph_orchestration": [
        {"tool": "check_file_exists", "target": "src/graph.py", "reason": "Graph orchestration file must exist."},
        {"tool": "grep_search", "target": "StateGraph(", "reason": "Verify graph compilation wiring exists."},
        {"tool": "grep_search", "target": "add_edge(", "reason": "Verify edge wiring for fan-in/fan-out."},
        {"tool": "grep_search", "target": "add_conditional_edges(", "reason": "Verify conditional routing exists."},
    ],
    "structured_output_enforcement": [
        {"tool": "check_file_exists", "target": "src/nodes/judges.py", "reason": "Judge implementation must exist."},
        {"tool": "grep_search", "target": "JudicialOpinion", "reason": "Structured opinion schema should be referenced."},
        {"tool": "grep_search", "target": "_invoke_structured(", "reason": "Structured invocation helper should exist."},
        {"tool": "grep_search", "target": "JSONDecodeError", "reason": "Malformed JSON handling should exist."},
    ],
    "judicial_nuance": [
        {"tool": "check_file_exists", "target": "src/nodes/judges.py", "reason": "Judge personas implementation should exist."},
        {"tool": "grep_search", "target": "PROSECUTOR_SYSTEM", "reason": "Prosecutor persona prompt should exist."},
        {"tool": "grep_search", "target": "DEFENSE_SYSTEM", "reason": "Defense persona prompt should exist."},
        {"tool": "grep_search", "target": "TECHLEAD_SYSTEM", "reason": "Tech Lead persona prompt should exist."},
    ],
    "chief_justice_synthesis": [
        {"tool": "check_file_exists", "target": "src/nodes/justice.py", "reason": "Chief justice synthesis should exist."},
        {"tool": "grep_search", "target": "generate_markdown_report", "reason": "Structured markdown synthesis should exist."},
        {"tool": "grep_search", "target": "_resolve_score(", "reason": "Deterministic resolution logic should exist."},
        {"tool": "grep_search", "target": "_has_security_override(", "reason": "Security override rule evaluation should exist."},
    ],
}


def _execute_tool_check(tool_name: str, target: str, repo_path: str) -> Tuple[bool, str, str]:
    result = ""
    found = False
    location = target or repo_path

    if tool_name == "check_file_exists":
        full = os.path.join(repo_path, target)
        found = os.path.exists(full)
        result = f"exists={found}"
        location = target

    elif tool_name == "read_file":
        full = os.path.join(repo_path, target)
        ok, content = _read_text_file(full, max_chars=2000)
        found = ok
        result = content[:500] if ok else content
        location = target

    elif tool_name == "grep_search":
        matches = []
        for py_file in Path(repo_path).rglob("*.py"):
            try:
                text = py_file.read_text(encoding="utf-8", errors="ignore")
                for i, line in enumerate(text.splitlines(), 1):
                    if target.lower() in line.lower():
                        matches.append(f"{py_file.relative_to(repo_path)}:{i}: {line.strip()}")
            except Exception:
                continue
        found = len(matches) > 0
        result = "\n".join(matches[:10])
        location = target

    elif tool_name == "ast_parse":
        full = os.path.join(repo_path, target)
        res = analyze_langgraph_graph_py(full)
        found = res.ok
        result = str(res.data)[:500] if res.ok else str(res.error)
        location = target

    elif tool_name == "git_log":
        res = extract_git_history(repo_path)
        found = res.ok
        commits: List[Dict[str, str]] = []
        if res.ok:
            commits = list((res.data or {}).get("commits", []))
            classifier = _classify_git_progression(commits)
            summary = _build_git_forensic_summary(commits, classifier)
            result = json.dumps(
                {
                    "git_history": {
                        "count": (res.data or {}).get("count", 0),
                        "empty_repo": (res.data or {}).get("empty_repo", False),
                        "commits_preview": commits[:10]
                    },
                    "git_progression_classifier": classifier,
                    "git_forensic_summary": summary,
                }
            )[:1200]
        else:
            result = str(res.error)
        if commits:
            location = f"git log --oneline --reverse (first={commits[0].get('hash','')} last={commits[-1].get('hash','')}, count={len(commits)})"
        else:
            location = "git log --oneline --reverse (first=none last=none, count=0)"

    else:
        result = f"Unsupported tool: {tool_name}"

    if tool_name == "grep_search" and "os.system(" in target and found:
        found = False
        result = f"Unsafe usage detected\n{result}"

    if tool_name == "grep_search" and "LANGCHAIN_API_KEY=" in target and found:
        found = False
        result = f"Potential committed secret pattern detected\n{result}"

    return found, result, location


def _run_dynamic_plan_checks(state: AgentState, repo_path: str, dimensions: List[dict]) -> List[Evidence]:
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "deepseek-r1:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
        format="json",
    )

    tool_menu = """
    Available tools:
    - check_file_exists(path): verify file exists in repo
    - read_file(path, chars): read file content
    - grep_search(term): find term across all .py files
    - ast_parse(path): analyze Python AST structure
    - git_log(): get commit history and timestamps
    """

    all_evidences: List[Evidence] = []

    for dim in dimensions:
        dim_id = str(dim.get("id", "") or "unknown_criterion")
        required_checks = REQUIRED_CHECKS_BY_CRITERION.get(dim_id, [])
        executed_signatures = set()

        for check in required_checks:
            tool_name = check.get("tool", "")
            target = check.get("target", "")
            reason = check.get("reason", "")
            signature = f"{tool_name}|{target}"
            executed_signatures.add(signature)
            try:
                found, result, location = _execute_tool_check(tool_name, target, repo_path)
            except Exception as e:
                found = False
                result = f"Execution error: {e!r}"
                location = target or repo_path

            rationale_text = reason
            content_text = str(result)[:500] if result else None
            confidence_value = 0.9
            if tool_name == "git_log" and found and result:
                try:
                    parsed = json.loads(str(result))
                    gsum = parsed.get("git_forensic_summary", {})
                    if isinstance(gsum, dict):
                        rationale_text = str(gsum.get("rationale", rationale_text))
                        location = str(gsum.get("location", location))
                        confidence_value = float(gsum.get("confidence", confidence_value))
                    content_text = json.dumps(parsed)[:1200]
                except Exception:
                    pass

            all_evidences.append(
                _evidence(
                    goal=f"Required check [{dim_id}]: {tool_name} on '{target}'",
                    found=found,
                    location=location,
                    rationale=rationale_text,
                    content=content_text,
                    confidence=confidence_value,
                    dimension_id=dim_id if tool_name == "git_log" else "",
                )
            )

        prompt = f"""You are a forensic investigator.
Rubric criterion: {json.dumps(dim, indent=2)}
Available tools: {tool_menu}
List ONLY checks needed for this criterion.
Reply JSON only: {{"checks": [{{"tool": str, "target": str, "reason": str}}]}}"""

        try:
            raw = llm.invoke(prompt).content
            clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            checks = json.loads(clean).get("checks", [])
        except Exception as e:
            all_evidences.append(
                _evidence(
                    goal=f"LLM investigation plan for: {dim_id}",
                    found=False,
                    location="rubric/week2_rubric.json",
                    rationale=f"LLM plan generation failed: {e!r}",
                    confidence=0.0,
                )
            )
            continue

        all_evidences.append(
            _evidence(
                goal=f"LLM investigation plan for: {dim_id}",
                found=True,
                location="rubric/week2_rubric.json",
                rationale=f"LLM dynamically selected {len(checks)} checks for this criterion.",
                content=json.dumps({"checks": checks}, indent=2),
                confidence=0.8,
            )
        )

        if not checks:
            all_evidences.append(
                _evidence(
                    goal=f"Dynamic execution skipped: no checks selected for {dim_id}",
                    found=False,
                    location="rubric/week2_rubric.json",
                    rationale="LLM produced an empty investigation plan; required baseline checks were still executed.",
                    confidence=1.0,
                )
            )
            continue

        for check in checks:
            tool_name = check.get("tool", "")
            target = check.get("target", "")
            reason = check.get("reason", "")
            signature = f"{tool_name}|{target}"
            if signature in executed_signatures:
                continue
            result = None
            found = False
            location = target or repo_path
            try:
                found, result, location = _execute_tool_check(tool_name, target, repo_path)
            except Exception as e:
                found = False
                result = f"Execution error: {e!r}"

            rationale_text = reason
            content_text = str(result)[:500] if result else None
            confidence_value = 0.75
            if tool_name == "git_log" and found and result:
                try:
                    parsed = json.loads(str(result))
                    gsum = parsed.get("git_forensic_summary", {})
                    if isinstance(gsum, dict):
                        rationale_text = str(gsum.get("rationale", rationale_text))
                        location = str(gsum.get("location", location))
                        confidence_value = float(gsum.get("confidence", confidence_value))
                    content_text = json.dumps(parsed)[:1200]
                except Exception:
                    pass

            all_evidences.append(
                _evidence(
                    goal=f"Dynamic check [{dim_id}]: {tool_name} on '{target}'",
                    found=found,
                    location=location,
                    rationale=rationale_text,
                    content=content_text,
                    confidence=confidence_value,
                    dimension_id=dim_id if tool_name == "git_log" else "",
                )
            )

    return all_evidences


def repo_investigator(state: AgentState) -> Dict[str, object]:
    repo_url = state.get("repo_url", "")

    rubric = load_rubric(state.get("rubric_path") or "rubric/week2_rubric.json")
    # FIXED: C12
    repo_dimensions = get_dimensions_for(rubric, "github_repo")

    def _analyze_repo(repo_path: str):
        evidences: List[Evidence] = []

        evidences.append(
            _evidence(
                goal="Clone repository in sandbox",
                found=True,
                location=repo_url,
                rationale="Repository successfully cloned into temporary sandbox directory.",
                content=f"Sandbox path: {repo_path}",
            )
        )

        evidences.append(
            _evidence(
                goal="Filter rubric dimensions for github_repo detective",
                found=True,
                location=state.get("rubric_path", "rubric/week2_rubric.json"),
                rationale="Filtered rubric by target_artifact before repo analysis.",
                content=json.dumps([d.get("id", "") for d in repo_dimensions]),
            )
        )

        expected_paths = [
            "src/state.py",
            "src/graph.py",
            "src/nodes/detectives.py",
            "src/nodes/judges.py",
            "src/nodes/justice.py",
            "src/nodes/aggregator.py",
            "src/nodes/skip.py",
            "src/tools/repo_tools.py",
            "rubric/week2_rubric.json",
            "README.md",
            "requirements.txt",
            "main.py",
        ]

        for rel in expected_paths:
            full = os.path.join(repo_path, rel)
            exists = os.path.exists(full)
            snippet = None
            if exists and os.path.isfile(full):
                try:
                    with open(full, "r", encoding="utf-8", errors="ignore") as f:
                        snippet = f.read(2000)
                except Exception:
                    snippet = None

            evidences.append(
                _evidence(
                    goal=f"Check whether file exists: {rel}",
                    found=exists,
                    location=rel,
                    rationale="Filesystem existence check inside sandbox clone.",
                    content=snippet,
                )
            )

        hist_res = extract_git_history(repo_path)
        if not hist_res.ok:
            evidences.append(
                _evidence(
                    goal="Extract git history (commit list)",
                    found=False,
                    location=os.path.join(repo_path, ".git"),
                    rationale=hist_res.error or "git history extraction failed.",
                )
            )
        else:
            count = int(hist_res.data.get("count", 0))
            commits = hist_res.data.get("commits", [])
            classifier = _classify_git_progression(commits)
            summary = _build_git_forensic_summary(commits, classifier)
            evidences.append(
                _evidence(
                    goal="Extract git history (commit list)",
                    found=True,
                    location=str(summary["location"]),
                    rationale=str(summary["rationale"]),
                    content=json.dumps(
                        {
                            "commit_count": count,
                            "commits_preview": commits[:10],
                            "git_progression_classifier": classifier,
                            "git_forensic_summary": summary,
                        }
                    )[:1400],
                    confidence=float(summary["confidence"]),
                    dimension_id="git_forensic_analysis",
                )
            )

        graph_file = os.path.join(repo_path, "src", "graph.py")
        if os.path.exists(graph_file):
            graph_res = analyze_langgraph_graph_py(graph_file)
            evidences.append(
                _evidence(
                    goal="AST analysis of src/graph.py (StateGraph/add_node/add_edge + edge patterns)",
                    found=graph_res.ok,
                    location="src/graph.py",
                    rationale=(
                        "Parsed src/graph.py using AST traversal."
                        if graph_res.ok
                        else graph_res.error or "AST graph analysis failed."
                    ),
                    content=str(graph_res.data)[:900] if graph_res.ok else None,
                )
            )
            graph_src_snippet = None
            try:
                with open(graph_file, "r", encoding="utf-8", errors="ignore") as f:
                    graph_src_snippet = f.read(3000)
            except Exception:
                graph_src_snippet = None
            evidences.append(
                _evidence(
                    goal="Extract full graph wiring code snippet from src/graph.py",
                    found=bool(graph_src_snippet),
                    location="src/graph.py",
                    rationale="Read raw source snippet directly from graph.py for wiring verification.",
                    content=graph_src_snippet,
                )
            )

        state_file = os.path.join(repo_path, "src", "state.py")
        if os.path.exists(state_file):
            state_res = find_typed_state_definitions(state_file)
            evidences.append(
                _evidence(
                    goal="AST analysis of src/state.py (BaseModel/TypedDict/Annotated reducers)",
                    found=state_res.ok,
                    location="src/state.py",
                    rationale=(
                        "Parsed src/state.py using AST traversal."
                        if state_res.ok
                        else state_res.error or "AST state analysis failed."
                    ),
                    content=str(state_res.data)[:900] if state_res.ok else None,
                )
            )
            reducer_snip = _extract_reducer_snippet(state_file)
            evidences.append(
                _evidence(
                    goal="Extract reducer wiring snippet from src/state.py (operator.ior for evidences, operator.add for opinions)",
                    found=bool(reducer_snip),
                    location="src/state.py",
                    rationale="Extracted relevant lines showing reducer assignments for parallel-safe merges.",
                    content=reducer_snip,
                )
            )

        tools_file = os.path.join(repo_path, "src", "tools", "repo_tools.py")
        if os.path.exists(tools_file):
            try:
                with open(tools_file, "r", encoding="utf-8", errors="ignore") as f:
                    tools_src = f.read()

                has_os_system = "os.system(" in tools_src
                has_tempdir = "TemporaryDirectory(" in tools_src
                has_pdf_reader = ("PdfReader" in tools_src) or ("PyPDF2" in tools_src)
                has_chunking = ("chunk_size" in tools_src) and ("chunk_overlap" in tools_src)
                has_query_fn = "query_pdf_chunks" in tools_src

                evidences.append(
                    _evidence(
                        goal="Check tool safety: ensure no os.system() usage in src/tools/repo_tools.py",
                        found=not has_os_system,
                        location="src/tools/repo_tools.py",
                        rationale="Direct source scan for forbidden os.system() call signature.",
                        content="os.system() NOT found" if not has_os_system else "os.system() FOUND",
                    )
                )
                evidences.append(
                    _evidence(
                        goal="Verify git clone uses tempfile.TemporaryDirectory()",
                        found=has_tempdir,
                        location="src/tools/repo_tools.py",
                        rationale="Direct source scan for TemporaryDirectory usage.",
                    )
                )
                evidences.append(
                    _evidence(
                        goal="Verify PDF ingestion capability exists (PdfReader/PyPDF2 usage present)",
                        found=has_pdf_reader,
                        location="src/tools/repo_tools.py",
                        rationale="Direct source scan for PDF reader usage.",
                    )
                )
                evidences.append(
                    _evidence(
                        goal="Verify PDF chunking + query functions exist (RAG-lite interface)",
                        found=bool(has_chunking and has_query_fn),
                        location="src/tools/repo_tools.py",
                        rationale="Direct source scan for chunk_size/chunk_overlap and query_pdf_chunks function name.",
                        content=f"chunking={has_chunking}, query_fn={has_query_fn}",
                    )
                )
            except Exception as e:
                evidences.append(
                    _evidence(
                        goal="Check tool safety/capabilities in src/tools/repo_tools.py",
                        found=False,
                        location="src/tools/repo_tools.py",
                        rationale=f"Failed to read tools source for verification: {e!r}",
                    )
                )

        dynamic_evidences = _run_dynamic_plan_checks(state, repo_path, repo_dimensions)
        return ToolResult(
            ok=True,
            data={
                "repo_path": repo_path,
                "evidences": {
                    "repo": evidences,
                    "dynamic_plan": dynamic_evidences,
                },
            },
        )

    # FIXED: C6
    clone_res = clone_repo_sandboxed(repo_url, analyzer=_analyze_repo)
    if not clone_res.ok:
        return {
            "repo_path": "",
            "evidences": {
                "repo": [
                    _evidence(
                        goal="Clone repository in sandbox",
                        found=False,
                        location=repo_url or "repo_url missing",
                        rationale=clone_res.error or "Clone failed for unknown reasons.",
                    ),
                    _evidence(
                        goal="Run blocker: repo evidence unavailable",
                        found=False,
                        location=repo_url or "repo_url missing",
                        rationale="Repository clone failed, so repo-dependent checks cannot execute.",
                    ),
                ]
            }
        }
    return clone_res.data


def _chunk_location(ch: Dict[str, object]) -> str:
    return f"page {ch.get('page', '?')} / chunk {ch.get('chunk_id', '?')}"


def _windowed_chunk_context(chunks: List[Dict[str, object]], index: int) -> str:
    start = max(0, index - 1)
    end = min(len(chunks), index + 2)
    parts: List[str] = []
    for i in range(start, end):
        parts.append(str(chunks[i].get("text", "") or ""))
    return "\n".join(parts).strip()


def _is_substantive_explanation(text: str, term: str) -> bool:
    lowered = (text or "").lower()
    if not lowered or term.lower() not in lowered:
        return False
    if len(lowered) < 140:
        return False
    cues = [
        "because",
        "therefore",
        "implemented",
        "implementation",
        "via",
        "using",
        "through",
        "ensures",
        "so that",
        "architecture",
        "stategraph",
        "parallel",
        "fan-in",
        "fan out",
        "synchronization",
        "judge",
    ]
    cue_hits = sum(1 for cue in cues if cue in lowered)
    sentence_like = lowered.count(".") + lowered.count(":") + lowered.count(";")
    return cue_hits >= 2 and sentence_like >= 2


def _term_regex(term: str) -> re.Pattern[str]:
    escaped = re.escape(term.lower())
    escaped = escaped.replace(r"\-", "[- ]")
    return re.compile(rf"\b{escaped}\b", flags=re.IGNORECASE)


def _find_term_occurrences(pdf_index: Dict[str, object], term: str) -> List[Dict[str, object]]:
    chunks = pdf_index.get("chunks", [])
    if not isinstance(chunks, list):
        return []
    pattern = _term_regex(term)
    hits: List[Dict[str, object]] = []
    for idx, ch in enumerate(chunks):
        text = str(ch.get("text", "") or "")
        if not text:
            continue
        if pattern.search(text.lower()) is None:
            continue
        hits.append(
            {
                "chunk": ch,
                "context_window": _windowed_chunk_context(chunks, idx),
                "location": _chunk_location(ch),
            }
        )
    return hits


def _extract_pdf_claimed_paths(pdf_index: Dict[str, object]) -> Dict[str, List[str]]:
    chunks = pdf_index.get("chunks", [])
    if not isinstance(chunks, list):
        return {}

    path_pattern = re.compile(
        r"\b(?:src|tests|rubric|audit)(?:[\\/][A-Za-z0-9._-]+)+\b|"
        r"\b(?:README\.md|main\.py|pyproject\.toml|requirements(?:\.txt|\.lock)|\.env(?:\.example)?)\b",
        flags=re.IGNORECASE,
    )
    claims: Dict[str, List[str]] = {}
    for ch in chunks:
        text = str(ch.get("text", "") or "")
        if not text:
            continue
        location = _chunk_location(ch)
        for raw in path_pattern.findall(text):
            path = str(raw or "").strip().strip("`'\".,;:()[]{}")
            path = path.replace("\\", "/")
            if not path:
                continue
            claims.setdefault(path, [])
            if location not in claims[path]:
                claims[path].append(location)
    return claims


def _build_repo_file_index(repo_path: str) -> set[str]:
    index: set[str] = set()
    if not repo_path or not os.path.isdir(repo_path):
        return index
    root = Path(repo_path)
    for p in root.rglob("*"):
        if p.is_file():
            try:
                rel = p.relative_to(root).as_posix()
                index.add(rel.lower())
            except Exception:
                continue
    return index


def _extract_json_object(raw: object) -> Dict[str, object]:
    text = str(raw or "").strip()
    if not text:
        return {}
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}


def doc_analyst(state: AgentState) -> Dict[str, Dict[str, List[Evidence]]]:
    pdf_path = state.get("pdf_path", "")
    evidences: List[Evidence] = []

    rubric = load_rubric(state.get("rubric_path") or "rubric/week2_rubric.json")
    # FIXED: C12
    doc_dimensions = get_dimensions_for(rubric, "pdf_report")
    evidences.append(
        _evidence(
            goal="Filter rubric dimensions for pdf_report detective",
            found=True,
            location=state.get("rubric_path", "rubric/week2_rubric.json"),
            rationale="Filtered rubric by target_artifact before PDF analysis.",
            content=json.dumps([d.get("id", "") for d in doc_dimensions]),
        )
    )

    if not pdf_path:
        evidences.append(
            _evidence(
                goal="Load PDF report",
                found=False,
                location="pdf_path missing",
                rationale="No pdf_path provided in AgentState.",
            )
        )
        return {"evidences": {"doc": evidences}}

    ingest_res = ingest_pdf_for_query(pdf_path)
    if not ingest_res.ok:
        evidences.append(
            _evidence(
                goal="Ingest PDF into queryable chunks (RAG-lite)",
                found=False,
                location=pdf_path,
                rationale=ingest_res.error or "PDF ingestion failed.",
            )
        )
        return {"evidences": {"doc": evidences}}

    pdf_index = ingest_res.data
    evidences.append(
        _evidence(
            goal="Ingest PDF into queryable chunks (RAG-lite)",
            found=True,
            location=pdf_path,
            rationale="PDF opened and chunked successfully.",
            content=f"Indexed pages={pdf_index.get('page_count_indexed')} chunks={pdf_index.get('chunk_count')}",
        )
    )

    required_terms = [
        "Dialectical Synthesis",
        "Metacognition",
        "Fan-In",
        "Fan-Out",
        "State Synchronization",
    ]

    keyword_dropping_terms: List[str] = []
    for term in required_terms:
        occurrences = _find_term_occurrences(pdf_index, term)
        found = len(occurrences) > 0
        locations = [str(h.get("location", "")) for h in occurrences[:5]]
        location_text = ", ".join(locations) if locations else pdf_path
        evidences.append(
            _evidence(
                goal=f"theoretical_depth: locate required term '{term}'",
                found=found,
                location=location_text,
                rationale="Direct chunk scan for rubric-required theoretical terms.",
                content=f"occurrence_count={len(occurrences)}",
                confidence=0.95 if found else 0.85,
                dimension_id="theoretical_depth",
            )
        )

        substantive_hits = 0
        previews: List[str] = []
        for hit in occurrences:
            context_window = str(hit.get("context_window", "") or "")
            if _is_substantive_explanation(context_window, term):
                substantive_hits += 1
                previews.append(context_window[:240])
            if len(previews) >= 2:
                break

        has_substantive = substantive_hits > 0
        evidences.append(
            _evidence(
                goal=f"theoretical_depth: verify substantive explanation for '{term}'",
                found=has_substantive,
                location=location_text,
                rationale="Context-window check around each term occurrence to detect explanatory depth vs keyword-only mention.",
                content=(
                    f"substantive_hits={substantive_hits}/{len(occurrences)}"
                    + (f"\npreview={previews[0]}" if previews else "")
                ),
                confidence=0.86,
                dimension_id="theoretical_depth",
            )
        )

        if found and not has_substantive:
            keyword_dropping_terms.append(term)
            evidences.append(
                _evidence(
                    goal=f"theoretical_depth: Keyword Dropping detected for '{term}'",
                    found=True,
                    location=location_text,
                    rationale="Term appears but nearby chunk context lacks substantive implementation explanation.",
                    content=f"flag=Keyword Dropping | term={term} | occurrences={len(occurrences)}",
                    confidence=0.82,
                    dimension_id="theoretical_depth",
                )
            )

    if keyword_dropping_terms:
        evidences.append(
            _evidence(
                goal="theoretical_depth: keyword-dropping summary",
                found=True,
                location=pdf_path,
                rationale="Aggregated from per-term context-window checks.",
                content=json.dumps({"keyword_dropping_terms": keyword_dropping_terms}),
                confidence=0.84,
                dimension_id="theoretical_depth",
            )
        )

    claimed_paths = _extract_pdf_claimed_paths(pdf_index)
    repo_path = (state.get("repo_path") or "").strip()
    repo_index = _build_repo_file_index(repo_path)
    normalized_claims = sorted(claimed_paths.keys(), key=lambda x: x.lower())
    verified: List[Dict[str, object]] = []
    hallucinated: List[Dict[str, object]] = []
    for claim in normalized_claims:
        in_repo = claim.lower() in repo_index
        record = {"path": claim, "locations": claimed_paths.get(claim, [])[:4]}
        if in_repo:
            verified.append(record)
        else:
            hallucinated.append(record)

    evidences.append(
        _evidence(
            goal="report_accuracy: extract file paths mentioned in PDF",
            found=bool(normalized_claims),
            location=pdf_path,
            rationale="Regex extraction over all indexed PDF chunks for repository-like path claims.",
            content=json.dumps(
                {
                    "path_count": len(normalized_claims),
                    "paths_preview": normalized_claims[:25],
                }
            ),
            confidence=0.90,
            dimension_id="report_accuracy",
        )
    )

    evidences.append(
        _evidence(
            goal="report_accuracy: Verified Paths list",
            found=bool(verified),
            location=(repo_path or "repo_path missing"),
            rationale="Cross-checked PDF-mentioned paths against repo filesystem index.",
            content=json.dumps({"verified_paths": verified[:40]}),
            confidence=0.92 if repo_index else 0.55,
            dimension_id="report_accuracy",
        )
    )
    evidences.append(
        _evidence(
            goal="report_accuracy: Hallucinated Paths list",
            found=bool(hallucinated),
            location=(repo_path or "repo_path missing"),
            rationale="Paths mentioned in PDF but absent from repository index are treated as hallucinated claims.",
            content=json.dumps({"hallucinated_paths": hallucinated[:40]}),
            confidence=0.92 if repo_index else 0.55,
            dimension_id="report_accuracy",
        )
    )

    return {"evidences": {"doc": evidences}}


def extract_images_from_pdf(pdf_path: str) -> List[Dict[str, object]]:
    try:
        import fitz
    except Exception:
        return []

    if not pdf_path or not os.path.isfile(pdf_path):
        return []

    images: List[Dict[str, object]] = []
    doc = fitz.open(pdf_path)
    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get("image", b"")
                if image_bytes:
                    images.append(
                        {
                            "page": page_idx + 1,
                            "xref": xref,
                            "ext": str(base_image.get("ext", "png") or "png"),
                            "b64": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    )
    finally:
        doc.close()

    return images


def vision_inspector_node(state: AgentState) -> Dict[str, Dict[str, List[Evidence]]]:
    # FIXED: C3+O1
    pdf_path = state.get("pdf_path", "")
    if not pdf_path:
        return {
            "evidences": {
                "vision_analysis": [
                    _evidence(
                        goal="Analyze architecture diagram from PDF images",
                        found=False,
                        location="pdf_path missing",
                        rationale="No PDF provided for vision inspection.",
                        confidence=0.0,
                    )
                ]
            }
        }

    images = extract_images_from_pdf(pdf_path)
    if not images:
        return {
            "evidences": {
                "vision_analysis": [
                    _evidence(
                        goal="Analyze architecture diagram from PDF images",
                        found=False,
                        location=pdf_path,
                        rationale="No extractable images found in PDF.",
                        confidence=0.0,
                        dimension_id="swarm_visual",
                    )
                ]
            }
        }

    repo_path = (state.get("repo_path") or "").strip()
    graph_truth = {"fan_out_from_start": None, "fan_in_to_node": None}
    graph_file = os.path.join(repo_path, "src", "graph.py") if repo_path else ""
    if graph_file and os.path.isfile(graph_file):
        graph_res = analyze_langgraph_graph_py(graph_file)
        if graph_res.ok and isinstance(graph_res.data, dict):
            graph_truth["fan_out_from_start"] = graph_res.data.get("fan_out_from_start")
            graph_truth["fan_in_to_node"] = graph_res.data.get("fan_in_to_node")

    llm = ChatOllama(
        model=os.getenv("VISION_MODEL", "qwen3-vl:4b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        format="json",
        temperature=0.0,
    )

    evidences: List[Evidence] = []
    parallel_match_found = False
    misleading_locations: List[str] = []

    for img in images[:5]:
        page = img.get("page")
        xref = img.get("xref")
        ext = str(img.get("ext", "png") or "png")
        b64 = str(img.get("b64", "") or "")
        location = f"{pdf_path}#page={page},image_xref={xref}"
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
            {
                "type": "text",
                "text": (
                    'Reply JSON only with keys: '
                    '{"diagram_type": str, "shows_parallel_detectives": bool, '
                    '"shows_parallel_judges": bool, "shows_fan_in": bool, '
                    '"is_linear_pipeline": bool, "contradicts_parallel_architecture": bool, '
                    '"rationale": str}'
                ),
            },
        ]

        parsed: Dict[str, object] = {}
        raw_text = ""
        try:
            raw = llm.invoke([HumanMessage(content=content)]).content
            raw_text = str(raw)
            parsed = _extract_json_object(raw)
        except Exception as e:
            raw_text = f"vision_error={e!r}"

        if not parsed:
            evidences.append(
                _evidence(
                    goal="swarm_visual: classify extracted diagram image",
                    found=False,
                    location=location,
                    rationale="Vision classification failed or returned non-JSON output.",
                    content=raw_text[:900] if raw_text else None,
                    confidence=0.35,
                    dimension_id="swarm_visual",
                )
            )
            continue

        pd = bool(parsed.get("shows_parallel_detectives", False))
        pj = bool(parsed.get("shows_parallel_judges", False))
        fi = bool(parsed.get("shows_fan_in", False))
        linear = bool(parsed.get("is_linear_pipeline", False))
        contradict = bool(parsed.get("contradicts_parallel_architecture", False))
        diag_type = str(parsed.get("diagram_type", "") or "unknown")
        rationale = str(parsed.get("rationale", "") or "No rationale provided.")

        if pd and pj and fi and not linear and not contradict:
            parallel_match_found = True
        if linear or contradict:
            misleading_locations.append(location)

        evidences.append(
            _evidence(
                goal="swarm_visual: classify extracted diagram image",
                found=True,
                location=location,
                rationale="Vision model classified diagram structure and architecture semantics.",
                content=json.dumps(
                    {
                        "diagram_type": diag_type,
                        "shows_parallel_detectives": pd,
                        "shows_parallel_judges": pj,
                        "shows_fan_in": fi,
                        "is_linear_pipeline": linear,
                        "contradicts_parallel_architecture": contradict,
                        "rationale": rationale,
                        "graph_truth_reference": graph_truth,
                    }
                )[:1200],
                confidence=0.72,
                dimension_id="swarm_visual",
            )
        )

    evidences.append(
        _evidence(
            goal="swarm_visual: verify fan-out/fan-in architecture depiction",
            found=parallel_match_found,
            location=pdf_path,
            rationale="Aggregated image classifications checked for parallel detectives + fan-in + parallel judges.",
            content=f"parallel_match_found={parallel_match_found} | graph_truth={json.dumps(graph_truth)}",
            confidence=0.78 if evidences else 0.5,
            dimension_id="swarm_visual",
        )
    )

    if misleading_locations:
        evidences.append(
            _evidence(
                goal="swarm_visual: Misleading Architecture Visual detected",
                found=True,
                location="; ".join(misleading_locations[:6]),
                rationale="At least one diagram appears linear or contradictory to expected graph parallelism.",
                content=json.dumps({"misleading_locations": misleading_locations[:20]}),
                confidence=0.80,
                dimension_id="swarm_visual",
            )
        )

    return {
        "evidences": {"vision_analysis": evidences}
    }


def dynamic_investigator_node(state: AgentState) -> Dict[str, object]:
    return {}
