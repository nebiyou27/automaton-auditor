# src/nodes/detectives.py
# =======================
# Detective layer: collect FACTS only.
# No scoring. No judgment. Only structured Evidence objects.

from __future__ import annotations

import os
from typing import Dict, List

from src.state import AgentState, Evidence
from src.tools.repo_tools import (
    analyze_langgraph_graph_py,
    clone_repo_sandboxed,
    extract_git_history,
    find_typed_state_definitions,
    ingest_pdf_for_query,
    query_pdf_chunks,
)


def _evidence(
    *,
    goal: str,
    found: bool,
    location: str,
    rationale: str,
    content: str | None = None,
    confidence: float = 1.0,
) -> Evidence:
    """
    Evidence helper.
    Keep this factual: 'goal' states what we checked, not what we concluded.
    """
    return Evidence(
        goal=goal,
        found=found,
        location=location,
        rationale=rationale,
        content=content,
        confidence=confidence,
    )


def repo_investigator(state: AgentState) -> Dict[str, Dict[str, List[Evidence]]]:
    """
    RepoInvestigator (Detective):
    - Clone repo safely in sandbox
    - Check for expected files
    - Extract git history facts (count + preview)
    - Run AST structural checks on src/graph.py and src/state.py (if present)

    Returns ONLY factual Evidence objects (no scoring, no verdict language).
    """
    repo_url = state.get("repo_url", "")
    evidences: List[Evidence] = []

    # --- Clone repo safely ---
    clone_res = clone_repo_sandboxed(repo_url)
    if not clone_res.ok:
        evidences.append(
            _evidence(
                goal="Clone repository in sandbox",
                found=False,
                location=repo_url or "repo_url missing",
                rationale=clone_res.error or "Clone failed for unknown reasons.",
            )
        )
        return {"evidences": {"repo": evidences}}

    repo_path = str(clone_res.data["repo_path"])

    evidences.append(
        _evidence(
            goal="Clone repository in sandbox",
            found=True,
            location=repo_url,
            rationale="Repository successfully cloned into sandbox directory.",
            content=f"Sandbox path: {repo_path}",
        )
    )

    # --- File existence checks (facts only) ---
    expected_paths = [
        "src/state.py",
        "src/graph.py",
        "src/nodes/detectives.py",
        "src/nodes/judges.py",
        "src/nodes/justice.py",
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
                    snippet = f.read(500)
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

    # --- Git history: report facts (count + preview), no threshold verdict ---
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
        preview = "\n".join([f"{c['hash']} {c['message']}" for c in commits[:10]])

        evidences.append(
            _evidence(
                goal="Extract git history (commit list)",
                found=True,
                location=".git/log",
                rationale=f"git log extracted successfully. commit_count={count}.",
                content=preview or "(no commits found)",
            )
        )

    # --- AST structural checks: src/graph.py ---
    graph_file = os.path.join(repo_path, "src", "graph.py")
    if os.path.exists(graph_file):
        graph_res = analyze_langgraph_graph_py(graph_file)
        if graph_res.ok:
            evidences.append(
                _evidence(
                    goal="AST analysis of src/graph.py (StateGraph/add_node/add_edge + edge patterns)",
                    found=True,
                    location="src/graph.py",
                    rationale="Parsed src/graph.py using AST traversal.",
                    content=str(graph_res.data)[:900],
                    confidence=1.0,
                )
            )
        else:
            evidences.append(
                _evidence(
                    goal="AST analysis of src/graph.py (StateGraph/add_node/add_edge + edge patterns)",
                    found=False,
                    location="src/graph.py",
                    rationale=graph_res.error or "AST graph analysis failed.",
                )
            )
    else:
        evidences.append(
            _evidence(
                goal="AST analysis of src/graph.py (StateGraph/add_node/add_edge + edge patterns)",
                found=False,
                location="src/graph.py",
                rationale="File missing in repository; AST analysis skipped.",
            )
        )

    # --- AST structural checks: src/state.py ---
    state_file = os.path.join(repo_path, "src", "state.py")
    if os.path.exists(state_file):
        state_res = find_typed_state_definitions(state_file)
        if state_res.ok:
            evidences.append(
                _evidence(
                    goal="AST analysis of src/state.py (BaseModel/TypedDict/Annotated reducers)",
                    found=True,
                    location="src/state.py",
                    rationale="Parsed src/state.py using AST traversal.",
                    content=str(state_res.data)[:900],
                    confidence=1.0,
                )
            )
        else:
            evidences.append(
                _evidence(
                    goal="AST analysis of src/state.py (BaseModel/TypedDict/Annotated reducers)",
                    found=False,
                    location="src/state.py",
                    rationale=state_res.error or "AST state analysis failed.",
                )
            )
    else:
        evidences.append(
            _evidence(
                goal="AST analysis of src/state.py (BaseModel/TypedDict/Annotated reducers)",
                found=False,
                location="src/state.py",
                rationale="File missing in repository; AST analysis skipped.",
            )
        )

    return {"evidences": {"repo": evidences}}


def doc_analyst(state: AgentState) -> Dict[str, Dict[str, List[Evidence]]]:
    """
    DocAnalyst (Detective):
    - Ingest PDF into chunked index (RAG-lite)
    - Query for key concepts and return citations (chunk_id/page)
    - Facts only (found/not found + where)

    Returns ONLY Evidence objects.
    """
    pdf_path = state.get("pdf_path", "")
    evidences: List[Evidence] = []

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

    key_concepts = [
        "Dialectical Synthesis",
        "Metacognition",
        "Fan-In",
        "Fan-Out",
        "State Synchronization",
        "LangGraph",
    ]

    for concept in key_concepts:
        qres = query_pdf_chunks(pdf_index, concept, top_k=3)
        if not qres.ok:
            evidences.append(
                _evidence(
                    goal=f"Query PDF for concept occurrence: {concept}",
                    found=False,
                    location=pdf_path,
                    rationale=qres.error or "Query failed.",
                )
            )
            continue

        matches = qres.data.get("matches", [])
        found = len(matches) > 0

        snippet = None
        if found:
            cites = []
            for m in matches:
                cites.append(f"{m.get('chunk_id')} (page {m.get('page')})")
            snippet = " | ".join(cites) + "\n\n" + (matches[0].get("text_preview") or "")

        evidences.append(
            _evidence(
                goal=f"Check whether concept appears in report: {concept}",
                found=found,
                location=pdf_path,
                rationale="Lexical query over chunked PDF index; returns matching chunk_ids/pages.",
                content=snippet,
                confidence=1.0,
            )
        )

    return {"evidences": {"doc": evidences}}