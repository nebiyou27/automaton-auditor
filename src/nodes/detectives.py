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


def _extract_reducer_snippet(state_file_path: str, max_lines: int = 25) -> str | None:
    """
    Pull only the lines that prove reducers are wired correctly.
    This makes judges cite the *exact* reducer assignments (ior for evidences, add for opinions).
    """
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


def repo_investigator(state: AgentState) -> Dict[str, Dict[str, List[Evidence]]]:
    """
    RepoInvestigator (Detective):
    - Clone repo safely in sandbox
    - Check for expected files
    - Extract git history facts (count + preview)
    - Run AST structural checks on src/graph.py and src/state.py (if present)
    - Provide line-level reducer snippet evidence (operator.ior vs operator.add)
    - Provide explicit tool-safety/tool-capability evidence (no os.system, PDF chunking, tempfile usage)

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

        # --- Reducer snippet (line-level proof) ---
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

    else:
        evidences.append(
            _evidence(
                goal="AST analysis of src/state.py (BaseModel/TypedDict/Annotated reducers)",
                found=False,
                location="src/state.py",
                rationale="File missing in repository; AST analysis skipped.",
            )
        )
        evidences.append(
            _evidence(
                goal="Extract reducer wiring snippet from src/state.py (operator.ior for evidences, operator.add for opinions)",
                found=False,
                location="src/state.py",
                rationale="File missing in repository; reducer snippet extraction skipped.",
            )
        )

    # --- Tool safety & capability checks (facts) ---
    tools_file = os.path.join(repo_path, "src", "tools", "repo_tools.py")
    if os.path.exists(tools_file):
        try:
            with open(tools_file, "r", encoding="utf-8", errors="ignore") as f:
                tools_src = f.read()

            has_os_system = "os.system(" in tools_src
            has_tempfile = "tempfile" in tools_src
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
                    goal="Verify tool sandboxing signals exist (tempfile usage in clone tooling)",
                    found=has_tempfile,
                    location="src/tools/repo_tools.py",
                    rationale="Direct source scan for tempfile usage indicating sandboxed operations.",
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
    else:
        evidences.append(
            _evidence(
                goal="Check tool safety/capabilities in src/tools/repo_tools.py",
                found=False,
                location="src/tools/repo_tools.py",
                rationale="Tools file missing in repository; tool verification skipped.",
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
            )
        )

    return {"evidences": {"doc": evidences}}