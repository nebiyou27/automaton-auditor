# src/nodes/detectives.py
# =======================
# Detective layer: collect FACTS only.
# No scoring. No judgment. Only structured Evidence objects.

from __future__ import annotations

import base64
import json
import os
from typing import Dict, List

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
) -> Evidence:
    return Evidence(
        goal=goal,
        found=found,
        location=location,
        rationale=rationale,
        content=content,
        confidence=confidence,
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


def repo_investigator(state: AgentState) -> Dict[str, Dict[str, List[Evidence]]]:
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

        return ToolResult(ok=True, data={"evidences": {"repo": evidences}})

    # FIXED: C6
    clone_res = clone_repo_sandboxed(repo_url, analyzer=_analyze_repo)
    if not clone_res.ok:
        return {
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


def extract_images_from_pdf(pdf_path: str) -> List[str]:
    try:
        import fitz
    except Exception:
        return []

    if not pdf_path or not os.path.isfile(pdf_path):
        return []

    images_b64: List[str] = []
    doc = fitz.open(pdf_path)
    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get("image", b"")
                if image_bytes:
                    images_b64.append(base64.b64encode(image_bytes).decode("ascii"))
    finally:
        doc.close()

    return images_b64


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

    images_b64 = extract_images_from_pdf(pdf_path)
    if not images_b64:
        return {
            "evidences": {
                "vision_analysis": [
                    _evidence(
                        goal="Analyze architecture diagram from PDF images",
                        found=False,
                        location=pdf_path,
                        rationale="No extractable images found in PDF.",
                        confidence=0.0,
                    )
                ]
            }
        }

    llm = ChatOllama(
        model=os.getenv("VISION_MODEL", "qwen3-vl:4b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        format="json",
        temperature=0.0,
    )

    content = []
    for b64 in images_b64[:3]:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })
    content.append({
        "type": "text",
        "text": 'Reply JSON only: {"parallel_detectives": bool, "fan_in_present": bool, "parallel_judges": bool, "diagram_type": str, "description": str}'
    })

    try:
        raw = llm.invoke([HumanMessage(content=content)]).content
    except Exception as e:
        return {
            "evidences": {
                "vision_analysis": [
                    _evidence(
                        goal="Analyze architecture diagram from PDF images",
                        found=False,
                        location=pdf_path,
                        rationale=f"Vision model call failed: {e!r}",
                        confidence=0.0,
                    )
                ]
            }
        }

    return {
        "evidences": {
            "vision_analysis": [
                _evidence(
                    goal="Analyze architecture diagram from PDF images",
                    found=True,
                    location=pdf_path,
                    rationale="Vision model analyzed extracted PDF images.",
                    content=str(raw)[:1200],
                    confidence=0.6,
                )
            ]
        }
    }


def dynamic_investigator_node(state: AgentState) -> Dict[str, Dict[str, List[Evidence]]]:
    """LLM reads each rubric criterion, selects tools, then executes them."""
    import re, json
    from pathlib import Path
    from langchain_ollama import ChatOllama
    from src.tools.rubric_utils import load_rubric, get_dimensions_for
    from src.tools.repo_tools import analyze_langgraph_graph_py, extract_git_history

    rubric = load_rubric(state.get("rubric_path", "rubric/week2_rubric.json"))
    dimensions = get_dimensions_for(rubric, "github_repo")

    # Get repo_path from clone evidence
    repo_path = None
    for ev in state.get("evidences", {}).get("repo", []):
        goal = getattr(ev, "goal", "") or ""
        content = getattr(ev, "content", "") or ""
        if "clone" in goal.lower() and getattr(ev, "found", False):
            for part in content.split():
                if os.path.isdir(part):
                    repo_path = part
                    break

    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "deepseek-r1:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
        format="json"
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
            all_evidences.append(_evidence(
                goal=f"LLM investigation plan for: {dim['id']}",
                found=False,
                location="rubric/week2_rubric.json",
                rationale=f"LLM plan generation failed: {e!r}",
                confidence=0.0,
            ))
            continue

        # Record the plan
        all_evidences.append(_evidence(
            goal=f"LLM investigation plan for: {dim['id']}",
            found=True,
            location="rubric/week2_rubric.json",
            rationale=f"PLAN_ONLY: LLM dynamically selected {len(checks)} checks for this criterion.",
            content=json.dumps({"plan_only": True, "checks": checks}, indent=2),
            confidence=0.8,
        ))

        # Execute each check the LLM selected
        if not repo_path:
            all_evidences.append(_evidence(
                goal=f"Dynamic execution skipped: repo unavailable for {dim['id']}",
                found=False,
                location="repo bucket",
                rationale="Repository path was not recovered from clone evidence; cannot execute plan checks.",
                confidence=1.0,
            ))
            continue

        if not checks:
            all_evidences.append(_evidence(
                goal=f"Dynamic execution skipped: no checks selected for {dim['id']}",
                found=False,
                location="rubric/week2_rubric.json",
                rationale="LLM produced an empty investigation plan; no executable checks available.",
                confidence=1.0,
            ))
            continue

        for check in checks:
            tool_name = check.get("tool", "")
            target = check.get("target", "")
            reason = check.get("reason", "")
            result = None
            found = False

            try:
                if tool_name == "check_file_exists":
                    full = os.path.join(repo_path, target)
                    found = os.path.exists(full)
                    result = f"exists={found}"

                elif tool_name == "read_file":
                    full = os.path.join(repo_path, target)
                    ok, content = _read_text_file(full, max_chars=2000)
                    found = ok
                    result = content[:500] if ok else content

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

                elif tool_name == "ast_parse":
                    full = os.path.join(repo_path, target)
                    res = analyze_langgraph_graph_py(full)
                    found = res.ok
                    result = str(res.data)[:500] if res.ok else res.error

                elif tool_name == "git_log":
                    res = extract_git_history(repo_path)
                    found = res.ok
                    result = str(res.data)[:500] if res.ok else res.error

            except Exception as e:
                found = False
                result = f"Execution error: {e!r}"

            all_evidences.append(_evidence(
                goal=f"Dynamic check [{dim['id']}]: {tool_name} on '{target}'",
                found=found,
                location=target or repo_path,
                rationale=reason,
                content=str(result)[:500] if result else None,
                confidence=0.75,
            ))

    return {"evidences": {"dynamic_plan": all_evidences}}
