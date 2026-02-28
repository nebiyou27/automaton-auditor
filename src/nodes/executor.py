from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from src.nodes.detectives import extract_images_from_pdf
from src.state import AgentState, Evidence, ToolCall, ToolRunMetadata
from src.tools.repo_tools import (
    ToolResult,
    analyze_langgraph_graph_py,
    clone_repo_sandboxed,
    extract_git_history,
    find_typed_state_definitions,
    ingest_pdf_for_query,
    query_pdf_chunks,
)


def _ollama_timeout_s() -> int:
    raw = str(os.getenv("OLLAMA_TIMEOUT_S", "45")).strip()
    try:
        value = int(raw)
    except Exception:
        value = 45
    return value if value > 0 else 45


def _ollama_num_predict() -> int:
    raw = str(os.getenv("OLLAMA_NUM_PREDICT", "256")).strip()
    try:
        value = int(raw)
    except Exception:
        value = 256
    return value if value > 0 else 256


def _evidence(
    *,
    dimension_id: str,
    goal: str,
    found: bool,
    location: str,
    rationale: str,
    content: str | None = None,
    confidence: float = 1.0,
) -> Evidence:
    return Evidence(
        dimension_id=dimension_id,
        goal=goal,
        found=found,
        location=location,
        rationale=rationale,
        content=content,
        confidence=confidence,
    )


def _serialize_size(value: object) -> int:
    try:
        return len(json.dumps(value, default=str))
    except Exception:
        return len(str(value))


def _as_tool_call(item: object) -> ToolCall | None:
    if isinstance(item, ToolCall):
        return item
    if isinstance(item, dict):
        try:
            return ToolCall(**item)
        except Exception:
            return None
    return None


def _tool_result(ok: bool, data: dict | None = None, error: str | None = None) -> ToolResult:
    return ToolResult(ok=ok, data=data, error=error)


def _resolve_pdf_path(state: AgentState, call: ToolCall) -> str:
    path = str(call.args.get("pdf_path", "")).strip()
    if path == "$state.pdf_path":
        path = str(state.get("pdf_path", "") or "").strip()
    if not path:
        path = str(state.get("pdf_path", "") or "").strip()
    return path


def _run_with_repo(
    state: AgentState,
    call: ToolCall,
    run_fn,
) -> ToolResult:
    repo_path = str(call.args.get("repo_path", "")).strip()
    if repo_path and os.path.isdir(repo_path):
        return run_fn(repo_path)

    repo_url = str(call.args.get("repo_url", "")).strip() or str(state.get("repo_url", "")).strip()
    if not repo_url:
        return _tool_result(False, error="repo_url is missing for repo-scoped tool.")

    return clone_repo_sandboxed(repo_url, analyzer=run_fn)


def _tool_clone(state: AgentState, call: ToolCall) -> ToolResult:
    repo_url = str(call.args.get("repo_url", "")).strip() or str(state.get("repo_url", "")).strip()
    if not repo_url:
        return _tool_result(False, error="repo_url is missing.")

    def _analyzer(repo_path: str) -> ToolResult:
        return _tool_result(True, data={"repo_path": repo_path, "repo_url": repo_url})

    return clone_repo_sandboxed(repo_url, analyzer=_analyzer)


def _tool_git_log(state: AgentState, call: ToolCall) -> ToolResult:
    max_commits = int(call.args.get("max_commits", 200))

    def _run(repo_path: str) -> ToolResult:
        return extract_git_history(repo_path=repo_path, max_commits=max_commits)

    return _run_with_repo(state, call, _run)


def _tool_ast_scan(state: AgentState, call: ToolCall) -> ToolResult:
    rel_path = str(call.args.get("path", "")).strip() or "src/graph.py"
    abs_path = rel_path if os.path.isabs(rel_path) else ""

    if abs_path and os.path.exists(abs_path):
        if abs_path.endswith("state.py"):
            return find_typed_state_definitions(abs_path)
        return analyze_langgraph_graph_py(abs_path)

    def _run(repo_path: str) -> ToolResult:
        target = os.path.join(repo_path, rel_path.replace("/", os.sep))
        if not os.path.exists(target):
            return _tool_result(False, error=f"AST target not found: {rel_path}")
        if rel_path.endswith("state.py"):
            return find_typed_state_definitions(target)
        return analyze_langgraph_graph_py(target)

    return _run_with_repo(state, call, _run)


def _tool_pdf_ingest(state: AgentState, call: ToolCall) -> ToolResult:
    pdf_path = _resolve_pdf_path(state, call)
    if not pdf_path:
        return _tool_result(False, error="pdf_path is missing.")
    chunk_size = int(call.args.get("chunk_size", 1200))
    chunk_overlap = int(call.args.get("chunk_overlap", 150))
    return ingest_pdf_for_query(pdf_path=pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def _tool_pdf_query(state: AgentState, call: ToolCall, active_pdf_index: dict | None) -> ToolResult:
    query = str(call.args.get("query", "")).strip()
    top_k = int(call.args.get("top_k", 5))
    if not query:
        return _tool_result(False, error="query is missing.")

    pdf_index = active_pdf_index
    if not isinstance(pdf_index, dict):
        pdf_path = _resolve_pdf_path(state, call)
        if not pdf_path:
            return _tool_result(False, error="pdf_index unavailable and pdf_path missing.")
        ing = ingest_pdf_for_query(pdf_path=pdf_path)
        if not ing.ok:
            return ing
        pdf_index = ing.data or {}

    return query_pdf_chunks(pdf_index=pdf_index, query=query, top_k=top_k)


def _tool_pdf_image_extract(state: AgentState, call: ToolCall) -> ToolResult:
    pdf_path = _resolve_pdf_path(state, call)
    if not pdf_path:
        return _tool_result(False, error="pdf_path is missing.")
    images = extract_images_from_pdf(pdf_path)
    images_b64 = [str(img.get("b64", "")) for img in images if isinstance(img, dict) and img.get("b64")]
    return _tool_result(
        ok=len(images_b64) > 0,
        data={"pdf_path": pdf_path, "image_count": len(images_b64), "images_b64": images_b64[:3]},
        error=None if images_b64 else "No extractable images found.",
    )


def _tool_vision_analyze(state: AgentState, call: ToolCall) -> ToolResult:
    images = call.args.get("images_b64")
    images_b64: List[str]
    if isinstance(images, list):
        images_b64 = [str(x) for x in images if isinstance(x, str) and x]
    else:
        extract_res = _tool_pdf_image_extract(state, call)
        if not extract_res.ok:
            return extract_res
        images_b64 = list((extract_res.data or {}).get("images_b64", []))

    if not images_b64:
        return _tool_result(False, error="No images available for vision analysis.")

    llm = ChatOllama(
        model=os.getenv("VISION_MODEL", "qwen3-vl:4b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        format="json",
        temperature=0.0,
        num_predict=_ollama_num_predict(),
        client_kwargs={"timeout": _ollama_timeout_s()},
    )
    content: List[Dict[str, Any]] = []
    for b64 in images_b64[:3]:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )
    content.append(
        {
            "type": "text",
            "text": (
                'Reply JSON only: {"parallel_detectives": bool, "fan_in_present": bool, '
                '"parallel_judges": bool, "diagram_type": str, "description": str}'
            ),
        }
    )
    try:
        raw = llm.invoke([HumanMessage(content=content)]).content
    except Exception as e:
        return _tool_result(False, error=f"Vision analysis failed: {e!r}")

    return _tool_result(True, data={"analysis": raw, "image_count_used": min(3, len(images_b64))})


def _tool_to_evidence(call: ToolCall, result: ToolResult) -> Evidence:
    payload = result.data if result.ok else {"error": result.error}
    if call.tool_name == "pdf_query" and result.ok:
        found = int((result.data or {}).get("match_count", 0)) > 0
    elif call.tool_name == "git_log" and result.ok:
        found = int((result.data or {}).get("count", 0)) > 0
    elif call.tool_name == "pdf_image_extract" and result.ok:
        found = int((result.data or {}).get("image_count", 0)) > 0
    else:
        found = bool(result.ok)

    location = str(
        call.args.get("path")
        or call.args.get("pdf_path")
        or call.args.get("repo_path")
        or call.args.get("repo_url")
        or ""
    ).strip()
    if location == "$state.pdf_path":
        location = ""
    if not location:
        location = str((result.data or {}).get("pdf_path", "")).strip() or str(stateful_location_hint(call))

    rationale = call.why
    if not result.ok and result.error:
        rationale = f"{call.why} Tool error: {result.error}"

    return _evidence(
        dimension_id=call.dimension_id,
        goal=f"Executor dispatched `{call.tool_name}` for `{call.dimension_id}`",
        found=found,
        location=location or "n/a",
        rationale=rationale,
        content=(json.dumps(payload, default=str)[:1200] if payload is not None else None),
        confidence=0.9 if found else 0.35,
    )


def stateful_location_hint(call: ToolCall) -> str:
    if call.tool_name in {"git_log", "clone", "ast_scan"}:
        return "repo"
    if call.tool_name in {"pdf_ingest", "pdf_query", "pdf_image_extract", "vision_analyze"}:
        return "pdf"
    return "unknown"


def executor_node(state: AgentState) -> Dict[str, object]:
    calls_raw = state.get("planned_tool_calls", []) or []
    calls: List[ToolCall] = []
    for item in calls_raw:
        call = _as_tool_call(item)
        if call is not None:
            calls.append(call)

    calls = calls[:3]
    if not calls:
        return {"evidences": {"executor": []}, "tool_runs": []}

    active_pdf_index = state.get("pdf_index") if isinstance(state.get("pdf_index"), dict) else None
    evidences: List[Evidence] = []
    runs: List[ToolRunMetadata] = []
    fatal_error_type = ""
    fatal_error_message = ""

    tool_map = {
        "clone": lambda c: _tool_clone(state, c),
        "git_log": lambda c: _tool_git_log(state, c),
        "ast_scan": lambda c: _tool_ast_scan(state, c),
        "pdf_ingest": lambda c: _tool_pdf_ingest(state, c),
        "pdf_query": lambda c: _tool_pdf_query(state, c, active_pdf_index),
        "pdf_image_extract": lambda c: _tool_pdf_image_extract(state, c),
        "vision_analyze": lambda c: _tool_vision_analyze(state, c),
        # Backward-compatible aliases
        "ast_parse": lambda c: _tool_ast_scan(state, c),
        "query_pdf_chunks": lambda c: _tool_pdf_query(state, c, active_pdf_index),
        "extract_images_from_pdf": lambda c: _tool_pdf_image_extract(state, c),
    }

    for call in calls:
        start = time.perf_counter()
        handler = tool_map.get(call.tool_name)
        if handler is None:
            result = _tool_result(False, error=f"Tool not allowed: {call.tool_name}")
        else:
            try:
                result = handler(call)
                if call.tool_name == "pdf_ingest" and result.ok:
                    active_pdf_index = result.data or active_pdf_index
            except Exception as e:
                result = _tool_result(False, error=f"Unhandled executor exception: {e!r}")
                if not fatal_error_type:
                    fatal_error_type = "executor_exception"
                    fatal_error_message = str(result.error or "")

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        output_payload = result.data if result.ok else {"error": result.error}
        runs.append(
            ToolRunMetadata(
                dimension_id=call.dimension_id,
                tool_name=call.tool_name,
                elapsed_ms=elapsed_ms,
                output_size=_serialize_size(output_payload),
            )
        )
        evidences.append(_tool_to_evidence(call, result))
        if (not result.ok) and (not fatal_error_type):
            err = str(result.error or "")
            if ("git clone" in err.lower()) or ("clone" in call.tool_name.lower()):
                fatal_error_type = "clone_failure"
                fatal_error_message = err

    out: Dict[str, object] = {
        "evidences": {"executor": evidences},
        "tool_runs": runs,
        "error_type": fatal_error_type,
        "error_message": fatal_error_message,
        "failed_node": "executor" if fatal_error_type else "",
    }
    if isinstance(active_pdf_index, dict):
        out["pdf_index"] = active_pdf_index
    return out
