# src/tools/repo_tools.py
# =======================
# Forensic tool functions used by detective nodes.
# These are pure tools (no LangGraph state). They must be safe, sandboxed,
# and return structured data to be wrapped into Evidence objects by detectives.

from __future__ import annotations

import ast
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from langchain_core.tools import tool
from PyPDF2 import PdfReader


@dataclass
class ToolResult:
    ok: bool
    data: Optional[dict] = None
    error: Optional[str] = None


# -------------------------------------------------------------------------------
# Git / Repo Tools (Sandboxed)
# -------------------------------------------------------------------------------

def clone_repo_sandboxed(
    repo_url: str,
    timeout_s: int = 90,
    analyzer: Optional[Callable[[str], ToolResult]] = None,
) -> ToolResult:
    """Clone a Git repo into a temporary sandbox and optionally run an analyzer callback. Input expects repo_url plus optional timeout_s and analyzer. Returns a ToolResult with clone metadata or an error."""
    if not repo_url or not isinstance(repo_url, str):
        return ToolResult(ok=False, error="Invalid repo_url (empty or not a string).")

    try:
        # FIXED: C6
        with tempfile.TemporaryDirectory(prefix="forensic_repo_clone_") as repo_path:
            # Use subprocess list args (no shell=True) to reduce injection risk.
            proc = subprocess.run(
                ["git", "clone", "--depth", "50", repo_url, repo_path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            if proc.returncode != 0:
                err = (proc.stderr or "").strip() or (proc.stdout or "").strip() or "Unknown git clone error."
                err_l = err.lower()
                if "authentication failed" in err_l or "could not read from remote repository" in err_l:
                    typed_err = f"git clone auth/permission error: {err}"
                elif "repository not found" in err_l:
                    typed_err = f"git clone repository not found: {err}"
                elif "could not resolve host" in err_l or "failed to connect" in err_l:
                    typed_err = f"git clone network error: {err}"
                else:
                    typed_err = f"git clone failed: {err}"
                return ToolResult(ok=False, error=typed_err)

            if analyzer is not None:
                return analyzer(repo_path)

            return ToolResult(
                ok=True,
                data={
                    "repo_path": repo_path,
                    "clone_depth": 50,
                    "timeout_s": timeout_s,
                },
            )
    except subprocess.TimeoutExpired:
        return ToolResult(ok=False, error=f"git clone timed out after {timeout_s}s.")
    except Exception as e:
        return ToolResult(ok=False, error=f"Unexpected error during clone: {e!r}")


def extract_git_history(repo_path: str, max_commits: int = 200) -> ToolResult:
    """Extract chronological commit history with hash, message, and timestamp from a local Git repo. Input expects repo_path and optional max_commits. Returns a ToolResult containing commit records or an error."""
    if not repo_path or not os.path.isdir(repo_path):
        return ToolResult(ok=False, error=f"Invalid repo_path: {repo_path}")

    if not os.path.isdir(os.path.join(repo_path, ".git")):
        return ToolResult(ok=False, error=f"Not a git repository: {repo_path}")

    try:
        proc = subprocess.run(
            ["git", "log", f"-n{max_commits}", "--pretty=format:%H|%s|%ai", "--reverse"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or "").strip() or (proc.stdout or "").strip()
            err_l = err.lower()
            if "does not have any commits yet" in err_l or "your current branch" in err_l:
                return ToolResult(ok=True, data={"commits": [], "count": 0, "empty_repo": True})
            return ToolResult(ok=False, error=err or "git log failed.")

        commits: List[Dict[str, str]] = []
        for line in proc.stdout.splitlines():
            if "|" not in line:
                continue
            parts = line.split("|", 2)
            if len(parts) != 3:
                continue
            commits.append({"hash": parts[0][:8], "message": parts[1], "date": parts[2]})

        return ToolResult(ok=True, data={"commits": commits, "count": len(commits), "empty_repo": len(commits) == 0})
    except subprocess.TimeoutExpired:
        return ToolResult(ok=False, error="git log timed out.")
    except Exception as e:
        return ToolResult(ok=False, error=f"Unexpected error reading git history: {e!r}")


# -------------------------------------------------------------------------------
# AST Tools (Deep structural analysis; not regex)
# -------------------------------------------------------------------------------

def _read_text_file(
    path: str,
    max_chars: int = 250_000,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> Tuple[bool, str]:
    if not os.path.exists(path):
        return False, f"File not found: {path}"
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            if start_line is not None or end_line is not None:
                lines = f.readlines()
                s = max(1, int(start_line) if start_line is not None else 1)
                e = int(end_line) if end_line is not None else len(lines)
                if e < s:
                    return True, ""
                return True, "".join(lines[s - 1 : e])[:max_chars]
            return True, f.read(max_chars)
    except Exception as e:
        return False, f"Failed to read file {path}: {e!r}"


def grep_search(repo_path: str, term: str) -> ToolResult:
    if not repo_path or not os.path.isdir(repo_path):
        return ToolResult(ok=False, error=f"Invalid repo_path: {repo_path}")
    if not isinstance(term, str) or not term:
        return ToolResult(ok=False, error="term must be a non-empty string.")

    matches: List[Dict[str, object]] = []
    for root, _, files in os.walk(repo_path):
        for filename in files:
            if not filename.endswith(".py"):
                continue
            file_path = os.path.join(root, filename)
            try:
                with open(file_path, "rb") as raw:
                    if b"\x00" in raw.read(4096):
                        continue
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_number, line in enumerate(f, start=1):
                        if term in line:
                            matches.append(
                                {
                                    "file": os.path.relpath(file_path, repo_path).replace("\\", "/"),
                                    "line_number": line_number,
                                    "line": line.rstrip("\r\n"),
                                }
                            )
            except (UnicodeDecodeError, OSError):
                continue

    return ToolResult(ok=True, data={"term": term, "count": len(matches), "matches": matches})


@tool
def analyze_langgraph_graph_py(graph_file_path: str) -> ToolResult:
    """Analyze a Python graph file via AST to detect StateGraph wiring and edge patterns. Input expects graph_file_path to a Python file. Returns a ToolResult with structural findings or a parse/read error."""
    ok, source_or_err = _read_text_file(graph_file_path)
    if not ok:
        return ToolResult(ok=False, error=source_or_err)

    try:
        tree = ast.parse(source_or_err)
    except SyntaxError as e:
        return ToolResult(ok=False, error=f"Syntax error parsing {graph_file_path}: {e}")

    findings = {
        "graph_file": graph_file_path,
        "has_stategraph_instantiation": False,
        "stategraph_calls": 0,
        "add_node_calls": 0,
        "add_edge_calls": 0,
        "edges": [],  # list of {"src": <str>, "dst": <str>}
        "fan_out_from_start": False,
        "fan_in_to_node": False,
        "possible_aggregator_nodes": [],
    }

    def _lit_str(n) -> Optional[str]:
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            return n.value
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "StateGraph":
                findings["has_stategraph_instantiation"] = True
                findings["stategraph_calls"] += 1

            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "add_node":
                    findings["add_node_calls"] += 1
                    if node.args:
                        name = _lit_str(node.args[0])
                        if name and "agg" in name.lower():
                            findings["possible_aggregator_nodes"].append(name)

                if node.func.attr == "add_edge":
                    findings["add_edge_calls"] += 1
                    if len(node.args) >= 2:
                        src = _lit_str(node.args[0])
                        dst = _lit_str(node.args[1])
                        if src and dst:
                            findings["edges"].append({"src": src, "dst": dst})

    start_out = [e for e in findings["edges"] if e["src"] in ("START", "__start__", "start")]
    if len(start_out) >= 2:
        findings["fan_out_from_start"] = True

    incoming_count: Dict[str, int] = {}
    for e in findings["edges"]:
        incoming_count[e["dst"]] = incoming_count.get(e["dst"], 0) + 1
    findings["fan_in_to_node"] = any(v >= 2 for v in incoming_count.values())

    return ToolResult(ok=True, data=findings)


@tool
def find_typed_state_definitions(state_file_path: str) -> ToolResult:
    """Inspect a state module with AST for class inheritance and reducer annotation usage. Input expects state_file_path to a Python source file. Returns a ToolResult with discovered structural facts or an error."""
    ok, source_or_err = _read_text_file(state_file_path)
    if not ok:
        return ToolResult(ok=False, error=source_or_err)

    try:
        tree = ast.parse(source_or_err)
    except SyntaxError as e:
        return ToolResult(ok=False, error=f"Syntax error parsing {state_file_path}: {e}")

    results = {
        "state_file": state_file_path,
        "classes": {},
        "has_operator_add": False,
        "has_operator_ior": False,
        "uses_annotated": False,
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "operator" and node.attr == "add":
                results["has_operator_add"] = True
            if node.value.id == "operator" and node.attr == "ior":
                results["has_operator_ior"] = True
        if isinstance(node, ast.Name) and node.id == "Annotated":
            results["uses_annotated"] = True

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append(b.id)
                elif isinstance(b, ast.Attribute):
                    bases.append(b.attr)
            results["classes"][node.name] = {"bases": bases}

    return ToolResult(ok=True, data=results)


# -------------------------------------------------------------------------------
# PDF Tools (RAG-lite chunk + query; no full-text dump)
# -------------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


@tool
def ingest_pdf_for_query(
    pdf_path: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    max_pages: int = 250,
) -> ToolResult:
    """Read a PDF, extract text, and split it into overlapping chunks for retrieval. Input expects pdf_path and optional chunk_size, chunk_overlap, and max_pages. Returns a ToolResult containing an index with chunk metadata or an error."""
    if not pdf_path or not isinstance(pdf_path, str):
        return ToolResult(ok=False, error="Invalid pdf_path.")
    if not os.path.isfile(pdf_path):
        return ToolResult(ok=False, error=f"PDF file not found: {pdf_path}")
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        return ToolResult(ok=False, error="Invalid chunk_size/chunk_overlap values.")

    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        return ToolResult(ok=False, error=f"Failed to open PDF: {e!r}")

    chunks: List[Dict[str, object]] = []
    chunk_idx = 0
    page_count = min(len(reader.pages), max_pages)

    for page_i in range(page_count):
        try:
            page_text = reader.pages[page_i].extract_text() or ""
        except Exception:
            page_text = ""
        page_text = page_text.strip()
        if not page_text:
            continue

        start = 0
        n = len(page_text)
        while start < n:
            end = min(n, start + chunk_size)
            text = page_text[start:end].strip()
            if text:
                chunks.append(
                    {
                        "chunk_id": f"p{page_i + 1}_c{chunk_idx}",
                        "page": page_i + 1,
                        "text": text,
                        "token_count": len(_tokenize(text)),
                    }
                )
                chunk_idx += 1
            if end >= n:
                break
            start = end - chunk_overlap

    return ToolResult(
        ok=True,
        data={
            "pdf_path": pdf_path,
            "page_count_indexed": page_count,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_count": len(chunks),
            "chunks": chunks,
        },
    )


@tool
def query_pdf_chunks(pdf_index: dict, query: str, top_k: int = 5) -> ToolResult:
    """Query an indexed PDF chunk structure using lexical overlap scoring. Input expects pdf_index, query text, and optional top_k. Returns a ToolResult with the best matching chunks or an error."""
    if not isinstance(pdf_index, dict):
        return ToolResult(ok=False, error="pdf_index must be a dict.")
    chunks = pdf_index.get("chunks")
    if not isinstance(chunks, list):
        return ToolResult(ok=False, error="pdf_index['chunks'] missing or invalid.")
    if not query or not isinstance(query, str):
        return ToolResult(ok=False, error="query must be a non-empty string.")
    if top_k <= 0:
        return ToolResult(ok=False, error="top_k must be > 0.")

    q_tokens = set(_tokenize(query))
    q_lower = query.lower()
    scored = []

    for ch in chunks:
        text = str(ch.get("text", ""))
        if not text:
            continue
        t_tokens = set(_tokenize(text))
        overlap = len(q_tokens.intersection(t_tokens))
        phrase_bonus = 1 if q_lower in text.lower() else 0
        score = overlap + phrase_bonus
        if score > 0:
            scored.append(
                {
                    "chunk_id": ch.get("chunk_id"),
                    "page": ch.get("page"),
                    "score": score,
                    "text_preview": text[:300],
                }
            )

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:top_k]

    return ToolResult(
        ok=True,
        data={
            "query": query,
            "top_k": top_k,
            "match_count": len(top),
            "matches": top,
        },
    )


@tool("grep_search")
def grep_search_tool(repo_path: str, term: str) -> ToolResult:
    """Search all Python files under a repo for a text term. Input expects repo_path and term. Returns ToolResult with file, line number, and line matches."""
    return grep_search(repo_path=repo_path, term=term)


@tool("read_file")
def read_file(path: str, max_chars: int = 250_000, start_line: Optional[int] = None, end_line: Optional[int] = None) -> Tuple[bool, str]:
    """Read text from a file, optionally by 1-indexed line range. Input expects path, max_chars, and optional start_line/end_line. Returns a success flag and file content or error string."""
    return _read_text_file(path=path, max_chars=max_chars, start_line=start_line, end_line=end_line)
