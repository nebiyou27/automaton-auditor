from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stub_module(name: str, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod


def _noop_node(_state):
    return {}


def test_build_graph_compiles_with_stubbed_nodes():
    # Stub all imported node modules so this smoke test stays deterministic
    # and independent from optional external runtimes (e.g., local LLMs).
    _stub_module(
        "src.nodes.detectives",
        doc_analyst=_noop_node,
        repo_investigator=_noop_node,
        vision_inspector_node=_noop_node,
    )
    _stub_module("src.nodes.aggregator", evidence_aggregator=_noop_node)
    _stub_module("src.nodes.error_handler", error_handler_node=_noop_node)
    _stub_module("src.nodes.judge_repair", judge_repair_node=_noop_node)
    _stub_module("src.nodes.planner", planner_node=_noop_node)
    _stub_module("src.nodes.executor", executor_node=_noop_node)
    _stub_module(
        "src.nodes.reflector",
        judge_gate_node=_noop_node,
        reflector_node=_noop_node,
    )
    _stub_module(
        "src.nodes.skip",
        skip_doc_analyst=_noop_node,
        skip_vision_inspector=_noop_node,
    )
    _stub_module(
        "src.nodes.judges",
        defense_judge=_noop_node,
        judge_barrier_node=_noop_node,
        prosecutor_judge=_noop_node,
        techlead_judge=_noop_node,
    )
    _stub_module("src.nodes.justice", chief_justice=_noop_node)

    sys.modules.pop("src.graph", None)
    graph = importlib.import_module("src.graph")
    compiled = graph.build_graph()

    assert compiled is not None


def test_route_vision_inspector_skips_when_disabled_or_pdf_missing():
    import src.graph as graph

    assert graph._route_vision_inspector({"enable_vision": False, "pdf_path": "x.pdf"}) == "skip_vision_inspector"
    assert graph._route_vision_inspector({"enable_vision": True, "pdf_path": ""}) == "skip_vision_inspector"
