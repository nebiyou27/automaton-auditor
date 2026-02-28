import argparse
import os
import pathlib
from datetime import datetime

from dotenv import load_dotenv

# Load .env BEFORE importing anything that initializes LLMs at import-time
load_dotenv()

if not (os.getenv("LANGCHAIN_API_KEY") or "").strip():
    print("⚠️  LANGCHAIN_API_KEY not set — LangSmith tracing will be skipped.")

from src.graph import build_graph  # noqa: E402


MODE_TO_FOLDER = {
    "self": "report_onself_generated",
    "peer": "report_onpeer_generated",
    "received": "report_bypeer_received",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Automaton Auditor pipeline.")
    parser.add_argument(
        "--repo",
        default="https://github.com/nebiyou27/automaton-auditor.git",
        help="GitHub repository URL to audit.",
    )
    parser.add_argument(
        "--pdf",
        default="",
        help="Optional local PDF path to analyze.",
    )
    parser.add_argument(
        "--rubric",
        default="rubric/week2_rubric.json",
        help="Path to the rubric JSON file.",
    )
    parser.add_argument(
        "--mode",
        choices=["self", "peer", "received"],
        default="self",
        help="Audit/report mode for canonical output routing.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional extra output report path (canonical mode path is always written).",
    )
    parser.add_argument(
        "--enable-vision",
        action="store_true",
        help="Enable vision mode flag (parsed for compatibility).",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=6,
        help="Maximum planner/executor/reflector loop iterations.",
    )
    parser.add_argument(
        "--tool-budget",
        type=int,
        default=20,
        help="Maximum number of tool calls that may be executed.",
    )
    parser.add_argument(
        "--ollama-timeout-s",
        type=int,
        default=int((os.getenv("OLLAMA_TIMEOUT_S") or "45").strip() or "45"),
        help="Ollama request timeout in seconds (also exported as OLLAMA_TIMEOUT_S).",
    )
    parser.add_argument(
        "--ollama-num-predict",
        type=int,
        default=int((os.getenv("OLLAMA_NUM_PREDICT") or "256").strip() or "256"),
        help="Ollama max generated tokens per call (also exported as OLLAMA_NUM_PREDICT).",
    )
    args = parser.parse_args()

    if not (args.repo or "").strip():
        parser.error("--repo must be non-empty.")
    if not os.path.isfile(args.rubric):
        parser.error(f"--rubric file not found: {args.rubric}")
    if args.pdf:
        if not args.pdf.lower().endswith(".pdf"):
            parser.error("--pdf must end with .pdf.")
        if not os.path.isfile(args.pdf):
            parser.error(f"--pdf file not found: {args.pdf}")
    if args.max_iters < 0:
        parser.error("--max-iters must be >= 0.")
    if args.tool_budget < 0:
        parser.error("--tool-budget must be >= 0.")
    if args.ollama_timeout_s <= 0:
        parser.error("--ollama-timeout-s must be > 0.")
    if args.ollama_num_predict <= 0:
        parser.error("--ollama-num-predict must be > 0.")

    return args


def main():
    args = parse_args()
    os.environ["OLLAMA_TIMEOUT_S"] = str(args.ollama_timeout_s)
    os.environ["OLLAMA_NUM_PREDICT"] = str(args.ollama_num_predict)
    repo_url = args.repo
    pdf_path = args.pdf
    rubric_path = args.rubric

    print("Inputs Summary")
    print(f"- repo: {repo_url}")
    print(f"- pdf: {pdf_path or '(none)'}")
    print(f"- rubric: {rubric_path}")
    print(f"- mode: {args.mode}")
    print(f"- out: {args.out or '(auto)'}")
    print(f"- enable_vision: {args.enable_vision}")
    print(f"- max-iters: {args.max_iters}")
    print(f"- tool-budget: {args.tool_budget}")
    print("Ollama Runtime Config")
    print(f"- OLLAMA_TIMEOUT_S: {args.ollama_timeout_s}")
    print(f"- OLLAMA_NUM_PREDICT: {args.ollama_num_predict}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    canonical_folder = MODE_TO_FOLDER[args.mode]
    canonical_out = pathlib.Path(f"audit/{canonical_folder}/audit_report_{timestamp}.md")

    graph = build_graph()
    result = graph.invoke(
        {
            "repo_url": repo_url,
            "pdf_path": pdf_path,
            "repo_path": "",
            "enable_vision": args.enable_vision,
            "rubric_path": rubric_path,
            "evidences": {},
            "opinions": [],
            "tool_runs": [],
            "iteration": 0,
            "max_iters": args.max_iters,
            "tool_budget": args.tool_budget,
            "judge_schema_failures": [],
            "final_report": "",
            "audit_mode": args.mode,
            "report_path": str(canonical_out),
        }
    )

    print("\n=== FINAL STATE KEYS ===")
    print(result.keys())
    canonical_out.parent.mkdir(parents=True, exist_ok=True)
    canonical_out.write_text(result["final_report"], encoding="utf-8")
    print(f"Canonical report saved to {canonical_out}")

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result["final_report"], encoding="utf-8")
        print(f"Extra report copy saved to {out}")

    print("\n=== EVIDENCE BUCKETS ===")
    for k, v in result["evidences"].items():
        print(k, len(v))


if __name__ == "__main__":
    main()
