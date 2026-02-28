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
        "--out",
        default="",
        help="Optional output report path. Defaults to timestamped audit path.",
    )
    parser.add_argument(
        "--enable-vision",
        action="store_true",
        help="Enable vision mode flag (parsed for compatibility).",
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

    return args


def main():
    args = parse_args()
    repo_url = args.repo
    pdf_path = args.pdf
    rubric_path = args.rubric

    print("Inputs Summary")
    print(f"- repo: {repo_url}")
    print(f"- pdf: {pdf_path or '(none)'}")
    print(f"- rubric: {rubric_path}")
    print(f"- out: {args.out or '(auto)'}")
    print(f"- enable_vision: {args.enable_vision}")

    graph = build_graph()
    result = graph.invoke(
        {
            "repo_url": repo_url,
            "pdf_path": pdf_path,
            "repo_path": "",
            "rubric_path": rubric_path,
            "evidences": {},
            "opinions": [],
            "tool_runs": [],
            "iteration": 0,
            "max_iters": 3,
            "tool_budget": 9,
            "final_report": "",
        }
    )

    print("\n=== FINAL STATE KEYS ===")
    print(result.keys())
    if args.out:
        out = pathlib.Path(args.out)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        is_self = "nebiyou27" in repo_url
        folder = "report_onself_generated" if is_self else "report_onpeer_generated"
        out = pathlib.Path(f"audit/{folder}/audit_report_{timestamp}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result["final_report"], encoding="utf-8")
    print(f"Report saved to {out}")
    print("\n=== EVIDENCE BUCKETS ===")
    for k, v in result["evidences"].items():
        print(k, len(v))


if __name__ == "__main__":
    main()
