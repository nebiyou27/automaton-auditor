import os
import pathlib

from dotenv import load_dotenv

# Load .env BEFORE importing anything that initializes LLMs at import-time
load_dotenv()

if not (os.getenv("LANGCHAIN_API_KEY") or "").strip():
    print("⚠️  LANGCHAIN_API_KEY not set — LangSmith tracing will be skipped.")

from src.graph import build_graph  # noqa: E402


def main():
    repo_url = "https://github.com/nebiyou27/automaton-auditor.git"
    pdf_path = ""  # Optional

    graph = build_graph()
    result = graph.invoke(
        {
            "repo_url": repo_url,
            "pdf_path": pdf_path,
            "rubric_path": "rubric/week2_rubric.json",
            "evidences": {},
            "opinions": [],
            "final_report": "",
        }
    )

    print("\n=== FINAL STATE KEYS ===")
    print(result.keys())
    # FIXED: C11b
    out = pathlib.Path("audit/report_onself_generated/audit_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result["final_report"], encoding="utf-8")
    print(f"Report saved to {out}")
    print("\n=== EVIDENCE BUCKETS ===")
    for k, v in result["evidences"].items():
        print(k, len(v))


if __name__ == "__main__":
    main()
