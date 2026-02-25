import os

from dotenv import load_dotenv

# Load .env BEFORE importing anything that initializes LLMs at import-time
load_dotenv()

if not (os.getenv("LANGCHAIN_API_KEY") or "").strip():
    # Prevent LangSmith 401 retries when running locally without a key.
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"

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
    print("\n=== FINAL REPORT ===")
    print(result["final_report"])
    print("\n=== EVIDENCE BUCKETS ===")
    for k, v in result["evidences"].items():
        print(k, len(v))


if __name__ == "__main__":
    main()
