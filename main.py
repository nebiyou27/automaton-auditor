from dotenv import load_dotenv
from src.graph import build_graph

load_dotenv()


def main():
    repo_url = "https://github.com/nebiyou27/automaton-auditor.git"  # Example target
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