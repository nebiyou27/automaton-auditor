# Automaton Auditor

Autonomous repository governance using a LangGraph swarm and a digital courtroom workflow.

## Refined Project Structure

```text
automaton-auditor/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── create_structure.py
├── main.py
├── src/
│   ├── __init__.py
│   ├── state.py
│   ├── graph.py
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── detectives.py
│   │   ├── judges.py
│   │   └── justice.py
│   └── tools/
│       ├── __init__.py
│       └── repo_tools.py
├── rubric/
│   └── week2_rubric.json
├── audit/
│   ├── report_onself_generated/
│   ├── report_onpeer_generated/
│   ├── report_bypeer_received/
│   └── langsmith_logs/
└── tests/
    └── .gitkeep
```

> The structure is generated programmatically to keep architecture reproducible and consistent.

## Quick Start

```bash
python create_structure.py
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Setup

Copy `.env.example` to `.env` and configure:

```env
GEMINI_API_KEY=your_key_here
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=automaton-auditor
```

## Architecture Overview

1. Detective layer (parallel evidence collection): `RepoInvestigator`, `DocAnalyst`, optional vision inspector.
2. Evidence aggregation (fan-in synchronization).
3. Judicial bench (parallel reasoning): Prosecutor, Defense, Tech Lead.
4. Chief Justice: deterministic conflict resolution + final markdown report.

## Running the Auditor

Set target inputs in `main.py`:

```python
repo_url = "https://github.com/target/repository"
pdf_path = "path_to_report.pdf"
```

Run:

```bash
python main.py
```

Generated reports are written to `audit/report_onself_generated/`.

## Observability

If `LANGCHAIN_TRACING_V2=true`, execution traces are available in LangSmith for full auditability.
