# Automaton Auditor

A LangGraph-based multi-agent auditing pipeline that evaluates a GitHub repository
and optional PDF report against a rubric using a Digital Courtroom pattern:

- **Detectives** collect factual evidence (code, docs, diagrams)
- **Judges** score each rubric criterion from distinct personas (Prosecutor, Defense, Tech Lead)
- **Chief Justice** applies deterministic rules to produce a final verdict

## Setup

1. Clone the repo and create a virtual environment
2. Install dependencies:
```powershell
   pip install -r requirements.txt
```
3. Copy `.env.example` to `.env` and fill in your values:
```powershell
   Copy-Item .env.example .env
```
4. Ensure Ollama is running and models are available:
```powershell
   ollama pull deepseek-r1:8b
   ollama pull qwen3-vl:4b
```

## Running
```powershell
python main.py <repo_url> <pdf_path> <rubric_path>
```

**Example — audit a peer's repo with their PDF:**
```powershell
python main.py https://github.com/peer/repo.git C:\docs\peer_report.pdf rubric/week2_rubric.json
```

**Example — audit without PDF (PDF analysis skipped):**
```powershell
python main.py https://github.com/nebiyou27/automaton-auditor.git
```

If arguments are omitted, defaults are used:
- `repo_url`: `https://github.com/nebiyou27/automaton-auditor.git`
- `pdf_path`: empty (routes through `skip_doc_analyst`)
- `rubric_path`: `rubric/week2_rubric.json`

Reports are written to:
audit/report_onpeer_generated/audit_report_YYYYMMDD_HHMM.md

## Project Structure
```text
automaton-auditor/
├── main.py
├── pyproject.toml
├── requirements.txt
├── requirements.lock
├── .env.example
├── rubric/
│   └── week2_rubric.json          # Machine-readable rubric (the "Constitution")
├── src/
│   ├── graph.py                   # LangGraph StateGraph definition
│   ├── state.py                   # AgentState, Evidence, JudicialOpinion models
│   ├── tools/
│   │   ├── repo_tools.py          # Git clone, AST parsing, PDF chunking tools
│   │   └── rubric_utils.py        # Rubric loader and artifact filter
│   └── nodes/
│       ├── detectives.py          # RepoInvestigator, DocAnalyst, VisionInspector
│       ├── aggregator.py          # Evidence fan-in synchronization
│       ├── judges.py              # Prosecutor, Defense, TechLead personas
│       ├── justice.py             # Chief Justice (deterministic synthesis)
│       └── skip.py               # No-op node for missing PDF routing
├── audit/
│   ├── langsmith_logs/            # LangSmith trace links
│   ├── report_bypeer_received/    # Report your peer generated about your work
│   ├── report_onpeer_generated/   # Report you generated about your peer
│   └── report_onself_generated/   # Report you generated about your own work
└── tests/
```

## Environment Variables

| Variable | Description |
|---|---|
| `OLLAMA_MODEL` | Text model for detectives and judges (default: `deepseek-r1:8b`) |
| `VISION_MODEL` | Vision model for diagram analysis (default: `qwen3-vl:4b`) |
| `OLLAMA_BASE_URL` | Ollama server URL (default: `http://localhost:11434`) |
| `LANGCHAIN_API_KEY` | LangSmith API key — leave empty to disable tracing |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing (`true`/`false`) |
| `LANGCHAIN_PROJECT` | LangSmith project name for trace grouping |

## Notes

- `requirements.txt` installs from `requirements.lock` for reproducible builds
- If `LANGCHAIN_API_KEY` is empty, tracing is automatically disabled in `main.py`
- VisionInspector requires `pymupdf` — included in `requirements.txt`
- `tests/` contains a placeholder pending test coverage