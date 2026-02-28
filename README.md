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
python main.py --repo <repo_url> --pdf <pdf_path> --rubric <rubric_path> --mode {self,peer,received}
```

**Example — audit a peer's repo with their PDF:**
```powershell
python main.py --repo https://github.com/peer/repo.git --pdf C:\docs\peer_report.pdf --rubric rubric/week2_rubric.json --mode peer
```

**Example — audit without PDF (PDF analysis skipped):**
```powershell
python main.py --repo https://github.com/nebiyou27/automaton-auditor.git --mode self
```

**Example - received report mode + extra output copy + vision flag:**
```powershell
python main.py --repo https://github.com/peer/repo.git --rubric rubric/week2_rubric.json --mode received --out audit/custom_report.md --enable-vision
```

If arguments are omitted, defaults are used:
- `--repo`: `https://github.com/nebiyou27/automaton-auditor.git`
- `--pdf`: empty (routes through `skip_doc_analyst`)
- `--rubric`: `rubric/week2_rubric.json`
- `--mode`: `self`
- `--out`: optional extra copy path (canonical mode path is always written)
- `--enable-vision`: parsed as a boolean flag

Reports are written to:
- `--mode self` -> `audit/report_onself_generated/audit_report_YYYYMMDD_HHMM.md`
- `--mode peer` -> `audit/report_onpeer_generated/audit_report_YYYYMMDD_HHMM.md`
- `--mode received` -> `audit/report_bypeer_received/audit_report_YYYYMMDD_HHMM.md`

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

