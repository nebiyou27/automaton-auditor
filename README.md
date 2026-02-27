# Automaton Auditor

Automaton Auditor is a LangGraph-based multi-agent auditing pipeline that evaluates a GitHub repository (and optionally a PDF report) against a rubric.

It uses a Digital Courtroom pattern:
- Detectives collect factual evidence.
- Judges score rubric criteria from different personas.
- Chief Justice produces a deterministic final report.

## Setup
1. Clone the repo
2. Create `.env` from `.env.example` and fill in your keys
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python main.py <repo_url> <pdf_path> rubric/week2_rubric.json`

## Project Structure

```text
automaton-auditor/
|- main.py
|- pyproject.toml
|- requirements.txt
|- requirements.lock
|- .env.example
|- rubric/
|  \- week2_rubric.json
|- src/
|  |- graph.py
|  |- state.py
|  |- tools/
|  |  |- repo_tools.py
|  |  \- rubric_utils.py
|  \- nodes/
|     |- detectives.py
|     |- aggregator.py
|     |- judges.py
|     |- justice.py
|     \- skip.py
|- audit/
|  |- langsmith_logs/
|  |- report_bypeer_received/
|  |- report_onpeer_generated/
|  \- report_onself_generated/
\- tests/
```

## Running

`main.py` accepts positional arguments:

```powershell
python main.py <repo_url> <pdf_path> <rubric_path>
```

Example:

```powershell
python main.py https://github.com/example/project.git C:\docs\peer_report.pdf rubric/week2_rubric.json
```

If arguments are omitted, defaults are used:
- `repo_url`: `https://github.com/nebiyou27/automaton-auditor.git`
- `pdf_path`: empty string (PDF analysis skipped)
- `rubric_path`: `rubric/week2_rubric.json`

Generated reports are written to:
- `audit/report_onpeer_generated/audit_report_YYYYMMDD_HHMM.md`

## Environment Variables

From `.env.example`:
- `OLLAMA_MODEL`
- `VISION_MODEL`
- `OLLAMA_BASE_URL`
- `LANGCHAIN_API_KEY`
- `LANGCHAIN_TRACING_V2`
- `LANGCHAIN_PROJECT`

Notes:
- If `LANGCHAIN_API_KEY` is empty, LangSmith tracing is skipped in `main.py`.
- Ensure Ollama is running locally and the configured model is available.

## Notes

- `requirements.txt` installs from `requirements.lock` for reproducibility.
- `tests/` currently contains a placeholder (`.gitkeep`).
- If `pdf_path` is empty, graph execution routes through `skip_doc_analyst`.
