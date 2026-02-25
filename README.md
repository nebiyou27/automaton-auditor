# Automaton Auditor

Automaton Auditor is a LangGraph-based, multi-agent auditing pipeline that evaluates a GitHub repository (and optionally a PDF report) against a rubric.

It uses a "Digital Courtroom" pattern:
- Detectives collect factual evidence.
- Judges score rubric criteria from different personas.
- Chief Justice produces a deterministic final report.

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
|  |  \- repo_tools.py
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

## Architecture

Execution flow (from `src/graph.py`):
1. `repo_investigator` runs from `START`.
2. `doc_analyst` runs if `pdf_path` is provided; otherwise `skip_doc_analyst`.
3. Both paths fan-in at `evidence_aggregator`.
4. Three judge nodes run in parallel:
   - `prosecutor`
   - `defense`
   - `techlead`
5. Fan-in to `chief_justice` for deterministic final synthesis.

## Core Components

- `src/state.py`
  - Defines:
    - `Evidence` (facts only)
    - `JudicialOpinion` (scores + arguments)
    - `AgentState` (shared LangGraph state)
  - Uses reducers for parallel safety:
    - `operator.ior` for evidence bucket merge
    - `operator.add` for opinion list merge

- `src/nodes/detectives.py`
  - `repo_investigator`: clones repo, checks files, reads git history, performs AST checks.
  - `doc_analyst`: ingests/query PDF into chunked evidence.

- `src/nodes/judges.py`
  - Uses local Ollama model (default `deepseek-r1:8b`) via `langchain_ollama`.
  - Generates structured JSON opinions per rubric criterion.

- `src/nodes/justice.py`
  - Resolves disagreements deterministically.
  - Produces markdown final verdict and remediation checklist.

- `src/tools/repo_tools.py`
  - Sandboxed git clone/history tools.
  - AST-based code analysis utilities.
  - PDF chunking and lexical query helpers.

## Prerequisites

- Python 3.11+ (3.12 recommended)
- Git installed and available in PATH
- Ollama running locally for judge nodes
- `uv` installed (recommended for lockfile-based reproducibility)

### Ollama setup

Install and start Ollama, then pull the model used by default:

```powershell
ollama pull deepseek-r1:8b
```

If you use another model, update `.env` (`OLLAMA_MODEL`).

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies (recommended: `uv` lock-based sync).

```powershell
uv sync --frozen
```

If you prefer `pip`, install from the pinned lock:

```powershell
pip install -r requirements.txt
```

3. Create your env file.

```powershell
Copy-Item .env.example .env
```

4. Update `.env` if needed:
- `OLLAMA_BASE_URL` (default `http://localhost:11434`)
- `OLLAMA_MODEL` (default `deepseek-r1:8b`)
- `LANGCHAIN_API_KEY` (optional; if empty, tracing is disabled in `main.py`)

## Dependency Management

- `pyproject.toml` is the canonical project metadata and dependency declaration.
- `requirements.lock` contains fully pinned runtime transitive dependencies for deterministic pip installs.
- `requirements.txt` is a compatibility shim that points to `requirements.lock`.

Regenerate lock artifacts after dependency updates:

```powershell
uv lock
uv export --frozen --no-dev --format requirements-txt -o requirements.lock
```

## Running

Run:

```powershell
python main.py
```

By default, `main.py` uses:
- `repo_url = "https://github.com/nebiyou27/automaton-auditor.git"`
- `pdf_path = ""` (PDF disabled)
- `rubric_path = "rubric/week2_rubric.json"`

Edit these in `main.py` to audit another repo or include a local PDF path.

## Input/Output State

Initial graph input keys:
- `repo_url`
- `pdf_path`
- `rubric_path`
- `evidences`
- `opinions`
- `final_report`

Primary outputs printed by `main.py`:
- Final state keys
- Final markdown report (`final_report`)
- Evidence bucket counts

## Rubric

Rubric is loaded from:
- `rubric/week2_rubric.json`

Current dimensions include:
- typed state definitions
- forensic tool engineering
- detective node implementation
- partial graph orchestration
- project infrastructure
- judicial nuance & dialectics

## Notes

- `tests/` currently contains only a placeholder (`.gitkeep`).
- `audit/` contains output/log folders (also placeholder-tracked).
- If `pdf_path` is empty, the graph safely routes through `skip_doc_analyst` to preserve fan-in behavior.
- Commit lockfile updates together with `pyproject.toml` changes to keep environments reproducible.
