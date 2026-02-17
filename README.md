## Development History

The project evolved from an experimental research notebook
(notebooks/medgemma_original_research.ipynb)
into a production-ready FastAPI + Docker architecture.


# MedGemma Evidence Engine

MedGemma is an **evidence integrity layer** for biomedical questions.
It generates a grounded report with **snippet-backed citations** and returns a **trust score** (citation coverage). Optionally it can run an **agentic self-repair loop** to rewrite unsupported bullets into “Insufficient evidence…” with a citation.

## What you get

- FastAPI endpoint: `POST /evidence_synthesis`
- Response JSON includes:
  - `report` (markdown report)
  - `trust_score` (coverage %)
  - `snippets` (evidence snippets with SIDs)
  - metrics fields (`metrics_all`, `metrics_sec2`, etc.)
  - `report_path` saved on disk (Docker volume-friendly)

---

## Requirements

- Docker Desktop (Windows/macOS/Linux)
- LM Studio running locally with OpenAI-compatible server enabled
  - Default URL: `http://localhost:1234/v1/chat/completions`
  - In Docker we use: `http://host.docker.internal:1234/v1/chat/completions`

---

## Quickstart (Docker Compose) — recommended

### 1) Start LM Studio server
Enable the local server in LM Studio and confirm it’s running on port `1234`.

### 2) Create output folder
From repo root:

```powershell
mkdir reports


## Examples

Requests:
- `examples/requests/ibuprofen_als.json`
- `examples/requests/donepezil_alzheimer.json`
- `examples/requests/metformin_t2d.json`

Reports:
- `examples/reports/ibuprofen_als.md`
- `examples/reports/donepezil_alzheimer.md`
- `examples/reports/metformin_t2d.md`
