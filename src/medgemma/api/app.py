from fastapi import FastAPI
from pydantic import BaseModel
from medgemma.pipeline.orchestrator import run_pipeline
from medgemma.agentic.repair_lmstudio import agentic_research_pipeline_lmstudio
from medgemma.generation.lmstudio_backend import generate_report_lmstudio
from medgemma.utils.reporting import save_markdown_report  
from medgemma.utils.report_postprocess import add_header_block

import os
from fastapi import Header, HTTPException, Depends

API_KEY = os.getenv("API_KEY")

def require_api_key(x_api_key: str = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
REPORTS_DIR = os.getenv("REPORTS_DIR", "/app/reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

app = FastAPI(title="MedGemma Evidence Engine")


class Query(BaseModel):
    drug: str
    disease: str
    agentic: bool = False

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/evidence_synthesis")
def evidence_synthesis(q: Query, _: None = Depends(require_api_key)):
    if q.agentic:
        def llm_generate(prompt: str, max_tokens: int):
            return generate_report_lmstudio(prompt, max_tokens=max_tokens)

        result = agentic_research_pipeline_lmstudio(
            disease=q.disease,
            drug=q.drug,
            llm_generate=llm_generate,
        )
    else:
        result = run_pipeline(disease=q.disease, drug=q.drug)

    # salva report
    try:
        report_path = save_markdown_report(result, REPORTS_DIR)
        result["report_path"] = report_path
    except Exception as e:
        result["report_save_error"] = str(e)
    snippets = result.get("snippets") or []
    if result.get("report"):
        result["report"] = add_header_block(result["report"], snippets)
        
    return result