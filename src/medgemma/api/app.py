from fastapi import FastAPI
from pydantic import BaseModel
from medgemma.pipeline.orchestrator import run_pipeline
from medgemma.agentic.repair_lmstudio import agentic_research_pipeline_lmstudio
from medgemma.generation.lmstudio_backend import generate_report_lmstudio

app = FastAPI(title="MedGemma Evidence Engine")


class Query(BaseModel):
    drug: str
    disease: str
    agentic: bool = False


@app.post("/evidence_synthesis")
def evidence_synthesis(q: Query):

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

    return result