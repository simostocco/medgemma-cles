# src/medgemma/pipeline/orchestrator.py

from __future__ import annotations

from typing import Any, Dict, Optional

from medgemma.retrieval.chembl import resolve_drug_to_chembl, build_molecule_evidence_pack
from medgemma.retrieval.pubmed import build_text_evidence_pack, make_snippets_from_text_pack
from medgemma.generation.lmstudio_backend import generate_report_lmstudio
from medgemma.generation.prompts import build_prompt
from medgemma.validation.citations import validate_citations


def run_pipeline(
    disease: str,
    drug: str,
    *,
    n_papers: int = 25,
    max_snippets: int = 10,
    max_activities: int = 400,
    sort: str = "relevance",
) -> Dict[str, Any]:
    """
    End-to-end pipeline (LM Studio backend):
    - drug -> ChEMBL resolve
    - optional molecular pack
    - PubMed retrieval -> snippets
    - build_prompt -> LM Studio generation
    - citation validation -> trust score
    """
    print(f"--- 🔎 Investigating {drug} for {disease} ---")

    # 1) Resolve drug -> ChEMBL
    drug_info = resolve_drug_to_chembl(drug)
    chembl_id = drug_info.get("best_chembl_id")

    # 1.5) Optional: build molecular evidence pack for mechanistic context
    mol_pack = None
    if chembl_id:
        try:
            mol_pack = build_molecule_evidence_pack(chembl_id, max_activities=max_activities)
        except Exception as e:
            print(f"⚠️ Could not build molecule evidence pack for {chembl_id}: {repr(e)}")
            mol_pack = None

    # 2) PubMed retrieval -> snippets
    text_pack = build_text_evidence_pack(
        disease=disease,
        drug_name=drug,
        chembl_id=chembl_id,
        n_papers=n_papers,
        sort=sort,
    )
    snippets = make_snippets_from_text_pack(text_pack, max_snippets=max_snippets)

    if not snippets:
        return {
            "error": "No snippets found",
            "metadata": {"drug": drug, "disease": disease, "chembl_id": chembl_id},
        }

    # 3) Generate report via LM Studio
    prompt = build_prompt(disease, drug, snippets, mol_pack=mol_pack)
    report_text = generate_report_lmstudio(prompt)

    # 4) Validate citations -> trust score
    v_results = validate_citations(report_text, snippets=snippets)
    total_b = int(v_results.get("n_bullets", 0) or 0)
    missing_b = int(v_results.get("n_missing_citations", 0) or 0)
    trust_score = round(((total_b - missing_b) / total_b) * 100, 2) if total_b > 0 else 0.0

    return {
        "metadata": {
            "disease": disease,
            "drug": drug,
            "chembl_id": chembl_id,
            "chembl_match_reason": drug_info.get("match_reason"),
            "chembl_preferred_name": drug_info.get("preferred_name"),
        },
        "molecule_pack": mol_pack,
        "snippets": snippets,
        "report": report_text,
        "trust_score": trust_score,
        "metrics": v_results,
        "sources": [
            {"sid": s.get("sid"), "pmid": s.get("pmid"), "title": s.get("title", "No Title")}
            for s in snippets
        ],
    }


# Backwards-compatible alias (optional)
def research_pipeline_orchestrator(disease_name: str, drug_name: str) -> Dict[str, Any]:
    return run_pipeline(disease=disease_name, drug=drug_name)
