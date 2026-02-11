# src/medgemma/pipeline/orchestrator.py
# Corrected + repo-ready version of your orchestrator.
# Fixes:
# - your snippet had an incomplete `def run_pipeline...` and duplicated function names
# - makes imports consistent with the repo structure
# - injects tokenizer/model/build_prompt_fn into generate_report (matches the corrected model.py)
# - adds typing + safer error handling
# - keeps return schema compatible with your notebook (metadata/snippets/report/metrics)

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

from medgemma.retrieval.chembl import resolve_drug_to_chembl, build_molecule_evidence_pack
from medgemma.retrieval.pubmed import build_text_evidence_pack, make_snippets_from_text_pack
from medgemma.generation.model import generate_report
from medgemma.generation.prompts import build_prompt
from medgemma.validation.citations import validate_citations


def run_pipeline(
    disease: str,
    drug: str,
    *,
    tokenizer,
    model,
    n_papers: int = 25,
    max_snippets: int = 10,
    max_activities: int = 400,
    sort: str = "relevance",
) -> Dict[str, Any]:
    """
    End-to-end pipeline:
    - drug -> ChEMBL resolve
    - optional molecular pack
    - PubMed retrieval -> snippets
    - TxGemma report generation
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

    # 3) Generate report with TxGemma (pass mol_pack for extra grounding)
    report_text = generate_report(
        disease=disease,
        drug=drug,
        snippets=snippets,
        mol_pack=mol_pack,
        build_prompt_fn=build_prompt,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=384,     # tune if you want longer outputs
        max_length=2048,        # tune if you hit truncation/OOM
    )

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
            {
                "sid": s.get("sid"),
                "pmid": s.get("pmid"),
                "title": s.get("title", "No Title"),
            }
            for s in snippets
        ],
    }


# Backwards-compatible alias (optional)
def research_pipeline_orchestrator(disease_name: str, drug_name: str, *, tokenizer, model) -> Dict[str, Any]:
    return run_pipeline(disease=disease_name, drug=drug_name, tokenizer=tokenizer, model=model)
