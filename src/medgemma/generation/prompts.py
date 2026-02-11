# prompts.py (corrected + repo-ready)
# Fixes:
# - adds typing + safer access to snippet/molecule fields
# - avoids KeyError when some fields are missing
# - normalizes/limits molecular targets (can be empty)
# - makes it explicit that ONLY [S#] citations are allowed
# - keeps your original content/structure

from __future__ import annotations

from typing import Any, Dict, List, Optional


CITATION_RULES = """
You are a neuroscience research evidence assistant.

You MUST follow these rules:
1) Use ONLY the provided evidence snippets [S1], [S2], ... as sources.
2) Every factual claim MUST include at least one citation like [S3].
3) If evidence is missing or weak, explicitly say "Insufficient evidence in the provided snippets" and do NOT guess.
4) Do NOT provide medical advice. Do NOT claim a treatment works. This is research support only.
5) Distinguish clearly between:
   - what is directly supported by snippets
   - what is a hypothesis (label as "Hypothesis")
6) Include an "Uncertainty & Limitations" section that mentions:
   - evidence quality may vary (reviews vs experiments)
   - abstracts are incomplete summaries
7) Use ONLY bracket citations in the form [S#]. Do NOT cite anything else.
""".strip()


REPORT_TEMPLATE = """
Return a structured report with these exact sections:

1) Question
- Restate the user's disease+molecule query in 1 sentence.

2) Evidence Summary (with citations)
- 4–8 bullet points summarizing what the snippets say about the molecule and disease.
- Each bullet MUST end with citations like [S2][S5].

3) Biological Rationale (with citations)
- Explain plausible biological mechanisms mentioned in the snippets.
- If you infer beyond the text, label it as Hypothesis and still cite supporting snippets.

4) Contradictions / Gaps (with citations if applicable)
- Note disagreements, missing info, or why evidence may not be strong.

5) Uncertainty & Limitations
- Include the required limitations.

6) Safety Note
- One short paragraph: not medical advice, not a validated therapeutic recommendation.
""".strip()


def build_prompt(
    disease: str,
    drug: str,
    snippets: List[Dict[str, Any]],
    mol_pack: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Builds a single prompt string that includes:
    - citation rules (anti-hallucination constraints)
    - report structure template
    - evidence snippets (PubMed abstracts)
    - optional molecular profile (ChEMBL context)
    """

    # Evidence snippets block (defensive: skip missing text)
    snippet_texts = [s.get("text", "").strip() for s in (snippets or [])]
    snippet_texts = [t for t in snippet_texts if t]
    evidence_block = "\n\n".join(snippet_texts)

    # Optional: add ChEMBL-based context (helps biological rationale grounding)
    mol_profile = ""
    if mol_pack:
        chembl_id = mol_pack.get("molecule_chembl_id", "")
        top_targets = mol_pack.get("top_targets") or []
        top_target_names = []
        for t in top_targets[:3]:
            name = (t.get("target_pref_name") or "").strip()
            if name:
                top_target_names.append(name)

        top_targets_str = ", ".join(top_target_names) if top_target_names else "N/A"

        mol_profile = (
            "MOLECULAR PROFILE:\n"
            f"- ChEMBL ID: {chembl_id or 'N/A'}\n"
            f"- Top Targets: {top_targets_str}\n\n"
        )

    prompt = (
        CITATION_RULES
        + "\n\n"
        + REPORT_TEMPLATE
        + "\n\n"
        + mol_profile
        + "EVIDENCE SNIPPETS:\n"
        + (evidence_block if evidence_block else "[No snippets provided]")
        + "\n\n"
        + f"USER QUESTION: Write the research report for {drug} in {disease}.\n"
        + "IMPORTANT FORMAT: In Section 2, write bullet points starting with '-' and END each bullet with citations like [S1][S2]."
    )

    return prompt
