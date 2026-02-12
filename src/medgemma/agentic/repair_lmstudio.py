from __future__ import annotations

import re
from typing import Any, Dict, List, Callable

from medgemma.pipeline.orchestrator import run_pipeline
from medgemma.validation.citations import validate_citations, validate_bullets_only
from medgemma.generation.prompts import build_prompt  


def get_evidence_summary_bullets(report_text: str) -> List[str]:
    """
    Extract bullets under section 2) Evidence Summary if possible, else all '- ' bullets.
    """
    lines = report_text.splitlines()

    # naive section extraction: start at "2) Evidence Summary" until "3) Biological Rationale"
    start = None
    end = None
    for i, ln in enumerate(lines):
        if re.search(r"^\s*\*{0,2}2\)\s*Evidence Summary", ln, flags=re.I):
            start = i
        if start is not None and re.search(r"^\s*\*{0,2}3\)\s*Biological Rationale", ln, flags=re.I):
            end = i
            break

    scope = lines[start:end] if start is not None else lines
    bullets = [l.strip() for l in scope if l.strip().startswith("- ")]
    if bullets:
        return bullets

    return [l.strip() for l in lines if l.strip().startswith("- ")]


def agentic_research_pipeline_lmstudio(
    disease: str,
    drug: str,
    llm_generate: Callable[[str, int], str],
    *,
    max_retries: int = 3,
    target_coverage: float = 90.0,
) -> Dict[str, Any]:
    """
    LM Studio agentic loop:
    baseline run_pipeline -> validate -> repair only missing-citation bullets -> revalidate
    """

    result = run_pipeline(disease=disease, drug=drug)
    n_rewritten_to_insufficient = 0
    
    snippets = result.get("snippets") or []
    if not snippets:
        result["agentic_used"] = False
        result["agentic_attempts"] = 0
        return result

    report = result.get("report", "") or ""
    v0 = validate_citations(report, snippets=snippets)

    result["metrics_all"] = v0
    result["trust_score"] = float(v0.get("coverage_pct", 0.0) or 0.0)

    if (result["trust_score"] >= target_coverage) and not v0.get("bad_reference_nums"):
        result["agentic_used"] = False
        result["agentic_attempts"] = 0
        sec2 = get_evidence_summary_bullets(report)
        result["metrics_sec2"] = validate_bullets_only(sec2, snippets=snippets)
        return result

    def has_cite(line: str) -> bool:
        return bool(re.search(r"\[S\d+\]", line))

    def repair_bullets_anywhere(bullets_to_fix: List[str]) -> List[str]:
        max_sid = max(int(s["sid"].replace("S", "")) for s in snippets if s.get("sid"))
        evidence = "\n\n".join(s.get("text", "") for s in snippets)

        bullets_block = "\n".join(
            f"{i+1}) {(b[2:].strip() if b.strip().startswith('- ') else b.strip())}"
            for i, b in enumerate(bullets_to_fix)
        )

        prompt = f"""
You are repairing ONLY bullet points that are missing citations in a grounded biomedical report.

CONSTRAINTS:
- Use ONLY citations [S1]..[S{max_sid}] from the provided snippets.
- Do NOT introduce any new factual claims.
- If the bullet meaning is NOT clearly supported by snippets, you MUST replace it with:
  "Insufficient evidence in provided snippets." + ONE citation.
- Do NOT introduce new scientific claims.
- Do NOT summarize snippet content unless it directly supports the original bullet.- If not supported, replace the bullet with: "Insufficient evidence in provided snippets." + ONE citation.
- Output exactly {len(bullets_to_fix)} bullets, numbered 1)..{len(bullets_to_fix)}), one per line.
- Output ONLY the numbered bullets (no extra text).

EVIDENCE SNIPPETS:
{evidence}

BULLETS TO FIX:
{bullets_block}
""".strip()

        repaired_text = llm_generate(prompt, 350)

        repaired_lines: List[str] = []
        for line in repaired_text.splitlines():
            line = line.strip()
            m = re.match(r"^\s*\d+\)\s+(.*)$", line)
            if m:
                fixed = m.group(1).strip()
                if not fixed.startswith("- "):
                    fixed = "- " + fixed.lstrip("-").strip()
                repaired_lines.append(fixed)

        # enforce exact length
        if len(repaired_lines) < len(bullets_to_fix):
            repaired_lines.extend(bullets_to_fix[len(repaired_lines):])
        repaired_lines = repaired_lines[: len(bullets_to_fix)]

        insuff_count = 0

        for line in repaired_lines:
            if "Insufficient evidence" in line:
                insuff_count += 1

        return repaired_lines, insuff_count

    # agentic loop: repair missing-citation bullets across report
    attempts = 0
    agentic_used = False

    for attempt in range(1, max_retries + 1):
        lines = report.splitlines()

        bullet_positions = [i for i, l in enumerate(lines) if l.strip().startswith("- ")]
        bullets = [lines[i].strip() for i in bullet_positions]

        bad_idx = [i for i, b in enumerate(bullets) if not has_cite(b)]
        if not bad_idx:
            break

        bullets_to_fix = [bullets[i] for i in bad_idx]
        repaired, insuff_count = repair_bullets_anywhere(bullets_to_fix)
        n_rewritten_to_insufficient += insuff_count

        for j, b_i in enumerate(bad_idx):
            lines[bullet_positions[b_i]] = repaired[j]

        report = "\n".join(lines)
        agentic_used = True
        attempts += 1

        v = validate_citations(report, snippets=snippets)
        result["metrics_all"] = v
        result["trust_score"] = float(v.get("coverage_pct", 0.0) or 0.0)

        if (result["trust_score"] >= target_coverage) and not v.get("bad_reference_nums"):
            break

    result["report"] = report
    result["agentic_used"] = agentic_used
    result["agentic_attempts"] = attempts

    # section metrics
    sec2 = get_evidence_summary_bullets(report)
    result["metrics_sec2"] = validate_bullets_only(sec2, snippets=snippets)
    result["n_rewritten_to_insufficient"] = n_rewritten_to_insufficient
    
    return result