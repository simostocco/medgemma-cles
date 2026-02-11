# repair.py (corrected version of YOUR script, same architecture)
# Notes:
# - keeps your approach (baseline -> validate -> repair missing-citation bullets -> revalidate)
# - fixes missing imports, robustness, and a couple of edge cases
# - does NOT change your external dependencies: it still expects:
#     research_pipeline_orchestrator, validate_citations,
#     get_evidence_summary_bullets, validate_bullets_only,
#     extract_section, normalize_bullet_line,
#     tokenizer, model, torch
#
# If you put this in a module, make sure those symbols are importable in scope.

from __future__ import annotations

import re
from typing import Any, Dict, List


def agentic_research_pipeline(
    disease: str,
    drug: str,
    max_retries: int = 3,
    target_coverage: float = 90.0,
):
    """
    Baseline stays unchanged.
    Agentic triggers if OVERALL citation coverage < target_coverage OR has bad refs.
    Then it repairs ONLY bullets missing citations (any section), without adding new facts.
    """

    # 0) Baseline
    result = research_pipeline_orchestrator(disease, drug)
    snippets = result.get("snippets") or []
    if not snippets:
        result["agentic_used"] = False
        result["agentic_attempts"] = 0
        return result

    report = result.get("report", "") or ""

    # 1) Baseline overall validation
    v0_all = validate_citations(report, snippets=snippets)
    result["metrics_all"] = v0_all
    result["trust_score"] = float(v0_all.get("coverage_pct", 0.0) or 0.0)

    # Gate on OVERALL coverage + bad refs
    if (result["trust_score"] >= target_coverage) and not v0_all.get("bad_reference_nums"):
        # Keep prints optional; leave as-is for notebook demo
        print("✅ Baseline overall grounding already strong; agentic not needed.")
        result["agentic_used"] = False
        result["agentic_attempts"] = 0
        # still compute sec2 metrics for UI
        sec2_bullets_final = get_evidence_summary_bullets(report)
        result["metrics_sec2"] = validate_bullets_only(sec2_bullets_final, snippets=snippets)
        return result

    # Helper: find bullet lines and their line indices
    def find_bullets(report_text: str):
        lines = report_text.splitlines()
        bullet_positions: List[int] = []
        bullets: List[str] = []
        for i, line in enumerate(lines):
            if line.strip().startswith("- "):
                bullet_positions.append(i)
                bullets.append(line.strip())
        return lines, bullet_positions, bullets

    # Helper: does a bullet contain any [S#]?
    def has_cite(b: str) -> bool:
        return bool(re.search(r"\[S\d+\]", b))

    # Helper: repair a list of bullets via your existing repair prompt style
    def repair_bullets_anywhere(bullets_to_fix: List[str]) -> List[str]:
        # Defensive max sid
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
- Keep each bullet's meaning if it is supported by snippets, and add best citation(s) at the end.
- If not supported, replace the bullet with: "Insufficient evidence in provided snippets." + ONE citation.
- Output exactly {len(bullets_to_fix)} bullets, numbered 1)..{len(bullets_to_fix)}), one per line.
- Output ONLY the numbered bullets (no extra text).

EVIDENCE SNIPPETS:
{evidence}

BULLETS TO FIX:
{bullets_block}
""".strip()

        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=350,
                do_sample=False,
                repetition_penalty=1.05,
            )

        repaired = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

        repaired_lines: List[str] = []
        for line in repaired.splitlines():
            line = line.strip()
            m = re.match(r"^\s*\d+\)\s+(.*)$", line)
            if m:
                fixed = m.group(1).strip()
                if not fixed.startswith("- "):
                    fixed = "- " + fixed.lstrip("-").strip()
                repaired_lines.append(fixed)

        # Enforce exact length (pad with originals; normalize)
        if len(repaired_lines) < len(bullets_to_fix):
            for i in range(len(repaired_lines), len(bullets_to_fix)):
                repaired_lines.append(bullets_to_fix[i])
        repaired_lines = repaired_lines[: len(bullets_to_fix)]

        return [normalize_bullet_line(x) for x in repaired_lines]

    # 2) Agentic loop
    agentic_used = False
    attempts = 0

    for attempt in range(1, max_retries + 1):
        lines, bullet_positions, bullets = find_bullets(report)
        bad_idx = [i for i, b in enumerate(bullets) if not has_cite(b)]

        if not bad_idx:
            break

        bullets_to_fix = [bullets[i] for i in bad_idx]
        print(
            f"⚠️ Agentic attempt {attempt}: repairing {len(bullets_to_fix)} "
            f"missing-citation bullets (whole report)..."
        )

        repaired = repair_bullets_anywhere(bullets_to_fix)

        # Apply repairs back into report
        for j, b_i in enumerate(bad_idx):
            line_pos = bullet_positions[b_i]
            lines[line_pos] = repaired[j]

        report = "\n".join(lines)

        agentic_used = True
        attempts += 1

        v_all = validate_citations(report, snippets=snippets)
        result["metrics_all"] = v_all
        result["trust_score"] = float(v_all.get("coverage_pct", 0.0) or 0.0)

        if (result["trust_score"] >= target_coverage) and not v_all.get("bad_reference_nums"):
            print(f"✅ Agentic PASS on attempt {attempt}: overall_coverage={result['trust_score']}%")
            break

    # 3) Finalize
    result["report"] = report
    result["agentic_used"] = agentic_used
    result["agentic_attempts"] = attempts

    # Final metrics
    result["metrics_all"] = validate_citations(result["report"], snippets=snippets)
    result["trust_score"] = float(result["metrics_all"].get("coverage_pct", 0.0) or 0.0)

    sec2_bullets_final = get_evidence_summary_bullets(result["report"])
    result["metrics_sec2"] = validate_bullets_only(sec2_bullets_final, snippets=snippets)

    return result


def repair_evidence_bullets(
    disease: str,
    drug: str,
    snippets: List[Dict[str, Any]],
    bullets_to_fix: List[str],
    max_new_tokens: int = 350,
) -> List[str]:
    """
    Repairs ONLY the provided bullets.
    Returns EXACTLY the same number of bullets, in the same order.
    """
    max_sid = max(int(s["sid"].replace("S", "")) for s in snippets if s.get("sid"))
    evidence = "\n\n".join(s.get("text", "") for s in snippets)

    bullets_block = "\n".join(
        f"{i+1}) {(b[2:].strip() if b.strip().startswith('- ') else b.strip())}"
        for i, b in enumerate(bullets_to_fix)
    )

    prompt = f"""
You are fixing ONLY the Evidence Summary bullets for a grounded biomedical report.

CONSTRAINTS:
- Use ONLY citations [S1]..[S{max_sid}] taken from the provided snippets.
- Do NOT introduce new topics. Stay strictly about: Drug={drug} and Disease={disease}.
- You MUST output exactly {len(bullets_to_fix)} bullets, numbered 1)..{len(bullets_to_fix)}).
- For each bullet:
  - If supported by snippets, keep meaning and add citation(s) at the end like [S3] or [S2][S5].
  - If not supported, replace with: "Insufficient evidence in provided snippets." plus ONE citation to the closest snippet.
- Output ONLY the numbered bullets, one per line. No extra text.

EVIDENCE SNIPPETS:
{evidence}

BULLETS TO FIX:
{bullets_block}
""".strip()

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
        )

    repaired = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

    repaired_lines: List[str] = []
    for line in repaired.splitlines():
        line = line.strip()
        m = re.match(r"^\s*\d+\)\s+(.*)$", line)
        if m:
            repaired_lines.append(normalize_bullet_line(m.group(1)))

    # Enforce exact length: pad or truncate safely
    if len(repaired_lines) < len(bullets_to_fix):
        for i in range(len(repaired_lines), len(bullets_to_fix)):
            repaired_lines.append(normalize_bullet_line(bullets_to_fix[i]))
    elif len(repaired_lines) > len(bullets_to_fix):
        repaired_lines = repaired_lines[: len(bullets_to_fix)]

    return repaired_lines


def patch_report_evidence_summary(report_text: str, repaired_map: Dict[int, str]) -> str:
    """
    Replace Evidence Summary bullets by their index (0-based) using repaired_map.
    Only touches section 2.
    """
    section_text, start_idx, end_idx = extract_section(
        report_text,
        start_pat=r"(^|\n)\s*(?:#{1,6}\s*)?(?:\*\*)?\s*2\)\s*Evidence Summary.*?:?\s*",
        end_pat=r"(^|\n)\s*(?:#{1,6}\s*)?(?:\*\*)?\s*3\)\s*Biological Rationale",
    )

    if start_idx == -1:
        return report_text

    lines = section_text.splitlines()
    bullet_positions = [i for i, l in enumerate(lines) if l.strip().startswith("- ")]
    bullets = [lines[i].strip() for i in bullet_positions]

    for b_idx, new_bullet in repaired_map.items():
        if 0 <= b_idx < len(bullets):
            bullets[b_idx] = normalize_bullet_line(new_bullet)

    for j, pos in enumerate(bullet_positions):
        lines[pos] = bullets[j]

    new_section = "\n".join(lines)
    return report_text[:start_idx] + new_section + report_text[end_idx:]
