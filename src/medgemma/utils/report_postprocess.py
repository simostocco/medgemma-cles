import re
from typing import Any, Dict, List

SID_RE = re.compile(r"\[S(\d+)\]")

def extract_used_sids(report: str) -> List[str]:
    nums = sorted({int(m.group(1)) for m in SID_RE.finditer(report)})
    return [f"S{n}" for n in nums]

def infer_evidence_strength(snippets: List[Dict[str, Any]]) -> str:
    text = " ".join((s.get("title","") + " " + s.get("text","")) for s in snippets).lower()

    # crude but effective signal for demos
    if any(k in text for k in ["randomized", "randomised", "double-blind", "placebo", "clinical trial"]):
        return "Human clinical signal present in retrieved snippets"
    if any(k in text for k in ["mouse", "mice", "rat", "rodent", "animal model", "preclinical"]):
        return "Preclinical / animal evidence dominates retrieved snippets"
    return "Mechanistic / indirect evidence in retrieved snippets"

def make_verdict(report: str, strength: str) -> str:
    insuff = report.lower().count("insufficient evidence")
    if insuff >= 3:
        return "Limited support in retrieved snippets; many claims are marked as insufficient."
    if "no direct evidence" in report.lower():
        return "No direct clinical evidence in retrieved snippets; conclusions are mainly mechanistic/preclinical."
    return f"Grounded summary from retrieved snippets. ({strength})"

def add_header_block(report: str, snippets: List[Dict[str, Any]]) -> str:
    used = extract_used_sids(report)
    strength = infer_evidence_strength(snippets)
    verdict = make_verdict(report, strength)

    header = (
        f"**Verdict:** {verdict}\n\n"
        f"**Evidence strength (from retrieved snippets):** {strength}\n\n"
        f"**Citations used:** {', '.join(used) if used else 'None'}\n\n"
        "---\n\n"
    )
    return header + report