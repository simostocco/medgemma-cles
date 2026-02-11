# src/medgemma/validation/citations.py
# Corrected + repo-ready.
# Fixes:
# - adds missing imports (re, typing)
# - makes snippet id extraction robust (sid like "S1" or int-ish)
# - fixes a subtle bug in validate_citations(): it was matching "S12" anywhere,
#   not specifically "[S12]" (could false-positive on titles/PMIDs/etc.)
# - keeps your return schema identical

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


_SID_BRACKET_RE = re.compile(r"\[S(\d+)\]")  # only count bracket citations [S#]
_BULLET_RE = re.compile(r"^(\*|-|\d+\))\s+")  # *, -, 1)


def _max_sid(snippets: Optional[List[Dict[str, Any]]]) -> int:
    if not snippets:
        return 0

    max_val = 0
    for s in snippets:
        sid = s.get("sid", "")
        if not sid:
            continue
        # sid can be "S1" (expected). Be defensive.
        m = re.match(r"^S(\d+)$", str(sid).strip(), flags=re.IGNORECASE)
        if m:
            max_val = max(max_val, int(m.group(1)))
    return max_val


def validate_citations(report_text: str, snippets: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Citation validator for generated reports.

    Goal: estimate grounding quality by checking whether bullet-like lines
    include snippet citations such as [S1], [S2], etc.
    """
    max_sid = _max_sid(snippets)

    lines = (report_text or "").splitlines()

    bullet_lines = [l.strip() for l in lines if _BULLET_RE.match(l.strip())]

    missing: List[str] = []
    cited_nums: List[int] = []

    for b in bullet_lines:
        matches = _SID_BRACKET_RE.findall(b)  # ✅ only [S#]
        if not matches:
            missing.append(b)
        else:
            cited_nums.extend(int(m) for m in matches)

    bad_refs = sorted({n for n in cited_nums if max_sid and (n < 1 or n > max_sid)})

    coverage_pct = round(((len(bullet_lines) - len(missing)) / len(bullet_lines)) * 100, 2) if bullet_lines else 0.0

    return {
        "n_bullets": len(bullet_lines),
        "n_missing_citations": len(missing),
        "coverage_pct": coverage_pct,
        "bad_reference_nums": bad_refs,
        "missing_examples": missing[:5],
    }


def validate_bullets_only(bullets: List[str], snippets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Citation coverage on a list of bullets only.
    """
    if not snippets:
        return {
            "n_bullets": len(bullets),
            "n_missing_citations": len(bullets),
            "coverage_pct": 0.0,
            "bad_reference_nums": [],
            "missing_examples": bullets[:5],
        }

    max_sid = _max_sid(snippets)

    bullets = [b.strip() for b in (bullets or []) if b and b.strip()]

    missing = [b for b in bullets if not _SID_BRACKET_RE.search(b)]
    cited_nums = [int(n) for n in _SID_BRACKET_RE.findall("\n".join(bullets))]
    bad_refs = sorted({n for n in cited_nums if n < 1 or n > max_sid})

    coverage_pct = round(((len(bullets) - len(missing)) / len(bullets)) * 100, 2) if bullets else 0.0

    return {
        "n_bullets": len(bullets),
        "n_missing_citations": len(missing),
        "coverage_pct": coverage_pct,
        "bad_reference_nums": bad_refs,
        "missing_examples": missing[:5],
    }
