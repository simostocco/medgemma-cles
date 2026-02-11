from __future__ import annotations

import os
import re
import json
import time
from typing import Any, Dict, List, Optional

import requests
from rapidfuzz import fuzz
from collections import defaultdict


# ==============================
# Configuration
# ==============================

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

# Portable cache dir (no Kaggle paths)
CACHE_DIR = os.getenv("MEDGEMMA_CACHE_DIR", "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

HEADERS_JSON = {"Accept": "application/json"}

# Words that often indicate "a form" rather than the parent/base compound
FORM_HINTS = [
    "hydrochloride", "hcl", "sodium", "potassium", "calcium",
    "monohydrate", "hydrate", "tartrate", "phosphate", "sulfate",
    "mesylate", "maleate", "acetate", "bromide", "chloride"
]


# ==============================
# Cache Helpers
# ==============================

def _cache_path(key: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", key.strip().lower())
    return os.path.join(CACHE_DIR, safe + ".json")


def _load_cache(key: str) -> Optional[Dict[str, Any]]:
    path = _cache_path(key)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_cache(key: str, obj: Dict[str, Any]) -> None:
    path = _cache_path(key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ==============================
# ChEMBL Search
# ==============================

def chembl_molecule_search(query: str, max_results: int = 25, use_cache: bool = True) -> Dict[str, Any]:
    cache_key = f"chembl_molecule_search__{query}__{max_results}"

    if use_cache:
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached

    url = f"{CHEMBL_BASE}/molecule/search"
    params = {"q": query, "limit": max_results}

    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=HEADERS_JSON, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if use_cache:
                    _save_cache(cache_key, data)
                return data

            if r.status_code >= 500:
                time.sleep(2)

        except Exception:
            time.sleep(2)

    return {"molecules": []}


# ==============================
# Scoring Logic
# ==============================

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _looks_like_form(name: str) -> bool:
    n = _norm(name)
    return any(h in n for h in FORM_HINTS)


def _score_candidate(query: str, preferred_name: str) -> float:
    q = _norm(query)
    pn = _norm(preferred_name)

    score = fuzz.token_set_ratio(q, pn)

    if _looks_like_form(pn) and not _looks_like_form(q):
        score -= 12

    if not pn:
        score -= 30

    if q and q in pn:
        score += 4

    return score


# ==============================
# Drug Resolution
# ==============================

def resolve_drug_to_chembl(drug_name: str, max_results: int = 25) -> Dict[str, Any]:
    raw = chembl_molecule_search(drug_name, max_results=max_results)

    candidates = raw.get("molecules") or raw.get("molecule") or []
    if not isinstance(candidates, list):
        candidates = []

    parsed: List[Dict[str, Any]] = []

    for c in candidates:
        chembl_id = c.get("molecule_chembl_id") or c.get("chembl_id")
        pref_name = c.get("pref_name") or c.get("preferred_name") or ""

        if not chembl_id:
            continue

        score = _score_candidate(drug_name, pref_name)

        parsed.append({
            "chembl_id": chembl_id,
            "preferred_name": pref_name,
            "score": score,
        })

    if not parsed:
        return {
            "query": drug_name,
            "best_chembl_id": None,
            "preferred_name": None,
            "match_reason": "no_results",
            "alternatives": [],
        }

    parsed.sort(key=lambda x: x["score"], reverse=True)
    best = parsed[0]

    q = _norm(drug_name)
    pn = _norm(best["preferred_name"])

    if q == pn:
        reason = "exact"
    elif fuzz.token_set_ratio(q, pn) >= 90:
        reason = "high_confidence_fuzzy"
    else:
        reason = "fuzzy"

    return {
        "query": drug_name,
        "best_chembl_id": best["chembl_id"],
        "preferred_name": best["preferred_name"],
        "match_reason": reason,
        "alternatives": [p["chembl_id"] for p in parsed[1:6]],
        "top_matches_debug": parsed[:8],
    }


# ==============================
# Generic ChEMBL API Helpers
# ==============================

def chembl_get(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    cache_key: Optional[str] = None,
    use_cache: bool = True,
    retries: int = 3,
) -> Dict[str, Any]:

    params = params or {}

    if cache_key is None:
        key = endpoint + "__" + "__".join([f"{k}={v}" for k, v in sorted(params.items())])
        cache_key = "chembl_get__" + re.sub(r"[^a-zA-Z0-9._-]+", "_", key)

    if use_cache:
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached

    url = CHEMBL_BASE + endpoint

    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS_JSON, timeout=60)

            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue

            r.raise_for_status()
            data = r.json()

            if use_cache:
                _save_cache(cache_key, data)

            time.sleep(0.2)  # polite rate limit
            return data

        except Exception:
            if attempt == retries - 1:
                return {}
            time.sleep(2 ** attempt)

    return {}


def chembl_paginated(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    limit_total: int = 200,
    page_size: int = 100,
    cache_prefix: str = "",
) -> List[Dict[str, Any]]:

    params = dict(params or {})
    params["limit"] = page_size
    params["offset"] = 0

    all_items: List[Dict[str, Any]] = []

    while True:
        cache_key = f"{cache_prefix}{endpoint}__{params['offset']}__{page_size}"
        data = chembl_get(endpoint, params=params, cache_key=cache_key)

        list_key = next((k for k, v in data.items() if isinstance(v, list)), None)
        if not list_key:
            break

        batch = data.get(list_key, [])
        all_items.extend(batch)

        if len(all_items) >= limit_total:
            break

        page_meta = data.get("page_meta", {})
        if not page_meta.get("next") or not batch:
            break

        params["offset"] += page_size

    return all_items


def build_molecule_evidence_pack(molecule_chembl_id: str, max_activities: int = 400) -> Dict[str, Any]:
    """
    Creates a compact summary of:
    - molecule details (name, SMILES if available)
    - top targets based on bioactivity records
    """

    # 1) Basic molecule info
    mol = chembl_get(f"/molecule/{molecule_chembl_id}") or {}
    preferred_name = mol.get("pref_name")
    smiles = (mol.get("molecule_structures") or {}).get("canonical_smiles")
    inchi = (mol.get("molecule_structures") or {}).get("standard_inchi")
    molecule_type = mol.get("molecule_type")

    # 2) Pull bioactivities
    activities = chembl_paginated(
        "/activity",
        params={"molecule_chembl_id": molecule_chembl_id},
        limit_total=max_activities,
        page_size=200,
        cache_prefix=f"act_{molecule_chembl_id}_",
    )

    # 3) Aggregate activities per target
    per_target = defaultdict(lambda: {
        "target_chembl_id": None,
        "target_pref_name": None,
        "n_records": 0,
        "types": defaultdict(int),
        "units": set(),
        "standard_relation": defaultdict(int),
        "standard_value_examples": [],
        "assay_chembl_ids": set(),
        "references": set(),  # document_chembl_id
    })

    def safe_float(x):
        try:
            return float(x)
        except Exception:
            return None

    for a in activities:
        tgt = a.get("target_chembl_id")
        if not tgt:
            continue

        std_type = a.get("standard_type") or ""
        std_value = safe_float(a.get("standard_value"))
        std_units = a.get("standard_units") or ""
        std_rel = a.get("standard_relation") or ""
        tgt_name = a.get("target_pref_name") or ""

        bucket = per_target[tgt]
        bucket["target_chembl_id"] = tgt
        if tgt_name:
            bucket["target_pref_name"] = tgt_name
        bucket["n_records"] += 1

        if std_type:
            bucket["types"][std_type] += 1
        if std_units:
            bucket["units"].add(std_units)
        if std_rel:
            bucket["standard_relation"][std_rel] += 1

        if std_value is not None and len(bucket["standard_value_examples"]) < 10:
            bucket["standard_value_examples"].append({
                "type": std_type,
                "value": std_value,
                "units": std_units,
                "relation": std_rel,
            })

        assay_id = a.get("assay_chembl_id")
        if assay_id:
            bucket["assay_chembl_ids"].add(assay_id)

        doc_id = a.get("document_chembl_id")
        if doc_id:
            bucket["references"].add(doc_id)

    # 4) Score/sort targets by number of records
    targets_summary = []
    for tgt, info in per_target.items():
        targets_summary.append({
            "target_chembl_id": info["target_chembl_id"],
            "target_pref_name": info["target_pref_name"],
            "n_records": info["n_records"],
            "activity_types": dict(sorted(info["types"].items(), key=lambda x: x[1], reverse=True)),
            "units": sorted(list(info["units"]))[:5],
            "relation_counts": dict(info["standard_relation"]),
            "value_examples": info["standard_value_examples"],
            "n_assays": len(info["assay_chembl_ids"]),
            "n_references": len(info["references"]),
        })

    targets_summary.sort(key=lambda x: x["n_records"], reverse=True)

    return {
        "molecule_chembl_id": molecule_chembl_id,
        "preferred_name": preferred_name,
        "molecule_type": molecule_type,
        "canonical_smiles": smiles,
        "standard_inchi": inchi,
        "n_activity_records_fetched": len(activities),
        "top_targets": targets_summary[:15],
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }