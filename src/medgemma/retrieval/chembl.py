from __future__ import annotations

import os
import re
import json
import time
from typing import Any, Dict, List, Optional

import requests
from rapidfuzz import fuzz


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
