# src/medgemma/retrieval/pubmed.py
# Corrected + repo-ready version of your PubMed module.
# Fixes:
# - adds missing imports (requests, time, typing, os, json, re)
# - removes Kaggle-specific constants and makes TOOL/EMAIL configurable via env vars
# - adds a portable cache (self-contained) so this file doesn't depend on chembl.py cache functions
#   (if you prefer shared cache utils, tell me and I’ll refactor)
# - improves retry/backoff and keeps NCBI polite rate limiting
# - adds safer XML parsing (guards)
# - keeps your return schema compatible

from __future__ import annotations

import os
import re
import json
import time
import hashlib
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import requests


# =========================
# NCBI / PubMed config
# =========================
NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Recommended by NCBI: identify your tool + email
NCBI_TOOL = os.getenv("NCBI_TOOL", "medgemma-cles")
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "") or None

# Polite: ~3 requests/sec baseline (NCBI varies; consider API key later)
NCBI_SLEEP_SECONDS = float(os.getenv("NCBI_SLEEP_SECONDS", "0.34"))


# =========================
# Portable cache (self-contained)
# =========================
CACHE_DIR = os.getenv("MEDGEMMA_CACHE_DIR", "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_key_to_path(key: str) -> str:
    # Hash avoids filename length issues + collisions
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"pubmed_{h}.json")

def _load_cache(key: str) -> Optional[Dict[str, Any]]:
    path = _cache_key_to_path(key)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def _save_cache(key: str, obj: Dict[str, Any]) -> None:
    path = _cache_key_to_path(key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================
# NCBI HTTP wrapper
# =========================
def _ncbi_get(endpoint: str, params: Dict[str, Any], timeout: int = 60, retries: int = 4) -> requests.Response:
    """
    Wrapper for NCBI E-utilities with polite rate limiting + retries.
    Retries on transient failures / throttling.
    """
    params = dict(params)
    params["tool"] = NCBI_TOOL
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL

    url = f"{NCBI_EUTILS}/{endpoint}"

    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)

            # Throttling / transient errors
            if r.status_code in (429, 500, 502, 503):
                wait = 2 ** attempt
                time.sleep(wait)
                continue

            r.raise_for_status()

            # Polite rate limit
            time.sleep(NCBI_SLEEP_SECONDS)
            return r

        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError(f"NCBI request failed after {retries} retries: {endpoint}")


# =========================
# PubMed API functions
# =========================
def pubmed_esearch(term: str, retmax: int = 20, sort: str = "relevance") -> List[str]:
    """
    Returns a list of PubMed IDs (PMIDs).
    sort can be 'relevance' or 'date'.
    """
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": retmax,
        "retmode": "json",
        "sort": sort,
    }
    r = _ncbi_get("esearch.fcgi", params=params)
    return r.json().get("esearchresult", {}).get("idlist", []) or []


def pubmed_efetch(pmids: List[str], rettype: str = "abstract") -> str:
    """
    Fetch details for a list of PMIDs.
    Returns raw XML text.
    """
    if not pmids:
        return ""

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": rettype,
    }
    r = _ncbi_get("efetch.fcgi", params=params)
    return r.text or ""


def _findtext(el: Optional[ET.Element], path: str, default: str = "") -> str:
    if el is None:
        return default
    v = el.findtext(path)
    return v.strip() if isinstance(v, str) else default


def parse_pubmed_xml(xml_text: str) -> List[Dict[str, Any]]:
    """
    Parses PubMed XML into a list of dicts:
    pmid, title, abstract, journal, year, authors (+ optional extras)
    """
    if not (xml_text or "").strip():
        return []

    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []

    articles: List[Dict[str, Any]] = []

    for pubmed_article in root.findall(".//PubmedArticle"):
        medline = pubmed_article.find("MedlineCitation")
        if medline is None:
            continue
        article = medline.find("Article")
        if article is None:
            continue

        pmid = _findtext(medline, "PMID", default="")
        title = _findtext(article, "ArticleTitle", default="")

        # Abstract (can be multi-part)
        abstract_texts: List[str] = []
        abstract = article.find("Abstract")
        if abstract is not None:
            for at in abstract.findall("AbstractText"):
                label = at.attrib.get("Label")
                txt = "".join(at.itertext()).strip()
                if not txt:
                    continue
                abstract_texts.append(f"{label}: {txt}" if label else txt)

        abstract_joined = "\n".join(abstract_texts)

        journal = _findtext(article, "Journal/Title", default="")
        year = (
            _findtext(article, "Journal/JournalIssue/PubDate/Year", default="")
            or _findtext(article, "Journal/JournalIssue/PubDate/MedlineDate", default="")
        )

        # Authors
        authors: List[str] = []
        author_list = article.find("AuthorList")
        if author_list is not None:
            for a in author_list.findall("Author"):
                collective = _findtext(a, "CollectiveName", default="")
                if collective:
                    authors.append(collective)
                    continue
                fore = _findtext(a, "ForeName", default="")
                last = _findtext(a, "LastName", default="")
                full = (fore + " " + last).strip()
                if full:
                    authors.append(full)

        # DOI
        doi = ""
        for aid in article.findall(".//ArticleId"):
            if aid.attrib.get("IdType") == "doi":
                doi = (aid.text or "").strip()
                break

        # Publication types
        pub_types = [
            _findtext(pt, ".", default="")
            for pt in article.findall("PublicationTypeList/PublicationType")
        ]
        pub_types = [p for p in pub_types if p]

        articles.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract_joined,
            "journal": journal,
            "year": year,
            "authors": authors[:10],
            "doi": doi,
            "pub_types": pub_types[:8],
            "has_abstract": bool(abstract_joined.strip()),
        })

    return articles


def build_text_evidence_pack(
    disease: str,
    drug_name: str,
    chembl_id: Optional[str] = None,
    n_papers: int = 20,
    sort: str = "relevance",
) -> Dict[str, Any]:
    """
    Builds a PubMed evidence pack:
      - query: ("disease"[Title/Abstract]) AND ("drug"[Title/Abstract])
      - pmids
      - parsed papers (title/abstract/etc)
    """

    query = f'("{disease}"[Title/Abstract]) AND ("{drug_name}"[Title/Abstract])'

    CACHE_VERSION = "v4"
    cache_key = f"pubmed_pack__{CACHE_VERSION}__{disease}__{drug_name}__{n_papers}__{sort}"

    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    pmids = pubmed_esearch(query, retmax=n_papers, sort=sort)

    if not pmids:
        pack = {
            "disease": disease,
            "drug_name": drug_name,
            "chembl_id": chembl_id,
            "query": query,
            "sort": sort,
            "pmids": [],
            "papers": [],
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_cache(cache_key, pack)
        return pack

    xml_text = pubmed_efetch(pmids)
    papers = parse_pubmed_xml(xml_text)

    # keep only papers with real abstracts
    papers = [p for p in papers if (p.get("abstract") or "").strip()]

    pack = {
        "disease": disease,
        "drug_name": drug_name,
        "chembl_id": chembl_id,
        "query": query,
        "sort": sort,
        "pmids": pmids,
        "papers": papers,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    _save_cache(cache_key, pack)
    return pack


def make_snippets_from_text_pack(
    text_pack: Dict[str, Any],
    max_snippets: int = 12,
    abstract_char_limit: int = 900,
) -> List[Dict[str, Any]]:
    """
    Converts pack['papers'] to snippet dicts used by the prompt.
    """
    papers = (text_pack.get("papers") or [])[:max_snippets]
    snippets: List[Dict[str, Any]] = []

    for i, p in enumerate(papers, start=1):
        sid = f"S{i}"
        title = (p.get("title") or "").strip()
        year = (p.get("year") or "").strip()
        journal = (p.get("journal") or "").strip()
        pmid = (p.get("pmid") or "").strip()

        abstract = (p.get("abstract") or "").strip().replace("\n", " ")
        if len(abstract) > abstract_char_limit:
            abstract = abstract[:abstract_char_limit].rsplit(" ", 1)[0] + "..."

        snippet_text = (
            f"[{sid}] Title: {title}\n"
            f"Year: {year} | Journal: {journal} | PMID: {pmid}\n"
            f"Abstract: {abstract}"
        )

        snippets.append({
            "sid": sid,
            "pmid": pmid,
            "title": title,
            "year": year,
            "journal": journal,
            "text": snippet_text,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        })

    return snippets
