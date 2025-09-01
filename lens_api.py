# lens_api.py — utilitários da Lens (busca externa)
import os, time, requests, re
from typing import List, Dict
from dotenv import load_dotenv

class LensDisabled(Exception):
    """A perna externa (Lens) está desativada por falta de API key."""

load_dotenv()

SEARCH_URL = "https://api.lens.org/patent/search"
DETAIL_URL = "https://api.lens.org/patent/{lens_id}"

def _get_api_key() -> str:
    key = os.getenv("LENS_API_KEY", "").strip()
    if not key:
        raise LensDisabled("LENS_API_KEY não definida (.env). Busca externa desativada.")
    return key

def _headers() -> dict:
    return {"Authorization": f"Bearer {_get_api_key()}", "Content-Type": "application/json"}

def _keywords(text: str, k: int = 8) -> str:
    tokens = re.findall(r"[a-zA-Z0-9\-_/]{3,}", (text or "").lower())
    out, seen = [], set()
    for t in tokens:
        if t not in seen:
            seen.add(t); out.append(t)
        if len(out) >= k: break
    return " ".join(out) if out else (text or "")[:64]

def _post(url: str, body: dict, retry: int = 3):
    for _ in range(retry):
        r = requests.post(url, headers=_headers(), json=body, timeout=60)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", "60"))); continue
        r.raise_for_status(); return r.json()
    raise RuntimeError("Falha na busca (rate limit/erro desconhecido).")

def _get(url: str, retry: int = 3):
    for _ in range(retry):
        r = requests.get(url, headers=_headers(), timeout=60)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", "60"))); continue
        r.raise_for_status(); return r.json()
    raise RuntimeError("Falha ao obter detalhes (rate limit/erro desconhecido).")

def buscar_ids_por_texto(claims_text: str, size: int = 30, offset: int = 0) -> List[str]:
    q = _keywords(claims_text, k=10)
    body = {
        "query": {"simple_query_string": {
            "query": q, "fields": ["claims^3", "description^2", "abstract"], "default_operator": "and"}},
        "size": size, "from": offset, "include": ["lens_id"]
    }
    data = _post(SEARCH_URL, body)
    return [d["lens_id"] for d in data.get("data", [])]

def detalhes_patente(lens_id: str) -> Dict:
    data = _get(DETAIL_URL.format(lens_id=lens_id))
    return {
        "lens_id": lens_id,
        "title": data.get("title", "") or "",
        "claims": data.get("claims", "") or "",
        "abstract": data.get("abstract", "") or "",
        "description": data.get("description", "") or "",
    }

def candidatos_por_claims(claims_text: str, max_docs: int = 25) -> List[Dict]:
    """Retorna até max_docs patentes (com texto) vindas da Lens para re-ranqueamento local."""
    ids = buscar_ids_por_texto(claims_text, size=max_docs)
    out = []
    for lid in ids:
        try:
            out.append(detalhes_patente(lid))
            time.sleep(0.3)  # cortesia de rate
        except Exception:
            continue
    return out
