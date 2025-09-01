import json
import numpy as np
from numpy.linalg import norm
from typing import List, Dict
from processamento_texto import gerar_embedding
from lens_api import candidatos_por_claims

CAMINHO_ARQUIVO = "base_patentes.json"

def _carregar_local():
    try:
        with open(CAMINHO_ARQUIVO, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return np.empty((0, 768), dtype=np.float32), []
    embs, meta = [], []
    for p in data:
        v = p.get("embedding")
        if isinstance(v, list) and v:
            v = np.asarray(v, dtype=np.float32)
            n = float(norm(v))
            if n > 0: v = v / n
            embs.append(v)
            meta.append({"lens_id": p.get("lens_id",""), "title": p.get("title",""), "claims": p.get("claims",""), "fonte": "local"})
    E = np.vstack(embs) if embs else np.empty((0, 768), dtype=np.float32)
    return E, meta

def _dedup_keep_best(rows: List[Dict]) -> List[Dict]:
    best = {}
    for r in rows:
        lid = r["lens_id"]
        if lid not in best or r["score"] > best[lid]["score"]:
            best[lid] = r
    return list(best.values())

def buscar_similares_hibrido(texto_claims: str, top_k: int = 10, candidatos_externos: int = 25):
    q = gerar_embedding(texto_claims=texto_claims)  # normalizado

    # LOCAL
    E, meta_local = _carregar_local()
    res_local = []
    if E.shape[0] > 0:
        sims = E @ q
        k_loc = min(max(1, top_k * 3), sims.size)
        idx = np.argpartition(sims, -k_loc)[-k_loc:]
        idx = idx[np.argsort(sims[idx])[::-1]]
        for i in idx:
            res_local.append({
                "lens_id": meta_local[i]["lens_id"],
                "title": meta_local[i]["title"],
                "claims": meta_local[i]["claims"][:300],
                "score": float(sims[i]),          # 0–1
                "fonte": "local"
            })

    # EXTERNA (Lens) → embed → cos
    res_ext = []
    try:
        candidatos = candidatos_por_claims(texto_claims, max_docs=candidatos_externos)
        for c in candidatos:
            emb = gerar_embedding(texto_claims=c.get("claims",""))
            sim = float(np.dot(emb, q))
            res_ext.append({
                "lens_id": c["lens_id"],
                "title": c["title"],
                "claims": (c.get("claims") or "")[:300],
                "score": sim,                     # 0–1
                "fonte": "externa"
            })
    except Exception:
        pass

    # Combina, deduplica, ordena e mapeia para Top-K
    combinados = _dedup_keep_best(res_local + res_ext)
    combinados.sort(key=lambda r: r["score"], reverse=True)

    out = []
    for rank, r in enumerate(combinados[:top_k], start=1):
        out.append({
            "rank": rank,
            "lens_id": r["lens_id"],
            "title": r["title"],
            "claims": r["claims"],
            "score": round(r["score"], 6),                       # 0–1 p/ logs ou depuração
            "similaridade_percentual": round(r["score"]*100, 2), # só para exibir
            "fonte": r["fonte"]
        })
    return out
