import json
import numpy as np
from numpy.linalg import norm
from processamento_texto import gerar_embedding

CAMINHO_ARQUIVO = "base_patentes.json"

def _carregar_base():
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
            meta.append({
                "lens_id": p.get("lens_id", ""),
                "title": p.get("title", ""),
                "claims": p.get("claims", "")
            })
    E = np.vstack(embs) if embs else np.empty((0, 768), dtype=np.float32)
    return E, meta

def buscar_similares(texto_claims: str, top_k: int = 10, incluir_percentual: bool = True):
    """
    Retorna APENAS ranque (Top-K). Campos:
      - rank (1..K)
      - score           -> 0–1 (interno; se quiser mostrar, ok)
      - similaridade_percentual -> 0–100 (apenas apresentação)
    """
    E, meta = _carregar_base()
    if E.shape[0] == 0:
        return []

    q = gerar_embedding(texto_claims=texto_claims)  # normalizado
    sims = E @ q  # cos

    k = min(max(1, int(top_k)), sims.size)
    idx = np.argpartition(sims, -k)[-k:]
    idx = idx[np.argsort(sims[idx])[::-1]]

    resultados = []
    for pos, i in enumerate(idx, start=1):
        score = float(sims[i])             # 0–1 (sem classificação)
        res = {
            "rank": pos,
            "lens_id": meta[i]["lens_id"],
            "title": meta[i]["title"],
            "claims": meta[i]["claims"][:300],
            "score": round(score, 6)
        }
        if incluir_percentual:
            res["similaridade_percentual"] = round(score * 100.0, 2)
        resultados.append(res)
    return resultados

if __name__ == "__main__":
    print("Cole as claims:")
    claims = input("> ")
    for r in buscar_similares(claims, top_k=10):
        print(f"#{r['rank']:02d} {r['lens_id']} | {r['similaridade_percentual']:.2f}% | {r['title']}")
