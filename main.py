# main.py — API FastAPI para triagem (Top-K ranqueado, híbrido com fallback)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json, os, numpy as np
from numpy.linalg import norm
from typing import List, Dict, Optional
from processamento_texto import gerar_embedding

# tenta importar a busca híbrida (Lens + local); se não existir, seguimos só com local
_HAS_HYBRID = False
try:
    from buscar_hibrido import buscar_similares_hibrido  # precisa do lens_api.py também
    _HAS_HYBRID = True
except Exception:
    buscar_similares_hibrido = None  # type: ignore

app = FastAPI(title="API de Triagem de Patentes (Top-K ranqueado — Local + Externa)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Base local
# -----------------------------------------------------------------------------
BASE_PATH = "base_patentes.json"
_EMB: Optional[np.ndarray] = None
_META: List[Dict] = []

def _load_base():
    """Carrega/normaliza a base local de embeddings."""
    global _EMB, _META
    if not os.path.exists(BASE_PATH):
        _EMB = np.empty((0, 768), dtype=np.float32)
        _META = []
        return

    with open(BASE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    embs, meta = [], []
    for p in data:
        v = p.get("embedding")
        if isinstance(v, list) and v:
            v = np.asarray(v, dtype=np.float32)
            n = float(norm(v))
            if n > 0:
                v = v / n
            embs.append(v)
            meta.append({
                "lens_id": p.get("lens_id", ""),
                "title": p.get("title", ""),
                "claims": p.get("claims", "")
            })

    _EMB = np.vstack(embs) if embs else np.empty((0, 768), dtype=np.float32)
    _META = meta

_load_base()

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class Patente(BaseModel):
    titulo: str = ""
    resumo: str = ""
    claims: str = ""
    descricao: str = ""
    top_k: int = 10

class ClaimsRequest(BaseModel):
    claims: str
    top_k: int = 10

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _lens_link(lens_id: str) -> str:
    return f"https://www.lens.org/lens/patent/{lens_id}" if lens_id else ""

def _topk_indices(sims: np.ndarray, k: int) -> np.ndarray:
    k = min(max(1, int(k)), sims.size)
    idx = np.argpartition(sims, -k)[-k:]
    return idx[np.argsort(sims[idx])[::-1]]

def _rank_resposta(indices: np.ndarray, sims: np.ndarray) -> List[Dict]:
    """Formata resposta Top-K para resultados **locais**."""
    out = []
    for i in indices:
        m = _META[i]
        score = float(sims[i])  # 0–1
        out.append({
            "lens_id": m["lens_id"],
            "titulo": m["title"],
            "title": m["title"],
            "claims": (m["claims"] or "")[:300],
            "link": _lens_link(m["lens_id"]),
            "score": round(score, 4),                           # 0–1 para logs/métricas
            "similaridade_percentual": round(score * 100.0, 2), # 0–100 para exibição
            "fonte": "local"
        })
    return out

def _consulta_para_texto(p: Patente) -> str:
    """Escolhe a melhor fonte de texto (prioriza claims)."""
    if isinstance(p.claims, str) and p.claims.strip():
        return p.claims
    partes = [p.titulo or "", p.resumo or "", p.descricao or ""]
    return " ".join([x for x in partes if x.strip()])

def _buscar_local(texto: str, top_k: int) -> List[Dict]:
    """Executa Top-K **somente na base local**."""
    if _EMB is None or _EMB.shape[0] == 0:
        return []
    q = gerar_embedding(texto_claims=texto, normalize=True).astype(np.float32)
    qn = float(norm(q))
    if qn > 0:
        q = q / qn
    sims = _EMB @ q  # produto interno == cos (pois normalizados)
    idx = _topk_indices(sims, top_k)
    return _rank_resposta(idx, sims)

# -----------------------------------------------------------------------------
# Rotas
# -----------------------------------------------------------------------------
@app.get("/")
def home():
    return {
        "status": "OK",
        "itens_na_base": len(_META),
        "modo_hibrido_disponivel": _HAS_HYBRID,
        "mensagem": "Use POST /verificar (payload completo) ou POST /buscar_similares (claims + top_k)."
    }

@app.post("/verificar")
def verificar(patente: Patente):
    """
    Recebe: titulo, resumo, claims, descricao, top_k.
    Tenta **híbrido** primeiro (se disponível) e faz fallback para **local**.
    """
    try:
        texto = _consulta_para_texto(patente).strip()
        if not texto:
            raise ValueError("Texto de entrada vazio (claims/título/resumo/descrição).")

        # 1) tentar híbrido
        if _HAS_HYBRID and callable(buscar_similares_hibrido):
            try:
                res = buscar_similares_hibrido(texto, top_k=int(patente.top_k))
                if res:  # já vem no formato certo; inclui 'fonte'
                    return {"resultados": res}
            except Exception:
                pass  # se Lens falhar, cai para local

        # 2) fallback local
        res_local = _buscar_local(texto, top_k=int(patente.top_k))
        if not res_local:
            # nenhuma base local e híbrido falhou
            raise RuntimeError("Sem resultados (base local vazia e busca externa indisponível).")
        return {"resultados": res_local}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/buscar_similares")
def buscar_similares(req: ClaimsRequest):
    """
    Recebe: claims, top_k.
    Tenta **híbrido** primeiro (se disponível) e faz fallback para **local**.
    """
    try:
        if not isinstance(req.claims, str) or not req.claims.strip():
            raise ValueError("Campo 'claims' vazio.")

        # 1) tentar híbrido
        if _HAS_HYBRID and callable(buscar_similares_hibrido):
            try:
                res = buscar_similares_hibrido(req.claims, top_k=int(req.top_k))
                if res:
                    return {"resultados": res}
            except Exception:
                pass  # se Lens falhar, cai para local

        # 2) fallback local
        res_local = _buscar_local(req.claims, top_k=int(req.top_k))
        if not res_local:
            raise RuntimeError("Sem resultados (base local vazia e busca externa indisponível).")
        return {"resultados": res_local}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- inicia o servidor ao rodar: python main.py ---
if __name__ == "__main__":
    import uvicorn
    print("🚀 Subindo API em http://127.0.0.1:8000 ...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
