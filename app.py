from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from buscar_hibrido import buscar_similares_hibrido  # use híbrido como padrão

app = FastAPI(title="API Triagem (Top-K ranqueado: Local + Externa)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class ClaimsRequest(BaseModel):
    claims: str
    top_k: int = 10

@app.post("/buscar_similares")
def api_buscar_similares(req: ClaimsRequest):
    try:
        res = buscar_similares_hibrido(req.claims, top_k=int(req.top_k))
        # resposta já vem com: rank, score (0–1), similaridade_percentual (0–100)
        return {"resultados": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
