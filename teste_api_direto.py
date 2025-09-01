import json
import numpy as np
from numpy.linalg import norm
from processamento_texto import gerar_embedding

# Carrega uma patente qualquer da base
with open("base_patentes.json", "r", encoding="utf-8") as f:
    base = json.load(f)

if not base:
    raise SystemExit("Base vazia.")

patente = base[0]
claims = patente.get("claims", "")
emb_salvo = np.array(patente.get("embedding", []), dtype=np.float32)

if emb_salvo.size == 0:
    raise SystemExit("Patente sem embedding salvo.")

# Regera embedding a partir das claims (normalizado)
emb_teste = gerar_embedding(texto_claims=claims, normalize=True).astype(np.float32)

# Garante normalização dos dois vetores
def _norm(v):
    n = float(norm(v))
    return v / n if n > 0 else v

emb_salvo = _norm(emb_salvo)
emb_teste = _norm(emb_teste)

cos = float(np.dot(emb_salvo, emb_teste))
print(f"🔬 Similaridade interna (cos): {cos:.4f}")
