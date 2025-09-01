# processamento_texto.py
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
from typing import Optional

# Usamos a limpeza centralizada
try:
    # se você já tem limpar_texto.py no projeto
    from limpar_texto import limpar_texto
except Exception:
    # fallback mínimo, caso o módulo não exista por algum motivo
    import re
    _FALLBACK_PT = {
        "de","da","do","das","dos","a","o","as","os","um","uma","uns","umas",
        "para","por","em","com","sem","que","e","ou","no","na","nos","nas",
        "ao","aos","à","às","se","como","entre","sobre","ante","após","até","contra",
        "pelo","pela","pelos","pelas"
    }
    _FALLBACK_EN = {
        "the","a","an","of","in","on","and","or","to","for","with","by","from","as","at","is","are"
    }
    def limpar_texto(texto: str) -> str:
        if not isinstance(texto, str):
            return ""
        texto = texto.lower()
        texto = re.sub(r"http\S+|www\.\S+", " ", texto)
        texto = re.sub(r"[^0-9a-záéíóúâêîôûãõç\s\-\_/]", " ", texto, flags=re.IGNORECASE)
        texto = re.sub(r"\s+", " ", texto).strip()
        stop = _FALLBACK_PT | _FALLBACK_EN
        return " ".join([t for t in texto.split() if t not in stop])

# Carrega o modelo uma única vez (paraphrase-multilingual-mpnet-base-v2)
_MODEL: Optional[SentenceTransformer] = SentenceTransformer(
    "paraphrase-multilingual-mpnet-base-v2"
)

def gerar_embedding(
    texto: Optional[str] = None,
    *,
    texto_claims: Optional[str] = None,
    texto_descricao: Optional[str] = None,
    texto_resumo: Optional[str] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Gera embedding (768d) para um texto.
    Prioridade de campos: texto_claims > texto > texto_descricao > texto_resumo.

    Parâmetros
    ----------
    texto, texto_claims, texto_descricao, texto_resumo : str | None
        Forneça qualquer um; o primeiro não-vazio será usado.
    normalize : bool
        Se True (padrão), retorna vetor L2-normalizado (||v||=1), ideal para cos_sim.

    Retorna
    -------
    np.ndarray (float32) com shape (768,)
    """
    # escolhe a fonte de texto
    source = None
    for t in (texto_claims, texto, texto_descricao, texto_resumo):
        if isinstance(t, str) and t.strip():
            source = t
            break

    if not source:
        # vetor nulo (evita quebra)
        return np.zeros(768, dtype=np.float32)

    # limpeza leve (remove URLs, normaliza espaços, tira stopwords básicas)
    txt = limpar_texto(source)

    # embedding (sem normalizar aqui; normalizamos manualmente abaixo)
    vec = _MODEL.encode(
        txt,
        convert_to_numpy=True,
        normalize_embeddings=False
    ).astype(np.float32)

    if normalize:
        n = float(norm(vec))
        if n > 0.0:
            vec = vec / n

    return vec

def processar_texto(texto: str) -> str:
    """Compatibilidade: antes existia 'processar_texto'. Aqui mapeia para limpar_texto."""
    return limpar_texto(texto)
# ----------------------------------------------------------------------
