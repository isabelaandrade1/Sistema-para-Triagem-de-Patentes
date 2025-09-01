# similaridade.py — métricas em 0–1; exibição em %
import numpy as np
from processamento_texto import gerar_embedding

def _sim_cos_01(texto1: str, texto2: str) -> float:
    """
    Similaridade de cosseno em 0–1.
    Pressupõe que gerar_embedding retorna vetor L2-normalizado.
    """
    e1 = gerar_embedding(texto=texto1)
    e2 = gerar_embedding(texto=texto2)
    sim = float(np.dot(e1, e2))           # cos in [-1, 1]
    # clamp p/ estabilidade de métricas
    if sim < 0.0: sim = 0.0
    elif sim > 1.0: sim = 1.0
    return round(sim, 6)

def calcular_similaridade_01(texto1: str, texto2: str) -> float:
    """API para métricas (ROC/PR, zona cinza etc.): retorna 0–1."""
    return _sim_cos_01(texto1, texto2)

def calcular_similaridade(texto1: str, texto2: str) -> float:
    """API de exibição: retorna percentual 0–100 (apenas apresentação)."""
    return round(_sim_cos_01(texto1, texto2) * 100.0, 2)
