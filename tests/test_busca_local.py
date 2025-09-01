# tests/test_busca_local.py
import unittest
import json
import tempfile
import os

from processamento_texto import gerar_embedding
import buscar_similares as bs  # usa CAMINHO_ARQUIVO dinamicamente

def _score(item) -> float:
    """Extrai o score do dict, aceitando diferentes chaves."""
    if "similaridade" in item:
        return float(item["similaridade"])              # 0–1
    if "score" in item:
        return float(item["score"])                     # 0–1
    if "similaridade_percentual" in item:
        return float(item["similaridade_percentual"]) / 100.0
    raise KeyError("Resultado sem campo de similaridade: esperava 'similaridade' ou 'score'.")

class TestBuscaLocal(unittest.TestCase):
    def setUp(self):
        doc1_claims = "Método para detectar anomalias em fluxos de eventos usando aprendizado de máquina."
        doc2_claims = "Dispositivo para cultivo automatizado de plantas em estufa doméstica."

        emb1 = gerar_embedding(texto_claims=doc1_claims, normalize=True).tolist()
        emb2 = gerar_embedding(texto_claims=doc2_claims, normalize=True).tolist()

        base = [
            {"lens_id": "DOC-1", "title": "Detecção de Anomalias", "claims": doc1_claims, "embedding": emb1},
            {"lens_id": "DOC-2", "title": "Cultivo Automatizado", "claims": doc2_claims, "embedding": emb2},
        ]

        tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json", encoding="utf-8")
        json.dump(base, tmp, ensure_ascii=False, indent=2)
        self.tmp_path = tmp.name
        tmp.close()

        bs.CAMINHO_ARQUIVO = self.tmp_path  # aponta o módulo p/ esta base

    def tearDown(self):
        try:
            os.unlink(self.tmp_path)
        except OSError:
            pass

    def test_topk_ordenado_e_coerente(self):
        consulta = "Sistema de aprendizado de máquina para detectar anomalias em streams."
        resultados = bs.buscar_similares(consulta, top_k=2)

        self.assertEqual(len(resultados), 2)
        # O mais similar deve ser o DOC-1
        self.assertEqual(resultados[0]["lens_id"], "DOC-1")

        s0 = _score(resultados[0])
        s1 = _score(resultados[1])
        self.assertGreaterEqual(s0, s1)  # ordenação decrescente

if __name__ == "__main__":
    unittest.main()
