import unittest
import numpy as np
from processamento_texto import gerar_embedding

class TestProcessamentoTexto(unittest.TestCase):
    def test_gerar_embedding_dim_e_norma(self):
        texto = "Um sistema para triagem de patentes."
        v = gerar_embedding(texto_claims=texto, normalize=True)
        self.assertIsInstance(v, (list, np.ndarray))
        v = np.asarray(v, dtype=float)
        self.assertEqual(v.shape[-1], 768)     # dimensão do mpnet
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=3)  # L2

if __name__ == "__main__":
    unittest.main()
