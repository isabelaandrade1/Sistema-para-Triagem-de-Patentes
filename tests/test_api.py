import unittest
import numpy as np
from fastapi.testclient import TestClient

import main  
from processamento_texto import gerar_embedding

class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # injeta uma base pequena diretamente em main._EMB / main._META
        t1 = "Método de detecção de anomalias com modelo de aprendizado."
        t2 = "Sistema de irrigação para cultivo de plantas."
        e1 = gerar_embedding(texto_claims=t1, normalize=True).astype(np.float32)
        e2 = gerar_embedding(texto_claims=t2, normalize=True).astype(np.float32)
        main._EMB = np.vstack([e1, e2])
        main._META = [
            {"lens_id":"T-1","title":"Detecção de anomalias","claims":t1},
            {"lens_id":"T-2","title":"Irrigação automática","claims":t2},
        ]
        cls.client = TestClient(main.app)

    def test_status_root(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("itens_na_base", data)
        self.assertGreaterEqual(data["itens_na_base"], 2)

    def test_buscar_similares(self):
        payload = {"claims":"Aprendizado de máquina para detectar anomalias em fluxo.","top_k":1}
        r = self.client.post("/buscar_similares", json=payload)
        self.assertEqual(r.status_code, 200)
        res = r.json().get("resultados", [])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["lens_id"], "T-1")  # mais similar ao t1

if __name__ == "__main__":
    unittest.main()
