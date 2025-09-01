import unittest
from similaridade import calcular_similaridade, calcular_similaridade_01

class TestSimilaridade(unittest.TestCase):
    def test_percentual_aproxima_100_para_textos_iguais(self):
        t = "Método e dispositivo para classificação de eventos em fluxo."
        pct = calcular_similaridade(t, t)   # retorna % (0–100)
        self.assertGreaterEqual(pct, 99.0)

    def test_escala_0_1_e_ordem(self):
        a = "Sistema de detecção de anomalias em tempo real."
        b = "Método para detecção de anomalias em streams de dados."
        c = "Aparelho de cultivo de plantas com irrigação automática."
        s_ab = calcular_similaridade_01(a, b)
        s_ac = calcular_similaridade_01(a, c)
        # iguais > relacionados > diferentes
        self.assertGreaterEqual(calcular_similaridade_01(a, a), 0.99)
        self.assertGreater(s_ab, s_ac)
        # faixa [0,1]
        for s in (s_ab, s_ac):
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

if __name__ == "__main__":
    unittest.main()
