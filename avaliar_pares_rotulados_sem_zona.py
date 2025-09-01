import json
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho do arquivo de entrada
CAMINHO_ARQUIVO = "pares_rotulados_base_teste_manual.json"

# Lê os pares com similaridade
with open(CAMINHO_ARQUIVO, "r", encoding="utf-8") as f:
    pares = json.load(f)

# Prepara os dados
y_true = [par["label"] for par in pares]
y_scores = [par["similaridade"] for par in pares]
y_pred = [1 if score >= 0.5 else 0 for score in y_scores]  # Limiar fixo: 0.5

# Avaliação
print("📊 Avaliação dos Pares Rotulados (todos os pares):")
print(f"• Total de pares avaliados: {len(pares)}")
print("• Acurácia:", round(sum([p == t for p, t in zip(y_pred, y_true)]) / len(y_true), 2))
print(classification_report(y_true, y_pred, digits=2))

# Matriz de confusão
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Diferente", "Similar"], yticklabels=["Diferente", "Similar"])
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.title("Matriz de Confusão")
plt.tight_layout()
plt.show()

# Curva ROC
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"Curva ROC (AUC = {roc_auc:.2f})", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("Falsos Positivos (FPR)")
plt.ylabel("Verdadeiros Positivos (TPR)")
plt.title("Curva ROC - Similaridade")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# Histograma
plt.figure(figsize=(6, 4))
similares = [par["similaridade"] for par in pares if par["label"] == 1]
diferentes = [par["similaridade"] for par in pares if par["label"] == 0]
plt.hist(similares, bins=10, alpha=0.6, label="Similares (label=1)", color="green")
plt.hist(diferentes, bins=10, alpha=0.6, label="Diferentes (label=0)", color="red")
plt.axvline(0.5, color='black', linestyle='--', label="Limiar = 0.5")
plt.title("Distribuição das Similaridades")
plt.xlabel("Score de Similaridade (%)")
plt.ylabel("Frequência")
plt.legend()
plt.tight_layout()
plt.show()

print("\n✅ Avaliação completa.")
