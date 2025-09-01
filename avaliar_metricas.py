import json
import csv
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    matthews_corrcoef,
    cohen_kappa_score
)
import matplotlib.pyplot as plt

# 📥 Carrega os pares com similaridade já calculada
with open("base_pares_avaliados.json", "r", encoding="utf-8") as f:
    pares = json.load(f)

y_true = []
y_scores = []
ids_pares = []

for par in pares:
    score = par.get("similaridade")
    label = par.get("label")

    if score is not None and label in [0, 1]:
        y_scores.append(score)
        y_true.append(label)
        ids_pares.append(f"{par['id1']} <-> {par['id2']}")

# 📌 Cálculo do threshold ótimo usando o índice de Youden (Youden's J statistic)
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
youden_index = tpr - fpr
optimal_idx = youden_index.argmax()
THRESHOLD = thresholds[optimal_idx]
print(f"\n📌 Threshold ótimo (Youden Index): {THRESHOLD:.2f}%")

# 🎯 Classificação binária com threshold
y_pred = [1 if score >= THRESHOLD else 0 for score in y_scores]

print("\n📊 Relatório de Classificação:")
print(classification_report(y_true, y_pred, digits=2))

# 🔲 Matriz de Confusão
cm = confusion_matrix(y_true, y_pred)
print("\n📉 Matriz de Confusão:")
print(cm)

# 📌 Outras Métricas
mcc = matthews_corrcoef(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
print(f"\n🔢 MCC (Coeficiente de Correlação de Matthews): {mcc:.2f}")
print(f"🧮 Kappa de Cohen: {kappa:.2f}")

# 💾 Exporta para CSV
with open("avaliacao_completa.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Par", "Label", "Similaridade", "Predição"])
    for id_par, label, score, pred in zip(ids_pares, y_true, y_scores, y_pred):
        writer.writerow([id_par, label, round(score, 2), pred])
print("✅ Resultados salvos em 'avaliacao_completa.csv'.")

# 📈 Curva ROC
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"Curva ROC (AUC = {roc_auc:.2f})", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.title("Curva ROC - Avaliação de Similaridade")
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.show()
