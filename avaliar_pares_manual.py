# avaliar_pares_manual.py
# © 2025 — Avaliação robusta de similaridade de patentes (claims)
# Saídas: métricas completas, curvas ROC/PR, sweep de limiar, bootstrap de ICs,
#         arquivos CSV/JSON e figuras, + análise detalhada da zona cinza.
# Calibração: APENAS piecewise (50–70) com ECE e diagrama de confiabilidade.

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # garante backend não interativo
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
)
from sklearn.calibration import calibration_curve

# --- Importa sua função de similaridade (mantida) ---
from similaridade import calcular_similaridade


# --------------------------
# Utilidades de visualização/IO
# --------------------------
def _ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def save_confusion_matrix(cm, labels, out_path: Path, title="Matriz de Confusão"):
    _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(5.2, 4.5), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=[f"Pred: {l}" for l in labels],
        yticklabels=[f"Real: {l}" for l in labels],
        ylabel="Classe real",
        xlabel="Classe predita",
        title=title,
    )
    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def save_classification_report_png(report_dict, out_path: Path, title="Relatório de Classificação"):
    """
    report_dict = classification_report(..., output_dict=True)
    Salva uma tabela bonitinha como PNG.
    """
    _ensure_parent(out_path)
    df = pd.DataFrame(report_dict).T
    cols = [c for c in ["precision", "recall", "f1-score", "support", "accuracy"] if c in df.columns]
    df = df[cols].copy()

    def fmt(x):
        if isinstance(x, (int, np.integer)): return f"{x:d}"
        try:
            return f"{float(x):.2f}"
        except Exception:
            return str(x)

    fig, ax = plt.subplots(figsize=(10, 0.6 + 0.35 * len(df)), dpi=150)
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=10)
    tbl = ax.table(
        cellText=df.map(fmt).values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.15)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return df  # útil para salvar CSV também

def save_metric_curve(x, y, xlabel, ylabel, title, out_path: Path):
    _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)
    ax.plot(x, y, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

# --------------------------
# Métricas & CIs
# --------------------------
def bootstrap_ci_metric(y_true, y_pred, metric_fn, n_boot=1000, seed=42):
    """IC bootstrap (95%) para uma métrica baseada em y_pred (discreto)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    base = float(metric_fn(y_true, y_pred))
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(float(metric_fn(y_true[idx], y_pred[idx])))
    low, high = np.percentile(boots, [2.5, 97.5])
    return (base, low, high)

def bootstrap_ci_auc(y_true, y_scores, n_boot=1000, seed=42):
    """IC bootstrap 95% para ROC AUC, PR AUC e AP."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n = len(y_true)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc_base = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc_base = auc(recall, precision)
    ap_base = average_precision_score(y_true, y_scores)

    roc_list, pr_list, ap_list = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_scores[idx]
        if len(np.unique(yt)) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(yt, ys)
        roc_list.append(auc(fpr_b, tpr_b))
        prec_b, rec_b, _ = precision_recall_curve(yt, ys)
        pr_list.append(auc(rec_b, prec_b))
        ap_list.append(average_precision_score(yt, ys))

    def pct_ci(vals):
        if len(vals) == 0:
            return (np.nan, np.nan)
        return tuple(np.percentile(vals, [2.5, 97.5]))

    roc_li, roc_ls = pct_ci(roc_list)
    pr_li, pr_ls = pct_ci(pr_list)
    ap_li, ap_ls = pct_ci(ap_list)

    return {
        "roc_auc": (roc_auc_base, roc_li, roc_ls),
        "pr_auc":  (pr_auc_base,  pr_li,  pr_ls),
        "ap":      (ap_base,      ap_li,  ap_ls),
    }

def sweep_threshold(y_true, y_scores, step=0.5):
    """Varre limiares únicos (0–100) para predição binária (>= t -> 1)."""
    rows = []
    for t in np.arange(0.0, 100.0 + 1e-9, step):
        pred = (y_scores >= t).astype(int)
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else np.nan
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = (2 * precision * recall) / (precision + recall) if (precision and recall) else np.nan
        tpr = recall
        fpr = fp / (fp + tn) if (fp + tn) else np.nan
        youden_j = (tpr - fpr) if (np.isfinite(tpr) and np.isfinite(fpr)) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        bacc = (tpr + spec) / 2 if (np.isfinite(tpr) and np.isfinite(spec)) else np.nan
        mcc = matthews_corrcoef(y_true, pred) if np.isfinite(acc) else np.nan
        kappa = cohen_kappa_score(y_true, pred) if np.isfinite(acc) else np.nan
        npv = tn / (tn + fn) if (tn + fn) else np.nan
        prevalence = (tp + fn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else np.nan

        rows.append({
            "threshold": t,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
            "specificity": spec, "balanced_accuracy": bacc, "youden_j": youden_j,
            "mcc": mcc, "kappa": kappa, "npv": npv, "prevalence": prevalence
        })
    return pd.DataFrame(rows)

# --------- Calibração piecewise 50–70 ----------
def piecewise_prob(scores, low=50.0, high=70.0, p_low=0.05, p_high=0.95):
    """
    Mapeia scores (%) -> probabilidade [0,1] compatível com a política:
    score < low -> p_low; score >= high -> p_high; entre [low, high) -> interpolação linear.
    """
    s = np.asarray(scores, dtype=float)
    p = np.empty_like(s, dtype=float)
    p[s < low] = p_low
    p[s >= high] = p_high
    mid = (s >= low) & (s < high)
    p[mid] = p_low + (p_high - p_low) * (s[mid] - low) / (high - low)
    return p

def ece_quantile(y_true, y_prob, n_bins=10):
    """Expected Calibration Error (ECE) com bins por quantis (evita bins vazios)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(y_prob, qs)
    edges = np.unique(edges)
    if len(edges) <= 2:
        return 0.0
    ece = 0.0
    total = len(y_prob)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i+1]
        if i < len(edges) - 2:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        conf = y_prob[mask].mean()
        acc = (y_true[mask] == (y_prob[mask] >= 0.5).astype(int)).mean()
        ece += (mask.sum() / total) * abs(acc - conf)
    return float(ece)

# --------------------------
# Script principal
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Avaliação robusta de similaridade (claims).")
    parser.add_argument("--input", default="base_100_patentes.json", help="Arquivo de entrada com pares e rótulos.")
    parser.add_argument("--out_json", default="pares_validos_classificados.json", help="JSON de saída com resultados.")
    parser.add_argument("--out_dir", default="relatorios", help="Pasta para figuras/CSVs.")
    parser.add_argument("--limiar_similar", type=float, default=70.0, help="Similar se score >= limiar_similar.")
    parser.add_argument("--limiar_diferente", type=float, default=50.0, help="Diferente se score < limiar_diferente.")
    parser.add_argument("--bootstrap", type=int, default=1000, help="n_boot para ICs (0 desativa).")
    parser.add_argument("--sweep_step", type=float, default=0.5, help="Passo do sweep de limiar (em %).")
    parser.add_argument("--calib_bins", type=int, default=10, help="Nº de bins (quantis) para confiabilidade.")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f:
        pares = json.load(f)

    y_true_bin, y_scores_bin, y_pred_bin = [], [], []
    y_true_full, y_scores_full = [], []  # todos os válidos (para curvas “inclui/estrito”)
    pares_validos, pares_invalidos, indefinidos = [], [], []

    print("🔍 Avaliando similaridade dos pares...\n")

    for par in pares:
        claims1 = par.get("claims1")
        claims2 = par.get("claims2")
        id1, id2 = par.get("id1"), par.get("id2")

        if not (claims1 and claims2):
            pares_invalidos.append((id1, id2))
            continue

        score = float(calcular_similaridade(claims1, claims2))  # 0–100 esperado
        label = int(par["label"])

        print(f"• {id1} <-> {id2} → Similaridade: {score:.2f}%")

        # Política: similar ≥ 70 ; diferente < 50 ; 50–70 = zona cinza
        if score >= args.limiar_similar:
            pred = 1
        elif score < args.limiar_diferente:
            pred = 0
        else:
            pred = None  # zona cinza

        pares_validos.append({
            "id1": id1, "id2": id2,
            "label": label,
            "similaridade": score,
            "predito": pred
        })

        # Guardar para análises completas
        y_true_full.append(label)
        y_scores_full.append(score)

        # Fora da zona cinza
        if pred is not None:
            y_true_bin.append(label)
            y_scores_bin.append(score)
            y_pred_bin.append(pred)
        else:
            indefinidos.append((id1, id2, score, label))

    print(f"\n✅ Total de pares avaliados: {len(pares_validos)}")
    print(f"❗ Pares ignorados por falta de dados: {len(pares_invalidos)}")
    print(f"⚠️ Casos indefinidos (zona cinza: {args.limiar_diferente}% ≤ score < {args.limiar_similar}%): {len(indefinidos)}")

    # Salva JSON/CSVs
    out_json = Path(args.out_json)
    _ensure_parent(out_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(pares_validos, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(pares_validos)
    df.to_csv(out_dir / "pares_todos.csv", index=False, encoding="utf-8")
    df_cls = df[df["predito"].notna()].copy()
    df_cls.to_csv(out_dir / "pares_classificados.csv", index=False, encoding="utf-8")
    if len(indefinidos) > 0:
        pd.DataFrame(indefinidos, columns=["id1", "id2", "score", "label"]).to_csv(
            out_dir / "pares_indefinidos.csv", index=False, encoding="utf-8"
        )

    # --------------------------
    # MÉTRICAS — fora da zona cinza + figuras
    # --------------------------
    print("\n📊 Relatório de Classificação (apenas pares fora da zona cinza):")
    if len(df_cls) == 0 or len(set(df_cls["label"])) < 2:
        print("⚠️ Amostra insuficiente para métricas binárias.")
    else:
        y_true_bin = np.array(y_true_bin, dtype=int)
        y_pred_bin = np.array(y_pred_bin, dtype=int)
        y_scores_bin = np.array(y_scores_bin, dtype=float)
        report_txt = classification_report(y_true_bin, y_pred_bin, digits=2)
        print(report_txt)
        report_dict = classification_report(y_true_bin, y_pred_bin, output_dict=True)
        report_df_png = save_classification_report_png(
            report_dict, out_dir / "classification_report.png",
            title="Relatório de Classificação — Fora da Zona Cinza"
        )
        report_df_png.to_csv(out_dir / "classification_report.csv", index=True, encoding="utf-8")

        cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        print("📉 Matriz de Confusão:")
        print(cm)
        save_confusion_matrix(cm, labels=[0, 1], out_path=out_dir / "confusion_matrix.png",
                              title="Matriz de Confusão — Fora da Zona Cinza")

        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        npv = tn / (tn + fn) if (tn + fn) else np.nan
        bacc = balanced_accuracy_score(y_true_bin, y_pred_bin)
        mcc = matthews_corrcoef(y_true_bin, y_pred_bin)
        kappa = cohen_kappa_score(y_true_bin, y_pred_bin)

        resumo = {
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "specificity": spec, "npv": npv, "balanced_accuracy": bacc,
            "mcc": mcc, "kappa": kappa
        }
        pd.DataFrame([resumo]).to_csv(out_dir / "metricas_extras.csv", index=False, encoding="utf-8")

        # --------------------------
        # Curvas ROC/PR + ICs (fora da cinza)
        # --------------------------
        scores01 = y_scores_bin / 100.0
        fpr, tpr, _ = roc_curve(y_true_bin, scores01)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true_bin, scores01)
        pr_auc = auc(recall, precision)
        ap = average_precision_score(y_true_bin, scores01)

        print(f"\n📈 ROC AUC: {roc_auc:.4f}")
        print(f"📈 PR AUC:  {pr_auc:.4f}")
        print(f"📈 Average Precision (AP): {ap:.4f}")

        if args.bootstrap > 0:
            ci = bootstrap_ci_auc(y_true_bin, scores01, n_boot=args.bootstrap, seed=42)
            (roc_c, roc_li, roc_ls) = ci["roc_auc"]
            (pr_c, pr_li, pr_ls)   = ci["pr_auc"]
            (ap_c, ap_li, ap_ls)   = ci["ap"]
            print(f"   ROC AUC 95% CI: [{roc_li:.4f}, {roc_ls:.4f}]")
            print(f"   PR  AUC 95% CI: [{pr_li:.4f}, {pr_ls:.4f}]")
            print(f"   AP      95% CI: [{ap_li:.4f}, {ap_ls:.4f}]")

        fig = plt.figure(figsize=(6, 4), dpi=150)
        plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("Falsos Positivos")
        plt.ylabel("Verdadeiros Positivos")
        plt.title("Curva ROC - (fora da zona cinza)")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / "roc_curve.png"); plt.close(fig)

        fig = plt.figure(figsize=(6, 4), dpi=150)
        plt.plot(recall, precision, linewidth=2, label=f"AUC = {pr_auc:.2f} | AP = {ap:.2f}")
        plt.xlabel("Recall")
        plt.ylabel("Precisão")
        plt.title("Curva Precision-Recall (fora da zona cinza)")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / "pr_curve.png"); plt.close(fig)

        # Hist classes
        similares = [s for s, l in zip(y_scores_bin, y_true_bin) if l == 1]
        diferentes = [s for s, l in zip(y_scores_bin, y_true_bin) if l == 0]
        fig = plt.figure(figsize=(6, 4), dpi=150)
        plt.hist(similares, bins=10, alpha=0.6, label="Similares (label=1)")
        plt.hist(diferentes, bins=10, alpha=0.6, label="Diferentes (label=0)")
        plt.axvline(args.limiar_similar, linestyle='--', label=f"Limiar Similar = {args.limiar_similar}%")
        plt.axvline(args.limiar_diferente, linestyle='--', label=f"Limiar Diferente = {args.limiar_diferente}%")
        plt.title("Distribuição das Similaridades (fora da zona cinza)")
        plt.xlabel("Score de Similaridade (%)")
        plt.ylabel("Frequência")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "hist_similaridades.png"); plt.close(fig)

        # --------------------------
        # Calibração piecewise (50–70)
        # --------------------------
        probs_piece = piecewise_prob(y_scores_bin, low=args.limiar_diferente, high=args.limiar_similar)
        ece_pw = ece_quantile(y_true_bin, probs_piece, n_bins=args.calib_bins)
        print(f"\n📏 ECE (piecewise 50–70, {args.calib_bins} bins por quantis): {ece_pw:.4f}")

        prob_true, prob_pred = calibration_curve(
            y_true_bin, probs_piece, n_bins=args.calib_bins, strategy="quantile"
        )
        fig = plt.figure(figsize=(6, 4), dpi=150)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Modelo (piecewise 50–70)')
        plt.plot([0, 1], [0, 1], '--', label='Perfeitamente calibrado')
        plt.xlabel('Confiança média no bin')
        plt.ylabel('Fração positiva')
        plt.title('Confiabilidade (piecewise 50–70) — fora da zona cinza')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / "reliability_piecewise.png"); plt.close(fig)

        # Sweep de limiar (sem zona cinza)
        sweep_df = sweep_threshold(y_true_bin, y_scores_bin, step=args.sweep_step)
        sweep_df.to_csv(out_dir / "sweep_threshold.csv", index=False, encoding="utf-8")

        if not sweep_df.empty:
            save_metric_curve(sweep_df["threshold"], sweep_df["f1"],
                              "Threshold (%)", "F1",
                              "F1 vs Threshold (fora da zona cinza)",
                              out_path=out_dir / "threshold_f1.png")
            save_metric_curve(sweep_df["threshold"], sweep_df["mcc"],
                              "Threshold (%)", "MCC",
                              "MCC vs Threshold (fora da zona cinza)",
                              out_path=out_dir / "threshold_mcc.png")
            save_metric_curve(sweep_df["threshold"], sweep_df["youden_j"],
                              "Threshold (%)", "Youden J",
                              "Youden J vs Threshold (fora da zona cinza)",
                              out_path=out_dir / "threshold_youdenj.png")

        if not sweep_df.empty:
            best_f1 = sweep_df.loc[sweep_df["f1"].idxmax()]
            best_yj = sweep_df.loc[sweep_df["youden_j"].idxmax()]
            best_mcc = sweep_df.loc[sweep_df["mcc"].idxmax()]
            resumo_thr = pd.DataFrame([
                {"criterio": "F1 máximo", "threshold": best_f1["threshold"], "f1": best_f1["f1"],
                 "precision": best_f1["precision"], "recall": best_f1["recall"], "mcc": best_f1["mcc"],
                 "youden_j": best_f1["youden_j"], "balanced_accuracy": best_f1["balanced_accuracy"], "accuracy": best_f1["accuracy"]},
                {"criterio": "Youden J máximo", "threshold": best_yj["threshold"], "youden_j": best_yj["youden_j"],
                 "balanced_accuracy": best_yj["balanced_accuracy"], "mcc": best_yj["mcc"],
                 "f1": best_yj["f1"], "precision": best_yj["precision"], "recall": best_yj["recall"], "accuracy": best_yj["accuracy"]},
                {"criterio": "MCC máximo", "threshold": best_mcc["threshold"], "mcc": best_mcc["mcc"],
                 "accuracy": best_mcc["accuracy"], "f1": best_mcc["f1"],
                 "precision": best_mcc["precision"], "recall": best_mcc["recall"],
                 "youden_j": best_mcc["youden_j"], "balanced_accuracy": best_mcc["balanced_accuracy"]}
            ])
            resumo_thr.to_csv(out_dir / "melhores_thresholds.csv", index=False, encoding="utf-8")
            print("\n🔎 Melhores thresholds (sem zona cinza):")
            print(resumo_thr)

        # ICs (bootstrap) p/ métricas discretas atuais
        if args.bootstrap > 0 and len(y_true_bin) > 0:
            def _acc(a, b): return np.mean(a == b)
            acc_c, acc_li, acc_ls = bootstrap_ci_metric(y_true_bin, y_pred_bin, _acc, n_boot=args.bootstrap)
            def _f1_weighted(a, b): return classification_report(a, b, output_dict=True)["weighted avg"]["f1-score"]
            f1_c, f1_li, f1_ls = bootstrap_ci_metric(y_true_bin, y_pred_bin, _f1_weighted, n_boot=args.bootstrap)
            mcc_c, mcc_li, mcc_ls = bootstrap_ci_metric(y_true_bin, y_pred_bin, matthews_corrcoef, n_boot=args.bootstrap)
            kap_c, kap_li, kap_ls = bootstrap_ci_metric(y_true_bin, y_pred_bin, cohen_kappa_score, n_boot=args.bootstrap)
            ci_df = pd.DataFrame([
                {"metrica": "Accuracy", "valor": acc_c, "li95": acc_li, "ls95": acc_ls},
                {"metrica": "F1 (weighted)", "valor": f1_c, "li95": f1_li, "ls95": f1_ls},
                {"metrica": "MCC", "valor": mcc_c, "li95": mcc_li, "ls95": mcc_ls},
                {"metrica": "Kappa", "valor": kap_c, "li95": kap_li, "ls95": kap_ls},
                {"metrica": "ROC AUC", "valor": roc_auc},
                {"metrica": "PR AUC", "valor": pr_auc},
                {"metrica": "Average Precision", "valor": ap},
            ])
            ci_df.to_csv(out_dir / "intervalos_confianca.csv", index=False, encoding="utf-8")

    # --------------------------
    # Erros com regra rígida (ignora cinza)
    # --------------------------
    falsos_positivos = [p for p in pares_validos if p["label"] == 0 and p["similaridade"] >= args.limiar_similar]
    falsos_negativos = [p for p in pares_validos if p["label"] == 1 and p["similaridade"] < args.limiar_diferente]
    pd.DataFrame(falsos_positivos).to_csv(out_dir / "falsos_positivos.csv", index=False, encoding="utf-8")
    pd.DataFrame(falsos_negativos).to_csv(out_dir / "falsos_negativos.csv", index=False, encoding="utf-8")
    print(f"\n⚠️ Falsos positivos (fora da zona cinza): {len(falsos_positivos)}")
    print(f"⚠️ Falsos negativos (fora da zona cinza): {len(falsos_negativos)}")

    # --------------------------
    # *** ZONA CINZA ***
    # --------------------------
    if len(indefinidos) > 0:
        df_grey = pd.DataFrame(indefinidos, columns=["id1", "id2", "score", "label"])
        df_grey["dist_70"] = abs(args.limiar_similar - df_grey["score"])
        df_grey["dist_50"] = abs(df_grey["score"] - args.limiar_diferente)
        df_grey["mais_proximo"] = np.where(df_grey["dist_70"] <= df_grey["dist_50"], "70% (similar)", "50% (diferente)")
        df_grey["pred_potencial"] = np.where(df_grey["mais_proximo"] == "70% (similar)", 1, 0)
        df_grey["acerto_potencial"] = (df_grey["pred_potencial"] == df_grey["label"]).astype(int)

        total = len(df)
        total_grey = len(df_grey)
        pct_grey = 100.0 * total_grey / total if total > 0 else 0.0
        acertos_pot = int(df_grey["acerto_potencial"].sum())
        pct_acerto_pot = 100.0 * acertos_pot / total_grey if total_grey > 0 else np.nan

        print("\n🟨 Zona cinza — resumo:")
        print(f"   • Quantidade: {total_grey} ({pct_grey:.2f}% do total)")
        print(f"   • Acerto potencial (lado mais próximo): {acertos_pot}/{total_grey} ({pct_acerto_pot:.2f}%)")

        df_grey.to_csv(out_dir / "zona_cinza_detalhe.csv", index=False, encoding="utf-8")

        resumo_grey = (df_grey.groupby(["mais_proximo", "label"]).size().reset_index(name="contagem"))
        resumo_grey_total = (df_grey.groupby("mais_proximo").size().reset_index(name="total_cat"))
        resumo_grey = resumo_grey.merge(resumo_grey_total, on="mais_proximo", how="left")
        resumo_grey["perc_na_categoria"] = 100.0 * resumo_grey["contagem"] / resumo_grey["total_cat"]
        resumo_grey.to_csv(out_dir / "zona_cinza_resumo.csv", index=False, encoding="utf-8")

        # Hist zona cinza
        fig = plt.figure(figsize=(6, 4), dpi=150)
        plt.hist(df_grey[df_grey["label"] == 1]["score"], bins=8, alpha=0.6, label="Label=1 (similares)")
        plt.hist(df_grey[df_grey["label"] == 0]["score"], bins=8, alpha=0.6, label="Label=0 (diferentes)")
        plt.axvline(args.limiar_diferente, linestyle='--', label=f"{args.limiar_diferente:.0f}%")
        plt.axvline(args.limiar_similar, linestyle='--', label=f"{args.limiar_similar:.0f}%")
        plt.title("Zona Cinza (50–70%) — Distribuição dos Scores")
        plt.xlabel("Score de Similaridade (%)")
        plt.ylabel("Frequência")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "grey_hist.png"); plt.close(fig)

        # Barras empilhadas
        pivot = df_grey.pivot_table(index="mais_proximo", columns="label", values="score", aggfunc="count", fill_value=0)
        for col in [0, 1]:
            if col not in pivot.columns:
                pivot[col] = 0
        pivot = pivot[[0, 1]].sort_index()

        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        pivot.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Zona Cinza — Proporção por Lado mais Próximo e Label")
        ax.set_xlabel("Lado mais próximo")
        ax.set_ylabel("Contagem")
        fig.tight_layout()
        fig.savefig(out_dir / "grey_stacked.png"); plt.close(fig)

        # Scatter com faixa 50–70
        try:
            fig = plt.figure(figsize=(7, 4), dpi=150)
            x = np.arange(len(df_grey))
            plt.axhspan(args.limiar_diferente, args.limiar_similar, alpha=0.15)
            cores = np.where(df_grey["label"] == 1, "tab:green", "tab:red")
            plt.scatter(x, df_grey["score"], c=cores)
            plt.xticks([])
            plt.xlabel("Pares na zona cinza (índice)")
            plt.ylabel("Score (%)")
            plt.title("Zona Cinza — Dispersão dos Scores por Label")
            plt.tight_layout()
            fig.savefig(out_dir / "grey_scatter.png"); plt.close(fig)
        except Exception as e:
            print(f"⚠️ Falha ao gerar scatter da zona cinza: {e}")

        lado_counts = df_grey["mais_proximo"].value_counts().to_dict()
        print(f"   • Próximo a 50%: {lado_counts.get('50% (diferente)', 0)} | Próximo a 70%: {lado_counts.get('70% (similar)', 0)}")

        # --------------------------
        # (Opcional) Métricas INCLUINDO a zona cinza (lado mais próximo) + imagens
        # --------------------------
        y_true_inc = df["label"].astype(int).to_numpy()
        scores_inc = df["similaridade"].astype(float).to_numpy()

        pred_inc = []
        for s in scores_inc:
            if s >= args.limiar_similar:
                pred_inc.append(1)
            elif s < args.limiar_diferente:
                pred_inc.append(0)
            else:
                d70 = abs(args.limiar_similar - s)
                d50 = abs(s - args.limiar_diferente)
                pred_inc.append(1 if d70 <= d50 else 0)
        pred_inc = np.array(pred_inc, dtype=int)

        print("\n🧮 Métricas INCLUINDO a zona cinza (lado mais próximo 50–70).")
        report_inc_txt = classification_report(y_true_inc, pred_inc, digits=2)
        print(report_inc_txt)
        report_inc_dict = classification_report(y_true_inc, pred_inc, output_dict=True)
        df_inc_png = save_classification_report_png(
            report_inc_dict, out_dir / "classification_report_including_grey.png",
            title="Relatório de Classificação — Incluindo Zona Cinza (lado mais próximo)"
        )
        df_inc_png.to_csv(out_dir / "classification_report_including_grey.csv", index=True, encoding="utf-8")

        cm_inc = confusion_matrix(y_true_inc, pred_inc, labels=[0, 1])
        print("📉 Matriz de Confusão (inclui zona cinza):")
        print(cm_inc)
        save_confusion_matrix(cm_inc, labels=[0, 1], out_path=out_dir / "confusion_matrix_including_grey.png",
                              title="Matriz de Confusão — Incluindo Zona Cinza (lado mais próximo)")

        # Curvas com TODOS os pares válidos
        y_true_full_arr = np.array(y_true_full, dtype=int)
        scores_full01 = np.array(y_scores_full, dtype=float) / 100.0
        fpr_f, tpr_f, _ = roc_curve(y_true_full_arr, scores_full01)
        roc_auc_f = auc(fpr_f, tpr_f)
        prec_f, rec_f, _ = precision_recall_curve(y_true_full_arr, scores_full01)
        pr_auc_f = auc(rec_f, prec_f)
        ap_f = average_precision_score(y_true_full_arr, scores_full01)
        print(f"\n📈 ROC AUC (inclui zona cinza): {roc_auc_f:.4f}")
        print(f"📈 PR  AUC (inclui zona cinza): {pr_auc_f:.4f}")
        print(f"📈 AP      (inclui zona cinza): {ap_f:.4f}")

        fig = plt.figure(figsize=(6, 4), dpi=150)
        plt.plot(fpr_f, tpr_f, linewidth=2, label=f"AUC = {roc_auc_f:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("Falsos Positivos")
        plt.ylabel("Verdadeiros Positivos")
        plt.title("Curva ROC — Incluindo Zona Cinza")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / "roc_curve_including_grey.png"); plt.close(fig)

        fig = plt.figure(figsize=(6, 4), dpi=150)
        plt.plot(rec_f, prec_f, linewidth=2, label=f"AUC = {pr_auc_f:.2f} | AP = {ap_f:.2f}")
        plt.xlabel("Recall")
        plt.ylabel("Precisão")
        plt.title("Curva Precision-Recall — Incluindo Zona Cinza")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / "pr_curve_including_grey.png"); plt.close(fig)

    # --------------------------
    # *** NOVO *** — MÉTRICAS MODO ESTRITO (ZONA CINZA = ERRO)
    # --------------------------
    # Aqui, qualquer par na zona cinza é contado como erro deliberadamente:
    # para forçar o erro, definimos a predição como o oposto do rótulo do par cinza.
    y_true_strict = []
    y_pred_strict = []
    for p in pares_validos:
        y_true_strict.append(int(p["label"]))
        s = float(p["similaridade"])
        if s >= args.limiar_similar:
            y_pred_strict.append(1)
        elif s < args.limiar_diferente:
            y_pred_strict.append(0)
        else:
            # ZONA CINZA: conta como errado => inverte o rótulo para garantir erro
            y_pred_strict.append(1 - int(p["label"]))

    y_true_strict = np.array(y_true_strict, dtype=int)
    y_pred_strict = np.array(y_pred_strict, dtype=int)

    print("\n🧷 MÉTRICAS — MODO ESTRITO (zona cinza = erro):")
    report_strict_txt = classification_report(y_true_strict, y_pred_strict, digits=2)
    print(report_strict_txt)
    report_strict_dict = classification_report(y_true_strict, y_pred_strict, output_dict=True)
    df_strict_png = save_classification_report_png(
        report_strict_dict, out_dir / "classification_report_STRICT.png",
        title="Relatório de Classificação — MODO ESTRITO (zona cinza = erro)"
    )
    df_strict_png.to_csv(out_dir / "classification_report_STRICT.csv", index=True, encoding="utf-8")

    cm_strict = confusion_matrix(y_true_strict, y_pred_strict, labels=[0, 1])
    print("📉 Matriz de Confusão — MODO ESTRITO:")
    print(cm_strict)
    save_confusion_matrix(cm_strict, labels=[0, 1],
                          out_path=out_dir / "confusion_matrix_STRICT.png",
                          title="Matriz de Confusão — MODO ESTRITO (zona cinza = erro)")

    # --- NOVO: Curvas ROC/PR (usando todos os pares e os mesmos scores 0–1) ---
    scores_full01 = np.array(y_scores_full, dtype=float) / 100.0
    fpr_s, tpr_s, _ = roc_curve(y_true_strict, scores_full01)
    roc_auc_s = auc(fpr_s, tpr_s)
    prec_s, rec_s, _ = precision_recall_curve(y_true_strict, scores_full01)
    pr_auc_s = auc(rec_s, prec_s)
    ap_s = average_precision_score(y_true_strict, scores_full01)

    fig = plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(fpr_s, tpr_s, linewidth=2, label=f"AUC = {roc_auc_s:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Falsos Positivos")
    plt.ylabel("Verdadeiros Positivos")
    plt.title("Curva ROC — MODO ESTRITO (zona cinza = erro)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "roc_curve_STRICT.png"); plt.close(fig)

    fig = plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(rec_s, prec_s, linewidth=2, label=f"AUC = {pr_auc_s:.2f} | AP = {ap_s:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precisão")
    plt.title("Curva Precision-Recall — MODO ESTRITO (zona cinza = erro)")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "pr_curve_STRICT.png"); plt.close(fig)

    # --- NOVO: Scatter destacando quais pontos viram ERRO no modo estrito ---
    # Correto/Errado sob a política estrita:
    correct_strict = (y_true_strict == y_pred_strict)
    fig = plt.figure(figsize=(7, 4), dpi=150)
    x = np.arange(len(y_scores_full))
    plt.axhspan(50.0, 70.0, alpha=0.15, label="Zona cinza (contada como erro)")
    colors = np.where(correct_strict, "tab:green", "tab:red")
    plt.scatter(x, y_scores_full, c=colors)
    plt.xticks([])
    plt.xlabel("Pares (índice)")
    plt.ylabel("Score de Similaridade (%)")
    plt.title("MODO ESTRITO — Pontos corretos (verde) vs. erro (vermelho)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "strict_error_scatter.png"); plt.close(fig)

    # --- NOVO: Comparativo de métricas entre modos (barras) ---
    # Modo 1: Fora da cinza (se existir)
    metrics_rows = []
    if len(df_cls) > 0:
        acc1 = np.mean(y_true_bin == y_pred_bin)
        macro_f1_1 = classification_report(y_true_bin, y_pred_bin, output_dict=True)["macro avg"]["f1-score"]
        mcc1 = matthews_corrcoef(y_true_bin, y_pred_bin)
        kap1 = cohen_kappa_score(y_true_bin, y_pred_bin)
        metrics_rows.append({"modo": "Fora da cinza", "accuracy": acc1, "macro_f1": macro_f1_1, "mcc": mcc1, "kappa": kap1})

    # Modo 2: Inclui cinza (lado mais próximo) — se já calculado acima
    if "pred_inc" in locals():
        acc2 = np.mean(y_true_inc == pred_inc)
        macro_f1_2 = classification_report(y_true_inc, pred_inc, output_dict=True)["macro avg"]["f1-score"]
        mcc2 = matthews_corrcoef(y_true_inc, pred_inc)
        kap2 = cohen_kappa_score(y_true_inc, pred_inc)
        metrics_rows.append({"modo": "Inclui cinza", "accuracy": acc2, "macro_f1": macro_f1_2, "mcc": mcc2, "kappa": kap2})

    # Modo 3: Estrito (cinza = erro)
    acc3 = np.mean(y_true_strict == y_pred_strict)
    macro_f1_3 = classification_report(y_true_strict, y_pred_strict, output_dict=True)["macro avg"]["f1-score"]
    mcc3 = matthews_corrcoef(y_true_strict, y_pred_strict)
    kap3 = cohen_kappa_score(y_true_strict, y_pred_strict)
    metrics_rows.append({"modo": "Estrito (erro)", "accuracy": acc3, "macro_f1": macro_f1_3, "mcc": mcc3, "kappa": kap3})

    if metrics_rows:
        df_modes = pd.DataFrame(metrics_rows)
        df_modes.to_csv(out_dir / "metrics_mode_comparison.csv", index=False, encoding="utf-8")

        # barras
        fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
        x = np.arange(len(df_modes["modo"]))
        width = 0.2
        ax.bar(x - 1.5*width, df_modes["accuracy"], width, label="Accuracy")
        ax.bar(x - 0.5*width, df_modes["macro_f1"], width, label="Macro-F1")
        ax.bar(x + 0.5*width, df_modes["mcc"], width, label="MCC")
        ax.bar(x + 1.5*width, df_modes["kappa"], width, label="Kappa")
        ax.set_xticks(x)
        ax.set_xticklabels(df_modes["modo"], rotation=0)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Valor")
        ax.set_title("Comparação de Métricas por Modo de Avaliação")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "metrics_mode_comparison.png"); plt.close(fig)

    # --------------------------
    # Fim
    # --------------------------
    print("\n📁 Arquivos exportados (pasta de relatórios):")
    print(f"  • {Path(args.out_dir).resolve()}/")
    print(f"  • {Path(args.out_json).resolve()} (todos os pares avaliados)")
    print("\n🏁 Avaliação finalizada com sucesso.")


if __name__ == "__main__":
    main()
