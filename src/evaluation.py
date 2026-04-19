import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)

FIGSIZE = (8, 6)
DPI = 150


def evaluate_model(name: str, model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, proba)
    ks = float((tpr - fpr).max())
    return {
        "model":        name,
        "roc_auc":      round(roc_auc_score(y, proba), 4),
        "pr_auc":       round(average_precision_score(y, proba), 4),
        "ks_statistic": round(ks, 4),
        "brier_score":  round(brier_score_loss(y, proba), 4),
    }


def plot_roc_curves(models: dict, X, y, output_dir: str = "plots") -> None:
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=FIGSIZE)
    for name, model in models.items():
        proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, proba)
        auc = roc_auc_score(y, proba)
        plt.plot(fpr, tpr, label=f"{name}  AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=DPI)
    plt.close()


def plot_pr_curves(models: dict, X, y, output_dir: str = "plots") -> None:
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=FIGSIZE)
    for name, model in models.items():
        proba = model.predict_proba(X)[:, 1]
        prec, rec, _ = precision_recall_curve(y, proba)
        pr_auc = average_precision_score(y, proba)
        plt.plot(rec, prec, label=f"{name}  PR-AUC={pr_auc:.3f}")
    baseline = float(y.mean())
    plt.axhline(baseline, color="k", linestyle="--", label=f"No-skill ({baseline:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — Model Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curves.png"), dpi=DPI)
    plt.close()


def plot_feature_importance(model, feature_names: list, output_dir: str = "plots", top_n: int = 15) -> None:
    os.makedirs(output_dir, exist_ok=True)
    importances = pd.Series(model.feature_importances_, index=feature_names).nlargest(top_n)
    plt.figure(figsize=(8, 5))
    importances.sort_values().plot(kind="barh", color="steelblue")
    plt.title(f"Top {top_n} Feature Importances (LightGBM)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=DPI)
    plt.close()


def plot_shap_summary(model, X_sample, output_dir: str = "plots") -> None:
    os.makedirs(output_dir, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Newer LGBM versions return a single array; older ones return [neg, pos]
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure()
    shap.summary_plot(sv, X_sample, show=False, plot_size=(10, 6))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
