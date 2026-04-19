"""
Credit Risk Assessment Pipeline
================================
Entry point for the full ML pipeline:
  data loading → cleaning → feature engineering → train/val/test split →
  model training (LR, RF, LightGBM with hyperparameter tuning) →
  evaluation (AUC, PR-AUC, KS) → interpretability (SHAP) →
  risk profiling (deciles, lift) → model persistence → submission export
"""

import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

from src.preprocessing import load_data, clean_data, engineer_features, split_data, TARGET
from src.modeling import train_logistic_baseline, train_random_forest, train_lgbm_tuned, save_model
from src.evaluation import (
    evaluate_model,
    plot_roc_curves,
    plot_pr_curves,
    plot_feature_importance,
    plot_shap_summary,
)
from src.risk_profiling import (
    build_decile_table,
    plot_lift_chart,
    plot_default_rate_by_decile,
    risk_band_analysis,
)

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_PATH   = "/Users/amastikbayev/Downloads/cs-training.csv"
TEST_PATH    = "/Users/amastikbayev/Downloads/cs-test.csv"
PLOTS_DIR    = "plots"
MODELS_DIR   = "models"
RANDOM_STATE = 42
LGBM_TUNE_ITER = 20   # number of RandomizedSearchCV iterations


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── 1. Load & clean ───────────────────────────────────────────────────────
    _section("1. Data loading & preprocessing")
    df_train_raw, df_test_raw = load_data(TRAIN_PATH, TEST_PATH)
    print(f"Raw train : {df_train_raw.shape}   Raw test : {df_test_raw.shape}")
    print(f"Class balance (train) : {df_train_raw[TARGET].value_counts(normalize=True).to_dict()}")

    df = clean_data(df_train_raw)
    df = engineer_features(df)
    print(f"After cleaning + feature engineering : {df.shape}")

    # ── 2. Train / Validation / Test split ────────────────────────────────────
    _section("2. Stratified 70 / 15 / 15 split")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, val_size=0.15, test_size=0.15, random_state=RANDOM_STATE
    )
    print(f"Train : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")
    for label, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        print(f"  Default rate ({label}) : {y.mean():.4f}")

    # ── 3. Model training ─────────────────────────────────────────────────────
    _section("3. Model training")

    print("► Logistic Regression (baseline) …")
    lr = train_logistic_baseline(X_train, y_train)

    print("► Random Forest …")
    rf = train_random_forest(X_train, y_train)

    print(f"► LightGBM — RandomizedSearchCV (n_iter={LGBM_TUNE_ITER}) …")
    lgbm = train_lgbm_tuned(X_train, y_train, n_iter=LGBM_TUNE_ITER)

    models = {"Logistic Regression": lr, "Random Forest": rf, "LightGBM": lgbm}

    # ── 4. Validation evaluation ──────────────────────────────────────────────
    _section("4. Validation evaluation")
    val_results = []
    for name, model in models.items():
        m = evaluate_model(name, model, X_val, y_val)
        val_results.append(m)
        print(f"  {name:<25}  AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  KS={m['ks_statistic']:.4f}  Brier={m['brier_score']:.4f}")

    # ── 5. Evaluation plots ───────────────────────────────────────────────────
    _section("5. ROC & PR curves → plots/")
    plot_roc_curves(models, X_val, y_val, output_dir=PLOTS_DIR)
    plot_pr_curves(models, X_val, y_val, output_dir=PLOTS_DIR)
    print("  Saved roc_curves.png, pr_curves.png")

    # ── 6. Held-out test evaluation ───────────────────────────────────────────
    _section("6. Final held-out test evaluation (LightGBM)")
    test_metrics = evaluate_model("LightGBM", lgbm, X_test, y_test)
    print(f"  AUC={test_metrics['roc_auc']:.4f}  PR-AUC={test_metrics['pr_auc']:.4f}  "
          f"KS={test_metrics['ks_statistic']:.4f}  Brier={test_metrics['brier_score']:.4f}")

    # ── 7. Interpretability ───────────────────────────────────────────────────
    _section("7. Interpretability (feature importance + SHAP)")
    plot_feature_importance(lgbm, X_train.columns.tolist(), output_dir=PLOTS_DIR)
    print("  Saved feature_importance.png")

    X_shap = X_val.sample(min(2000, len(X_val)), random_state=RANDOM_STATE)
    plot_shap_summary(lgbm, X_shap, output_dir=PLOTS_DIR)
    print("  Saved shap_summary.png")

    # ── 8. Risk profiling ─────────────────────────────────────────────────────
    _section("8. Risk profiling — deciles & risk bands")
    lgbm_test_proba = lgbm.predict_proba(X_test)[:, 1]

    decile_table = build_decile_table(y_test, lgbm_test_proba)
    plot_lift_chart(decile_table, output_dir=PLOTS_DIR)
    plot_default_rate_by_decile(decile_table, output_dir=PLOTS_DIR)
    print("  Saved lift_chart.png, default_rate_by_decile.png")

    print("\nDecile analysis (D0 = highest predicted risk):")
    print(decile_table.to_string(index=False))

    band_summary = risk_band_analysis(y_test, lgbm_test_proba)
    print("\nRisk band summary:")
    print(band_summary.to_string(index=False))

    # ── 9. Save model ─────────────────────────────────────────────────────────
    _section("9. Persist LightGBM model")
    model_path = os.path.join(MODELS_DIR, "lgbm_credit_risk.pkl")
    save_model(lgbm, model_path)
    print(f"  Model saved → {model_path}")

    # ── 10. Submission export ─────────────────────────────────────────────────
    _section("10. Submission export")
    df_test_clean = clean_data(df_test_raw.drop(columns=[TARGET], errors="ignore"))
    df_test_clean = engineer_features(df_test_clean)
    df_test_clean = df_test_clean[X_train.columns]          # align features

    submission_proba = lgbm.predict_proba(df_test_clean)[:, 1]
    submission = pd.DataFrame({
        "Id": range(1, len(df_test_clean) + 1),
        "SeriousDlqin2yrs": submission_proba,
    })
    submission.to_csv("submission.csv", index=False)
    print("  Saved submission.csv")

    # ── Summary ───────────────────────────────────────────────────────────────
    _section("Pipeline complete")
    print(f"  LightGBM test AUC  : {test_metrics['roc_auc']:.4f}")
    print(f"  LightGBM test KS   : {test_metrics['ks_statistic']:.4f}")
    print(f"  Plots directory    : {PLOTS_DIR}/")
    print(f"  Saved model        : {model_path}")


if __name__ == "__main__":
    main()
