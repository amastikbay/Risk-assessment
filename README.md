# Credit Risk Assessment — LightGBM with Interpretability & Risk Profiling

Predicts the probability of serious credit delinquency within two years using gradient boosting, SHAP-based interpretability, and actionable risk segmentation. This project deliberately goes beyond classical logistic regression: LightGBM captures non-linear interactions in credit behaviour that linear models miss, lifting AUC from ~0.79 (logistic baseline) to **0.85** without hand-crafted transformations.

---

## Portfolio context

| Project | Approach | Emphasis |
|---|---|---|
| `credit_scoring_analysis` | Logistic regression, scorecards | Statistical transparency, regulatory-friendly |
| **this repo** | LightGBM + SHAP + risk deciles | Predictive performance, modern ML, explainability |

The two repos complement each other: the first demonstrates classical credit-scoring methodology; this one shows how a quant/ML team would push further using gradient boosting while retaining explainability through SHAP.

---

## Dataset — Give Me Some Credit (Kaggle)

150,000 labelled training observations, 101,503 unlabelled test observations.

| Feature | Description |
|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Total revolving balance / credit limit |
| `age` | Borrower age in years |
| `NumberOfTime30-59DaysPastDueNotWorse` | Count of 30–59 day late payments |
| `DebtRatio` | Monthly debt payments / gross income |
| `MonthlyIncome` | Gross monthly income (USD) |
| `NumberOfOpenCreditLinesAndLoans` | Open credit lines + installment loans |
| `NumberOfTimes90DaysLate` | Count of 90+ day late payments |
| `NumberRealEstateLoansOrLines` | Mortgage and real estate loan count |
| `NumberOfTime60-89DaysPastDueNotWorse` | Count of 60–89 day late payments |
| `NumberOfDependents` | Dependents in family unit |

**Target** — `SeriousDlqin2yrs`: 1 if the borrower was 90+ days past due on any debt within two years, 0 otherwise. Class imbalance: ~6.7 % positive rate.

---

## Project structure

```
├── main.py                    # Full pipeline — run this
├── src/
│   ├── preprocessing.py       # Loading, cleaning, feature engineering, splitting
│   ├── modeling.py            # LR baseline, Random Forest, LightGBM tuning
│   ├── evaluation.py          # Metrics, ROC/PR plots, SHAP summary
│   └── risk_profiling.py      # Decile analysis, lift chart, risk bands
├── plots/                     # All output charts (auto-created)
├── models/                    # Saved model artifacts (auto-created)
├── requirements.txt
└── submission.csv             # Kaggle submission (auto-created)
```

---

## Pipeline

### 1. Data preprocessing

- Drop index artefact column and rows with missing `MonthlyIncome` / `NumberOfDependents`.
- Winsorise `RevolvingUtilizationOfUnsecuredLines` to [0, 1] and clip `DebtRatio` / `MonthlyIncome` at the 99th percentile to suppress extreme outliers.

### 2. Feature engineering

Four derived features improve model signal:

| Feature | Formula | Intuition |
|---|---|---|
| `TotalDaysLate` | sum of all past-due counts | Aggregate delinquency severity |
| `IncomePerDependent` | income / (dependents + 1) | Financial headroom per person |
| `UtilizationToDebt` | utilization / (debt ratio + ε) | Leverage relative to obligations |
| `DebtToIncome` | debt ratio × income | Absolute debt load in dollars |

### 3. Train / validation / test split (70 / 15 / 15)

Stratified split preserves the ~6.7 % default rate across all three sets.

- **Train (70 %)** — model fitting and cross-validation.
- **Validation (15 %)** — model selection during hyperparameter search.
- **Test (15 %)** — single held-out evaluation reported as final performance.

### 4. Hyperparameter tuning

`RandomizedSearchCV` with 3-fold CV over 20 random draws from:

```python
n_estimators      ∈ {200, 300, 500, 700}
learning_rate     ∈ {0.01, 0.05, 0.1, 0.15}
max_depth         ∈ {4, 6, 8, −1}
num_leaves        ∈ {31, 63, 127}
min_child_samples ∈ {20, 50, 100}
subsample         ∈ {0.6, 0.8, 1.0}
colsample_bytree  ∈ {0.6, 0.8, 1.0}
```

`class_weight='balanced'` is applied to all models to handle the 6.7 % minority class without oversampling.

### 5. Model comparison

| Model | Validation AUC | Validation PR-AUC | KS Statistic |
|---|---|---|---|
| Logistic Regression | ~0.79 | ~0.30 | ~0.41 |
| Random Forest | ~0.83 | ~0.38 | ~0.50 |
| **LightGBM (tuned)** | **~0.85** | **~0.43** | **~0.55** |

LightGBM's ability to model interaction effects (e.g., high utilization AND repeated late payments) explains the lift over the linear baseline.

### 6. Interpretability

**Feature importance** — LightGBM's split-count importances show which features appear most across the ensemble. `RevolvingUtilizationOfUnsecuredLines` and `TotalDaysLate` consistently rank highest.

**SHAP (SHapley Additive exPlanations)** — each prediction is decomposed into individual feature contributions, enabling:

- *Global*: which features drive risk across all borrowers (summary plot).
- *Local*: why a specific borrower received a high risk score — actionable for adverse action notices.

SHAP satisfies regulatory explainability requirements (FCRA, ECOA) that raw model outputs cannot meet.

### 7. Risk segmentation

**Risk bands** (business rule overlay):

| Band | Score threshold | Expected use |
|---|---|---|
| Low | < 5 % | Auto-approve; standard credit limits |
| Medium | 5 – 10 % | Manual review; reduced limits |
| High | > 10 % | Decline or secured product only |

**Decile analysis** — borrowers ranked by predicted score and split into 10 equal groups. The top decile captures a disproportionate share of actual defaults (lift > 4×). A risk team can set acceptance cut-offs directly from the lift table without needing to understand model internals.

---

## Key results

```
LightGBM test AUC      : 0.8499
LightGBM test PR-AUC   : ~0.43
LightGBM test KS       : ~0.55
Top decile lift        : ~4× baseline default rate
Logistic baseline AUC  : ~0.79  →  +6 pp uplift from gradient boosting
```

---

## How a risk team would use this model

**Origination decisioning** — score every applicant at point-of-sale; set hard cut-offs by risk band or approval rate target derived from the lift table.

**Limit management** — borrowers who migrate from Low to Medium band at account review trigger a proactive limit reduction or early contact.

**Early warning system** — monthly model refresh on the existing portfolio; accounts crossing into High band enter a collections workflow before delinquency occurs.

**Adverse action notices** — SHAP local explanations are translated into human-readable reasons ("high revolving utilization", "multiple recent late payments") as required by FCRA / ECOA.

---

## Production considerations

**Feature drift monitoring** — track PSI (Population Stability Index) on top features monthly. PSI > 0.25 signals distributional shift and triggers a model review.

**Periodic retraining** — credit data is cyclical (recessions, rate cycles). Recommended cadence: quarterly challenger training on recent data with champion-vs-challenger A/B testing before promotion.

**Governance** — model documentation should include: training data lineage, performance benchmarks on protected classes (ECOA fair lending analysis), cut-off justification, and formal MRM sign-off.

**Regulatory explainability** — SHAP values should be logged per prediction for audit trails. Ensure the top-N adverse factors are stored alongside each decision.

**Serving** — the serialised `.pkl` artefact wraps cleanly in a FastAPI/Flask REST endpoint. Feature preprocessing must be co-located with the model in the serving layer to prevent train/serve skew.

---

## Quick start

```bash
pip install -r requirements.txt

# Edit TRAIN_PATH / TEST_PATH in main.py to point at your local CSVs
python main.py
```

**Output:**

- `plots/roc_curves.png` — ROC comparison across three models
- `plots/pr_curves.png` — Precision-Recall comparison
- `plots/feature_importance.png` — top-15 LightGBM feature importances
- `plots/shap_summary.png` — SHAP beeswarm plot
- `plots/lift_chart.png` — decile lift chart
- `plots/default_rate_by_decile.png` — default rate per decile
- `models/lgbm_credit_risk.pkl` — serialised LightGBM model
- `submission.csv` — Kaggle-format predictions

---

## Loading the saved model

```python
import joblib

model = joblib.load("models/lgbm_credit_risk.pkl")

# X must contain the same engineered features as training
proba = model.predict_proba(X)[:, 1]
```
