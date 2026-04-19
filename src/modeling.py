import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier


def train_logistic_baseline(X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            C=0.1,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def train_lgbm_tuned(X_train, y_train, n_iter: int = 20) -> LGBMClassifier:
    """Train LightGBM with RandomizedSearchCV over the key hyperparameters."""
    param_dist = {
        "n_estimators":      [200, 300, 500, 700],
        "learning_rate":     [0.01, 0.05, 0.1, 0.15],
        "max_depth":         [4, 6, 8, -1],
        "num_leaves":        [31, 63, 127],
        "min_child_samples": [20, 50, 100],
        "subsample":         [0.6, 0.8, 1.0],
        "colsample_bytree":  [0.6, 0.8, 1.0],
        "reg_alpha":         [0.0, 0.1, 0.5],
        "reg_lambda":        [0.0, 0.1, 1.0],
    }

    base = LGBMClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    search = RandomizedSearchCV(
        base,
        param_dist,
        n_iter=n_iter,
        cv=3,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"  Best CV AUC : {search.best_score_:.4f}")
    print(f"  Best params : {search.best_params_}")
    return search.best_estimator_


def save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
