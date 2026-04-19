import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

TARGET = "SeriousDlqin2yrs"
MISSING_COLS = ["MonthlyIncome", "NumberOfDependents"]


def load_data(train_path: str, test_path: str | None = None):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path) if test_path else None
    return df_train, df_test


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])
    df = df.dropna(subset=MISSING_COLS)

    # Winsorize extreme outliers that would otherwise dominate tree splits
    df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(0, 1)
    df["DebtRatio"] = df["DebtRatio"].clip(0, df["DebtRatio"].quantile(0.99))
    df["MonthlyIncome"] = df["MonthlyIncome"].clip(0, df["MonthlyIncome"].quantile(0.99))
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    delinq_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    df["TotalDaysLate"] = df[delinq_cols].sum(axis=1)
    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    df["UtilizationToDebt"] = df["RevolvingUtilizationOfUnsecuredLines"] / (df["DebtRatio"] + 1e-6)
    df["DebtToIncome"] = df["DebtRatio"] * df["MonthlyIncome"]
    return df


def split_data(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
):
    """70/15/15 stratified split so that the test set is a true held-out evaluation."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
