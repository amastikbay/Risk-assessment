import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 150


def assign_risk_band(prob: float) -> str:
    if prob < 0.05:
        return "Low"
    elif prob < 0.10:
        return "Medium"
    return "High"


def build_decile_table(y_true: pd.Series, y_proba: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true.values, "y_proba": y_proba})
    df["decile"] = pd.qcut(df["y_proba"].rank(method="first"), q=10, labels=False)
    df["decile"] = 9 - df["decile"]  # Decile 0 = highest predicted risk

    table = (
        df.groupby("decile", observed=True)
        .agg(count=("y_true", "count"), defaults=("y_true", "sum"), avg_score=("y_proba", "mean"))
        .reset_index()
    )
    overall_rate = y_true.mean()
    table["default_rate"] = table["defaults"] / table["count"]
    table["lift"] = table["default_rate"] / overall_rate
    table["cum_capture_rate"] = table["defaults"].cumsum() / table["defaults"].sum()
    table.insert(0, "decile", table.pop("decile"))
    return table


def plot_lift_chart(decile_table: pd.DataFrame, output_dir: str = "plots") -> None:
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(decile_table["decile"], decile_table["lift"], color="steelblue", edgecolor="white")
    plt.axhline(1.0, color="crimson", linestyle="--", linewidth=1.5, label="No-skill baseline")
    plt.xticks(decile_table["decile"], [f"D{d}" for d in decile_table["decile"]])
    plt.xlabel("Decile (0 = highest predicted risk)")
    plt.ylabel("Lift")
    plt.title("Lift Chart by Decile (Test Set)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lift_chart.png"), dpi=DPI)
    plt.close()


def plot_default_rate_by_decile(decile_table: pd.DataFrame, output_dir: str = "plots") -> None:
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(decile_table["decile"], decile_table["default_rate"] * 100, color="coral", edgecolor="white")
    plt.xticks(decile_table["decile"], [f"D{d}" for d in decile_table["decile"]])
    plt.xlabel("Decile (0 = highest predicted risk)")
    plt.ylabel("Default Rate (%)")
    plt.title("Default Rate by Risk Decile (Test Set)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "default_rate_by_decile.png"), dpi=DPI)
    plt.close()


def risk_band_analysis(y_true: pd.Series, y_proba: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true.values, "y_proba": y_proba})
    df["risk_band"] = df["y_proba"].apply(assign_risk_band)

    order = ["Low", "Medium", "High"]
    summary = (
        df.groupby("risk_band", observed=True)
        .agg(count=("y_true", "count"), defaults=("y_true", "sum"), avg_score=("y_proba", "mean"))
        .reindex(order)
        .reset_index()
    )
    summary["default_rate_pct"] = (summary["defaults"] / summary["count"] * 100).round(2)
    summary["pct_population"] = (summary["count"] / len(df) * 100).round(1)
    return summary
