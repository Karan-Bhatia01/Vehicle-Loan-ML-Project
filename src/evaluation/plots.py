import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from pathlib import Path
import pandas as pd


def plot_reliability_diagram(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    title: str,
    save_path: Path,
    marker: str = "o",
) -> None:
    """Plot reliability (calibration) diagram for one or more models."""
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for name, preds in predictions.items():
        prob_true, prob_pred = calibration_curve(y_true, preds, n_bins=10, strategy="quantile")
        plt.plot(prob_pred, prob_true, marker=marker, label=name)

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Actual Fraction of Defaults")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(
    model,
    feature_names: list[str],
    title: str,
    save_path: Path,
    top_n: int = 15,
) -> None:
    """Plot horizontal bar chart of top feature importances."""
    if not hasattr(model, "feature_importances_"):
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    top.plot(kind="barh", color="steelblue")
    plt.xlabel("Importance")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
