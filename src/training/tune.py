import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.evaluation.metrics import compute_all_metrics
from src.training.train import create_classifier

SEARCH_SPACES = {
    "XGBoost": {
        "n_estimators": randint(200, 600),
        "max_depth": randint(3, 8),
        "learning_rate": uniform(0.01, 0.15),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "min_child_weight": randint(1, 10),
        "reg_alpha": uniform(0, 1.0),
        "reg_lambda": uniform(0.5, 2.0),
        "gamma": uniform(0, 0.5),
    },
    "LightGBM": {
        "n_estimators": randint(200, 600),
        "max_depth": randint(3, 8),
        "learning_rate": uniform(0.01, 0.15),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "num_leaves": randint(20, 80),
        "min_child_samples": randint(10, 100),
        "reg_alpha": uniform(0, 1.0),
        "reg_lambda": uniform(0.5, 2.0),
    },
    "CatBoost": {
        "iterations": randint(200, 600),
        "depth": randint(3, 8),
        "learning_rate": uniform(0.01, 0.15),
        "l2_leaf_reg": uniform(1, 9),
        "bagging_temperature": uniform(0, 1),
        "random_strength": uniform(0, 1),
    },
}


def tune_models(
    names: list[str],
    X_tune: pd.DataFrame,
    y_tune: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    sample_weights: np.ndarray,
    mono_constraints: dict,
    tuning_cfg: dict,
    random_state: int = 42,
) -> dict:
    """RandomizedSearchCV per model, then score the best estimator on the holdout set."""
    cv = StratifiedKFold(n_splits=tuning_cfg["cv_folds"], shuffle=True, random_state=random_state)
    results = {}

    for name in names:
        print(f"\nTuning {name}...")
        # Each search candidate runs single-threaded; the search itself parallelises across candidates.
        search = RandomizedSearchCV(
            estimator=create_classifier(name, {}, mono_constraints, n_jobs=1),
            param_distributions=SEARCH_SPACES[name],
            n_iter=tuning_cfg["n_iter"],
            scoring=tuning_cfg["scoring"],
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
        )
        search.fit(X_tune, y_tune, sample_weight=sample_weights)

        metrics = compute_all_metrics(y_holdout, search.best_estimator_.predict_proba(X_holdout)[:, 1])
        results[name] = {"best_params": search.best_params_, "holdout_metrics": metrics}

        print(f"  CV PR-AUC:      {search.best_score_:.4f}")
        print(f"  Holdout PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"  Holdout F2:     {metrics['f2']:.4f} (@{metrics['f2_threshold']:.2f})")

    return results
