import time

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint, uniform
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.evaluation.metrics import compute_all_metrics


def get_search_spaces(mono_constraints: dict) -> dict:
    """Return model templates and param distributions for the top 3 model types."""
    return {
        "XGBoost": {
            "model": XGBClassifier(
                monotone_constraints=mono_constraints,
                n_jobs=1, eval_metric="auc", random_state=42,
            ),
            "params": {
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
        },
        "LightGBM": {
            "model": LGBMClassifier(n_jobs=1, verbose=-1, random_state=42),
            "params": {
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
        },
        "CatBoost": {
            "model": CatBoostClassifier(
                eval_metric="AUC", random_seed=42, verbose=0, thread_count=1,
            ),
            "params": {
                "iterations": randint(200, 600),
                "depth": randint(3, 8),
                "learning_rate": uniform(0.01, 0.15),
                "l2_leaf_reg": uniform(1, 9),
                "bagging_temperature": uniform(0, 1),
                "random_strength": uniform(0, 1),
            },
        },
    }


def tune_models(
    top_names: list[str],
    X_tune: pd.DataFrame,
    y_tune: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    sample_weights: np.ndarray,
    mono_constraints: dict,
    n_iter: int = 30,
    cv_folds: int = 3,
    random_state: int = 42,
) -> dict:
    """Tune each named model with RandomizedSearchCV. Returns results dict."""
    spaces = get_search_spaces(mono_constraints)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = {}

    for name in top_names:
        if name not in spaces:
            print(f"Skipping {name} (no search space)")
            continue

        config = spaces[name]
        print(f"\nTuning {name}...")
        start = time.time()

        search = RandomizedSearchCV(
            estimator=config["model"],
            param_distributions=config["params"],
            n_iter=n_iter,
            scoring="average_precision",
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_tune, y_tune, sample_weight=sample_weights)
        elapsed = round(time.time() - start, 1)

        holdout_preds = search.best_estimator_.predict_proba(X_holdout)[:, 1]
        metrics = compute_all_metrics(y_holdout, holdout_preds)

        results[name] = {
            "best_params": search.best_params_,
            "cv_score": search.best_score_,
            "holdout_metrics": metrics,
            "estimator": search.best_estimator_,
        }

        print(f"  CV PR-AUC:      {search.best_score_:.4f}")
        print(f"  Holdout PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"  Holdout F2:     {metrics['f2']:.4f} (@{metrics['f2_threshold']:.2f})")
        print(f"  Time: {elapsed}s")

    return results
