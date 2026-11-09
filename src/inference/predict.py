import pickle
from pathlib import Path

import pandas as pd

from src.data.preprocess import build_features


def load_pipeline(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_bundle(model_name: str, params: dict, encoder, calibrator, feature_columns: list[str],
                 threshold: float, metrics: dict, features_cfg: dict) -> dict:
    """Everything `predict` needs, in the format saved to models/loan_default_pipeline.pkl."""
    return {
        "model_name": model_name,
        "best_params": params,
        "target_encoder": encoder,
        "calibrated_model": calibrator,
        "cat_cols": features_cfg["cat_cols"],
        "skewed_cols": features_cfg["skewed_cols"],
        "feature_columns": feature_columns,
        "optimal_f2_threshold": threshold,
        "metrics": metrics,
    }


def save_pipeline(pipeline: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)


def _patch_legacy_encoder(encoder) -> None:
    """Add attributes that newer category_encoders expect but older pickled encoders lack."""
    defaults = {"min_group_lumping_": {}, "keep_components": False, "composite_members_": []}
    for attr, value in defaults.items():
        if not hasattr(encoder, attr):
            setattr(encoder, attr, value)

    ordinal = getattr(encoder, "ordinal_encoder", None)
    if ordinal is not None:
        for attr, value in {"min_group_lumping_": {}, "index_start": 1}.items():
            if not hasattr(ordinal, attr):
                setattr(ordinal, attr, value)


def predict(pipeline: dict, raw_input: pd.DataFrame) -> dict:
    """Score raw loan applications. Returns probabilities, APPROVE/REJECT decisions and the threshold."""
    X = build_features(raw_input, pipeline["cat_cols"], pipeline["skewed_cols"])
    X = X.reindex(columns=pipeline["feature_columns"], fill_value=0)

    encoder = pipeline["target_encoder"]
    _patch_legacy_encoder(encoder)
    X = encoder.transform(X).fillna(0)

    proba = pipeline["calibrated_model"].predict_proba(X)[:, 1]
    threshold = float(pipeline["optimal_f2_threshold"])

    return {
        "probability": [float(p) for p in proba],
        "decision": ["REJECT" if p > threshold else "APPROVE" for p in proba],
        "threshold": threshold,
    }
