import pickle
from pathlib import Path

import category_encoders as ce
import pandas as pd


class TargetEncoderWrapper:
    """Thin wrapper around ce.TargetEncoder with save/load support."""

    def __init__(self, cols: list[str], smoothing: int = 10, min_samples_leaf: int = 20):
        self.cols = cols
        self.encoder = ce.TargetEncoder(
            cols=cols, smoothing=smoothing, min_samples_leaf=min_samples_leaf,
        )

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.encoder.fit_transform(X, y).fillna(0)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.transform(X).fillna(0)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.encoder, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            self.encoder = pickle.load(f)
