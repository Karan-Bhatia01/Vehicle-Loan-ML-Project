import category_encoders as ce
import pandas as pd


class TargetEncoderWrapper:
    """Smoothed target encoder that fills unseen/NaN encodings with 0."""

    def __init__(self, cols: list[str], smoothing: int = 10, min_samples_leaf: int = 20):
        self.encoder = ce.TargetEncoder(cols=cols, smoothing=smoothing, min_samples_leaf=min_samples_leaf)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.encoder.fit_transform(X, y).fillna(0)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.transform(X).fillna(0)
