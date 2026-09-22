"""Numeric preparation for neural models: median imputation + standardisation, fit on training rows."""

from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class TabularPrep:
    def __init__(self, clip: float = 10.0):
        self.imp = SimpleImputer(strategy="median")
        self.sc = StandardScaler()
        self.clip = clip

    def fit(self, X) -> "TabularPrep":
        X = np.asarray(X, dtype=np.float32)
        self.sc.fit(self.imp.fit_transform(X))
        return self

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        Z = self.sc.transform(self.imp.transform(X))
        Z = np.nan_to_num(Z, nan=0.0, posinf=self.clip, neginf=-self.clip)
        return np.clip(Z, -self.clip, self.clip).astype(np.float32)

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)
