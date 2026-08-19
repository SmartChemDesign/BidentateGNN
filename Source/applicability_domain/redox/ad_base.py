from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np


class ApplicabilityDomain(ABC):
    def __init__(self, cache_path: Optional[Path] = None):
        self.cache_path = Path(cache_path) if cache_path else None

    @abstractmethod
    def fit(self, x_train: np.ndarray) -> None:
        """Fit the AD model on training latent vectors.

        Args:
            x_train: Training vectors, shape (n_train, d).
        """

    @abstractmethod
    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """Return boolean inlier array for test vectors.

        Args:
            x_test: Test vectors, shape (n_test, d).

        Returns:
            Boolean array of shape (n_test,); True = within AD.
        """

    @abstractmethod
    def distance(self, x_test: np.ndarray) -> np.ndarray:
        """Return a scalar distance metric for each test vector.

        Smaller values indicate the point is closer to the training distribution.

        Args:
            x_test: Test vectors, shape (n_test, d).

        Returns:
            Float array of shape (n_test,).
        """

    def summary(self, x_test: np.ndarray) -> dict:
        """Return a dict with inlier flags and distances for x_test."""
        inlier = self.predict(x_test)
        dist   = self.distance(x_test)
        n = len(inlier)
        return {
            "n_total":   n,
            "n_inlier":  int(inlier.sum()),
            "n_outlier": int((~inlier).sum()),
            "pct_inlier": float(inlier.mean() * 100),
            "dist_mean": float(dist.mean()),
            "dist_std":  float(dist.std()),
            "inlier":    inlier,
            "distances": dist,
        }