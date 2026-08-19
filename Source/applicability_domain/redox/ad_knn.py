# kNN-based applicability domain with adaptive per-point thresholds.
#
# A test point is within the AD if its distance to at least one training point
# does not exceed that point's threshold t[i].
#
# Threshold computation (Sahigara et al., Molecules 2012, 17, 4791):
#   1. Build a KDTree on the training set.
#   2. For each training point compute mean distance to k nearest neighbours.
#   3. Set r_ref = Q75 + 1.5 * IQR of those mean distances.
#   4. t[i] = sum(dist to k NN of i) / count(training points within r_ref of i)
#
# vect_in_ad uses KDTree instead of the original O(n) norm loop for speed,
# preserving identical semantics.

import logging
import os

import numpy as np
from sklearn.neighbors import KDTree

logger = logging.getLogger(__name__)


class KnnAD:
    def __init__(self, t_values_path: str, x_train: np.ndarray,
                 n_neighb: int = 5, leaf_size: int = 10):
        self.t_values_path = t_values_path
        self.x_train = np.asarray(x_train, dtype=np.float32)
        self.n_neighb = n_neighb
        self.leaf_size = leaf_size
        self._tree = KDTree(self.x_train, leaf_size=self.leaf_size)

        if os.path.exists(self.t_values_path):
            self.t_values = np.load(self.t_values_path)
            logger.info("Loaded t-values from %s", self.t_values_path)
        else:
            self.calc_t_values()

    def calc_mean_dists(self) -> tuple[KDTree, np.ndarray]:
        dist, _ = self._tree.query(self.x_train, k=self.n_neighb + 1)
        mean = np.mean(dist[:, 1:], axis=1)
        return self._tree, mean

    def calc_t_values(self) -> None:
        _, mean_dists = self.calc_mean_dists()
        q75, q25 = np.percentile(mean_dists, 75), np.percentile(mean_dists, 25)
        iqr = q75 - q25
        scale = (q75 + 1.5 * iqr) / np.median(mean_dists) if np.median(mean_dists) > 0 else 1.5
        self.t_values = mean_dists * scale

        self.t_values = np.asarray(self.t_values, dtype=np.float64)
        np.save(self.t_values_path, self.t_values)

        dist_k, idx_k = self._tree.query(self.x_train, k=2)
        nn_dist = dist_k[:, 1]
        t_nn    = self.t_values[idx_k[:, 1]]
        self_inlier = (nn_dist <= t_nn).mean()
        logger.info(
            "t-values: min=%.4f  median=%.4f  max=%.4f  scale=%.3f  "
            "train self-inlier rate=%.1f%%",
            self.t_values.min(), float(np.median(self.t_values)),
            self.t_values.max(), scale, 100 * self_inlier,
        )
        logger.info("t-values saved to %s", self.t_values_path)

    def vect_in_ad(self, input_vector: np.ndarray) -> bool:
        input_vector = np.asarray(input_vector, dtype=np.float32).reshape(1, -1)
        dist_nn, idx_nn = self._tree.query(input_vector, k=1)
        return bool(dist_nn[0, 0] <= self.t_values[idx_nn[0, 0]])

    def get_dataset_ad(self, x_test: np.ndarray) -> np.ndarray:
        """Return boolean array: True = within AD for each test vector."""
        return np.array([self.vect_in_ad(v) for v in x_test])