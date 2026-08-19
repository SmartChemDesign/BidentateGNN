import logging
import os
import pickle

import numpy as np
from scipy import sparse
from sklearn import model_selection
from sklearn.neighbors import KernelDensity

logger = logging.getLogger(__name__)


class MahalanobisAD:
    def __init__(self, x_train: np.ndarray, out_path: str, n_components: int = 50):
        self.x_train = np.asarray(x_train)
        self.out_path = out_path
        self.n_components = n_components
        os.makedirs(out_path, exist_ok=True)

        density_path  = os.path.join(out_path, "density_model.pkl")
        dist_mean_path = os.path.join(out_path, "distance_mean.npy")

        if os.path.exists(density_path) and os.path.exists(dist_mean_path):
            with open(density_path, "rb") as f:
                self.density_model = pickle.load(f)
            self.density_mean  = np.load(os.path.join(out_path, "density_mean.npy"))
            self.density_std   = np.load(os.path.join(out_path, "density_std.npy"))
            self.distance_mean = float(np.load(dist_mean_path))
            self.distance_std  = float(np.load(os.path.join(out_path, "distance_std.npy")))
            # Load PCA if it was used
            pca_path = os.path.join(out_path, "pca.pkl")
            if os.path.exists(pca_path):
                with open(pca_path, "rb") as f:
                    self.pca = pickle.load(f)
                x_reduced = self.pca.transform(self.x_train)
            else:
                self.pca = None
                x_reduced = self.x_train
            self._x_reduced_train = x_reduced
            centroid = x_reduced - np.mean(x_reduced, axis=0)
            self.distance_matrix = sparse.csr_matrix(
                np.asarray(centroid).astype(np.float32)
            )
            logger.info("Loaded Mahalanobis AD from %s", out_path)
        else:
            self.estimate_density(scale=None)
            self.estimate_distance()

    def get_distance_reduced(self, input_vector_reduced: np.ndarray) -> float:
        x_ref = self._x_reduced_train if hasattr(self, "_x_reduced_train") else self.x_train
        train_mean  = np.mean(x_ref, axis=0)
        train_shape = x_ref.shape[0] - 1
        cov = (self.distance_matrix.T @ self.distance_matrix).toarray() / train_shape
        cov += np.eye(cov.shape[0]) * 1e-4
        cov_inv = np.linalg.inv(cov)
        delta = input_vector_reduced - train_mean
        val = delta @ cov_inv @ delta.T
        return float(np.sqrt(val)) if val >= 0 else float("nan")

    def get_distance(self, input_vector: np.ndarray) -> float:
        if hasattr(self, "pca") and self.pca is not None:
            input_vector_reduced = self.pca.transform(
                np.asarray(input_vector).reshape(1, -1)
            )[0]
        else:
            input_vector_reduced = np.asarray(input_vector)
        return self.get_distance_reduced(input_vector_reduced)

    def estimate_distance(self) -> None:
        from sklearn.decomposition import PCA

        n, d = self.x_train.shape
        if d > self.n_components:
            logger.info(
                "Applying PCA: %d -> %d dimensions before Mahalanobis computation.",
                d, self.n_components,
            )
            self.pca = PCA(n_components=self.n_components, random_state=42)
            x_reduced = self.pca.fit_transform(self.x_train)
        else:
            self.pca = None
            x_reduced = self.x_train

        centroid = x_reduced - np.mean(x_reduced, axis=0)
        self.distance_matrix = sparse.csr_matrix(
            np.asarray(centroid).astype(np.float32)
        )
        self._x_reduced_train = x_reduced  # keep for get_distance
        dist_list = np.apply_along_axis(self.get_distance_reduced, 1, x_reduced)
        self.distance_mean = float(np.nanmean(dist_list))
        self.distance_std  = float(np.nanstd(dist_list))
        np.save(os.path.join(self.out_path, "distance_mean.npy"), self.distance_mean)
        np.save(os.path.join(self.out_path, "distance_std.npy"),  self.distance_std)
        # Save PCA if used
        if self.pca is not None:
            import pickle
            with open(os.path.join(self.out_path, "pca.pkl"), "wb") as f:
                pickle.dump(self.pca, f)
        logger.info(
            "Distance stats: mean=%.4f std=%.4f", self.distance_mean, self.distance_std
        )

    def estimate_density(self, scale=None) -> None:
        if scale is None:
            bandwidth = np.logspace(-1, 2, 20)
        else:
            bandwidth = np.linspace(0.1, 0.5, 5)

        grid = model_selection.GridSearchCV(
            KernelDensity(), {"bandwidth": bandwidth}, cv=3
        )
        grid.fit(self.x_train)

        self.density_model = KernelDensity(**grid.best_params_).fit(self.x_train)
        samples = self.density_model.score_samples(self.x_train)
        self.density_mean = np.mean(samples)
        self.density_std  = np.std(samples)

        with open(os.path.join(self.out_path, "density_model.pkl"), "wb") as f:
            pickle.dump(self.density_model, f)
        np.save(os.path.join(self.out_path, "density_mean.npy"), self.density_mean)
        np.save(os.path.join(self.out_path, "density_std.npy"),  self.density_std)
        logger.info(
            "KDE bandwidth=%.4f  density mean=%.4f std=%.4f",
            grid.best_params_["bandwidth"], self.density_mean, self.density_std,
        )

    def vect_in_ad(self, input_vector: np.ndarray) -> tuple[bool, bool]:
        dist = self.get_distance(
            np.asarray(input_vector).reshape(np.mean(self.x_train, axis=0).shape)
        )
        dens = abs(
            self.density_model.score_samples(
                np.asarray(input_vector).reshape(1, -1)
            )[0]
        )
        in_ad_distance = dist <= self.distance_mean + 3 * self.distance_std
        in_ad_density  = dens >= self.density_mean - 3 * self.density_std
        return in_ad_distance, in_ad_density

    def get_dataset_ad(
        self, x_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ad_by_distances, ad_by_densities = [], []
        for x_predict in x_test:
            dist_bool, dens_bool = self.vect_in_ad(x_predict)
            ad_by_distances.append(dist_bool)
            ad_by_densities.append(dens_bool)
        return np.array(ad_by_distances), np.array(ad_by_densities)