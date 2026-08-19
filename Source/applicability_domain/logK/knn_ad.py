import warnings
import os

import numpy as np
from sklearn.neighbors import KDTree


class knnAD:
    def __init__(self, t_values_path, x_train, n_neighb=5, leaf_size=10):
        self.t_values_path = t_values_path
        self.x_train = x_train
        self.n_neighb = n_neighb
        self.leaf_size = leaf_size

        if os.path.exists(self.t_values_path):
            self.t_values = np.load(self.t_values_path)
        else:
            self.calc_t_values()

    def calc_t_values(self):
        """
        Calculate t-values for each vector in self.x_train
        """
        tree, mean = self.calc_mean_dists()
        r_ref = np.percentile(mean, 75) + 1.5 * (np.percentile(mean, 75) - np.percentile(mean, 25))
        dist_matr, ind = tree.query(self.x_train, self.x_train.shape[0])
        counts = []
        for line in dist_matr:
            condition = line[1:] < r_ref
            counts.append(len(np.extract(condition, line[1:])))

        nearest_dist, ind = tree.query(self.x_train, self.n_neighb)
        self.t_values = []
        for n, d in zip(counts, nearest_dist):
            if n > 0:
                self.t_values.append(sum(d) / n)
            else:
                self.t_values.append(np.nan)

        for i, value in enumerate(self.t_values):
            if value is None:
                self.t_values[i] = np.nanmin(self.t_values)

        np.save(self.t_values_path, np.array(self.t_values))

    def calc_mean_dists(self):
        tree = KDTree(self.x_train, self.leaf_size)
        dist, ind = tree.query(self.x_train, k=self.n_neighb + 1)
        mean = [np.mean(i) for i in dist[1:]]
        return tree, mean

    def vect_in_ad(self, input_vector):
        """
        Get AD prediction for single vector

        Parameters
        ----------
        input_vector: array-like
            vector to get an AD prediction

        Returns
        -------
        vect_in_ad: bool
            True for x-inlier, False for x-outlier

        """
        distances = [np.linalg.norm(input_vector - temp_vector) for temp_vector in self.x_train]

        for dist, t_val in zip(distances, self.t_values):
            if dist <= t_val:
                return True

        return False

    def get_dataset_ad(self, x_test):
        """
        Get AD for whole test dataset

        Parameters
        ----------
        x_test: array-like
            Matrix with vectors for prediction

        Returns
        -------
        ad: np.array
            bool array - True for x-outliers and False for x-inliers
        """
        return np.array([self.vect_in_ad(vect) for vect in x_test])
