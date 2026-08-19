import os
import pickle

import numpy as np
from scipy import sparse
from sklearn import model_selection
from sklearn.neighbors import KernelDensity


class MahalanobisAD:
    def __init__(self, x_train, out_path):
        self.x_train = np.asarray(x_train)
        self.out_path = out_path
        self.estimate_density(scale=None)
        self.estimate_distance()

        if os.path.exists(self.out_path):
            with open(os.path.join(self.out_path, "density_model.pkl")) as f:
                self.density_model = pickle.load(f)
            self.density_mean = np.load(os.path.join(self.out_path, "density_mean.npy"))
            self.density_std = np.load(os.path.join(self.out_path, "density_std.npy"))
            self.distance_mean = np.load(os.path.join(self.out_path, "distance_mean.npy"))
            self.distance_std = np.load(os.path.join(self.out_path, "distance_std.npy"))

    def get_distance(self, input_vector):
        """
        :param input_vector: feature vector for prediction
        :return: Mahalanobis distance between feature vector and self.x_train
        """
        train_mean = np.mean(self.x_train, axis=0)
        train_shape = self.x_train.shape[0] - 1
        variance_covariance_matrix = (self.distance_matrix.T @ self.distance_matrix) / train_shape
        mahalanobis_distance = np.sqrt(
            (input_vector - train_mean) @ variance_covariance_matrix.T @ (input_vector - train_mean).T
        )
        return mahalanobis_distance

    def estimate_density(self, scale=None):
        """
        Build model for mutivariate kernel density estimation.
        Takes self.x_train tensor as input, builds a kernel density model
        with optimised parameters for input vectors,
        and computes density for all vectors in self.x_train.
        """

        if scale is None:
            bandwidth = np.logspace(-1, 2, 20)
        else:
            bandwidth = np.linspace(0.1, 0.5, 5)

        # find best parameters for KD
        grid = model_selection.GridSearchCV(KernelDensity(), {'bandwidth': bandwidth}, cv=3)
        grid.fit(self.x_train)

        # compute KD for self.x_train
        self.density_model = KernelDensity(**grid.best_params_).fit(self.x_train)
        samples = self.density_model.score_samples(self.x_train)
        self.density_mean = np.mean(samples)
        self.density_std = np.std(samples)

        with open(os.path.join(self.out_path, "density_model.pkl"), "w") as f:
            pickle.dump(self.density_model, f)
        np.save(os.path.join(self.out_path, "density_mean.npy"), self.density_mean)
        np.save(os.path.join(self.out_path, "density_std.npy"), self.density_std)

    def estimate_distance(self):
        """
        Takes self.x_train dataframe,
        calculates Mahalonobis distance between whole self.x_train
        and each separate vector from self.x_train
        """

        centroid = self.x_train - np.tile(np.mean(self.x_train, axis=0), (self.x_train.shape[0], 1))
        self.distance_matrix = sparse.csr_matrix(np.asarray(centroid).astype(dtype='float32'))
        dist_list = np.apply_along_axis(self.get_distance, 1, self.x_train)

        self.distance_mean = np.mean(dist_list)
        self.distance_std = np.std(dist_list)

        np.save(os.path.join(self.out_path, "distance_mean.npy"), self.distance_mean)
        np.save(os.path.join(self.out_path, "distance_std.npy"), self.distance_std)

    def vect_in_ad(self, input_vector):
        """
        Get AD prediction for single vector

        Parameters
        ----------
        input_vector: array-like
            vector to get an AD prediction

        Returns
        -------
        vect_in_ad_by_distance: bool
            True for x-inlier, False for x-outlier
        vect_in_ad_by_dencity: bool
            True for x-inlier, False for x-outlier
        """
        dist = self.get_distance(np.asarray(input_vector).reshape(np.mean(self.x_train, axis=0).shape))
        dens = abs(self.density_model.score_samples(input_vector.reshape(1, -1)))

        distance = True if dist <= self.distance_mean + 3 * self.distance_std else False
        density = True if dens >= self.density_mean - 3 * self.density_std else False

        return distance, density

    def get_dataset_ad(self, x_test):
        """
        Get AD for whole test dataset

        Parameters
        ----------
        x_test: array-like
            Matrix with vectors for prediction

        Returns
        -------
        ad_by_distances: np.array
            bool array - True for x-outliers and False for x-inliers
        ad_by_densities: np.array
            bool array - True for x-outliers and False for x-inliers

        """
        ad_by_distances = []
        ad_by_densities = []
        for x_predict in x_test:
            distance_bool, density_bool = self.vect_in_ad(x_predict)
            ad_by_distances += [distance_bool]
            ad_by_densities += [density_bool]
        return np.array(ad_by_distances), np.array(ad_by_densities)
