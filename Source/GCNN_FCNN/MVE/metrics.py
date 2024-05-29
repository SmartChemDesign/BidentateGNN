import torch
from sklearn.metrics import r2_score, mean_absolute_error

from Source.data import root_mean_squared_error


def negative_log_likelihood(y_pred, y_true):
    y_mean, log_y_var = y_pred[:, 0].unsqueeze(-1), y_pred[:, 1].unsqueeze(-1)
    y_var = torch.exp(log_y_var)
    return 0.5 * torch.mean(log_y_var + (y_true - y_mean) ** 2 / y_var)


def r2_score_MVE(y_true, y_pred):
    return r2_score(y_true, y_pred[:, 0])


def root_mean_squared_error_MVE(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred[:, 0])


def mean_absolute_error_MVE(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred[:, 0])
