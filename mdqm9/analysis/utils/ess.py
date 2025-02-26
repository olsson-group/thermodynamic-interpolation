import os
import sys
sys.path.append('../thermointerpolants')

import pandas as pd
import numpy as np
import scipy.stats as stats


def calc_ti_weights(E0s: np.array, E1s: np.array, neg_dlogps_ti: np.array):
    phis = E1s - E0s + neg_dlogps_ti
    return np.exp(-phis)


def calc_importance_weights(z0s: np.array, E1s: np.array, neg_dlogps_bg: np.array, neg_dlogps_ti: np.array):   # obs: it's ok to set neg_dlogps_ti to zero
    # reshape x0s to (n_samples, n_atoms*3)
    z0s_reshaped = []
    n_samples, _, _ = z0s.shape
    for i in range(n_samples):
        z0s_reshaped.append(z0s[i].flatten())
    z0s = np.array(z0s_reshaped)

    log_pzs = calc_log_mvnormal_pzs(z0s)
    weights = np.exp(-E1s - log_pzs - (neg_dlogps_bg + neg_dlogps_ti))
    return weights


def calc_log_mvnormal_pzs(z0s: np.array):
    norm = stats.multivariate_normal(mean=np.zeros(z0s.shape[1]), cov=np.eye(z0s.shape[1]))
    log_pzs = norm.logpdf(z0s)
    return log_pzs


def calc_ESS(weights: np.array):
    squared_sum_w = np.square(np.sum(weights))
    sum_w_squared = np.sum(np.square(weights))
    return squared_sum_w / sum_w_squared

