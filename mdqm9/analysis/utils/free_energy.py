import os
import sys
sys.path.append('../thermodynamic-interpolants')

import numpy as np
from mdqm9.analysis.utils import sensititvity


def calc_phis_tfep(E0s: np.array, E1s: np.array, neg_dlogps_ti: np.array, k=None):
    phis = E1s - E0s + neg_dlogps_ti

    if k is not None:
        exp_phis = np.exp(-phis)
        indexes_to_keep = sensititvity.filter_iqr(exp_phis, k=k)
        exp_phis = exp_phis[indexes_to_keep]
        phis = -np.log(exp_phis)
        return phis, indexes_to_keep
    return phis, np.ones_like(phis, dtype=bool)


def calc_phis_bg(Es: np.array, neg_dlogps_bg: np.array, k=None):
    phis = Es + neg_dlogps_bg
    
    if k is not None:
        indexes_to_keep = sensititvity.filter_iqr(phis, k=k)
        phis = phis[indexes_to_keep]
    return phis


def calc_phis_bg_tfep(E0s: np.array, neg_dlogps_bg_T0: np.array, E1s: np.array, neg_dlogps_bg_T1: np.array, k=None):
    phis = E1s + neg_dlogps_bg_T1 - E0s - neg_dlogps_bg_T0
    
    if k is not None:
        exp_phis = np.exp(-phis)
        indexes_to_keep = sensititvity.filter_iqr(exp_phis, k=k)
        exp_phis = exp_phis[indexes_to_keep]
        phis = -np.log(exp_phis)
    return phis


def calc_tfep_dF(phis: np.array, weights: np.array) -> float:
    weighted_obs = np.exp(-phis)*weights
    mean = weighted_obs.sum()
    normalized_mean = mean / weights.sum()

    return -np.log(normalized_mean)


def calc_bg_dF(phis):
    return phis.mean()


