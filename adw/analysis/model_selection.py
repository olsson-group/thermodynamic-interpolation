import numpy as np
import sys
sys.path.append('../thermodynamic-interpolation')

from gedmd.rff import cv_generator_rff
from gedmd.rff_tools import sample_rff_gaussian
import pandas as pd

if __name__ == "__main__":
    md_samples = pd.read_csv("md_samples.csv")
    beta = 0.75
    col_name = f"{beta:.2f}"
    md_samples_t = md_samples[col_name].sample(10000, random_state=42)
    X = np.array(md_samples_t).reshape(1, -1)

    # Kernel bandwidths
    sigma_list = np.array([1e-2, 5e-2, 1e-1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 2.0])
    signum = sigma_list.shape[0]
    # Feature sizes:
    p_list = np.array([50, 100, 200, 300, 400, 500])
    pnum = p_list.shape[0]
    # tolerance for singular values of M:
    cut_svd = 1e-4
    # Number of eigenvalues:
    nev = 4
    # Number of tests:
    ntest = 20
    # Ration of training and test data:
    rtrain = .75

    d = np.zeros((signum, pnum, ntest, nev), dtype=complex)
    dtest = np.zeros((signum, pnum, ntest), dtype=complex)

    """ Score models for all values of sigma and p:"""
    for ii in range(signum):
        for jj in range(pnum):
            sigma = sigma_list[ii]
            p = p_list[jj]
            print("Scoring sigma=%.2f, p=%d..."%(sigma, p))
            # Generate Fourier features:
            Omega = sample_rff_gaussian(1, p, sigma)
            # Compute eigenvalues and test scores for this model:
            d_ij, dtest_ij = cv_generator_rff(X, Omega, (2.0/beta), rtrain, ntest, nev, tol=cut_svd)
            d[ii, jj, :, :] = d_ij
            dtest[ii, jj, :] = -dtest_ij
    print("Complete.")
    # Save Results:
    di = {}
    di["EV"] = d
    di["VAMP"] = dtest
    np.savez("model_selection_results.npz", **di)
