import numpy as np
import sys
sys.path.append('../thermodynamic-interpolation')

from gedmd.rff import cv_generator_rff
from gedmd.rff_tools import sample_rff_gaussian

T = 600 # take 600k as an example

k_B_kJ_per_mol_K = 0.008314462618
beta = 1 / (k_B_kJ_per_mol_K * T)

md_samples = np.load(f"/molecular_torsions/md_torsions_{T}k.npy")
X = md_samples.T

# Kernel bandwidths
sigma_list = np.array([5.0, 7.0, 9.0, 10.0, 11.0, 12.0, 12.5, 13.0])
signum = sigma_list.shape[0]

# Feature sizes:
p_list = np.array([50, 100, 300, 500, 1000])
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
        n_torsions = X.shape[0]
        Omega = sample_rff_gaussian(n_torsions, p, sigma)
        # Compute eigenvalues and test scores for this model:
        d_ij, dtest_ij = cv_generator_rff(X, Omega, (1.0/beta), rtrain, ntest, nev, tol=cut_svd)
        d[ii, jj, :, :] = d_ij
        dtest[ii, jj, :] = -dtest_ij
print("Complete.")

# Save Results:
di = {}
di["EV"] = d
di["VAMP"] = dtest
np.savez("model_selection_results.npz", **di)