import numpy as np
import sys
sys.path.append('../thermodynamic-interpolation')
from gedmd.rff_tools import sample_rff_gaussian
from gedmd.rff import spectral_analysis_rff_generator
from tqdm import tqdm

np.random.seed(0)

def gedmd(X, Omega, n_env, beta, cut_svd):
    dj, Wj, M = spectral_analysis_rff_generator(
        X, Omega, n_env, a=1 / beta, tol=cut_svd, reversible=True
    )
    return dj, Wj, M

def bootstrap_eigenvalues(samples, Omega, n_env, beta, cut_svd, n_bootstrap=1000):
    eigenvalues = np.zeros((n_bootstrap, n_env))
    len_samples = samples.shape[1]

    for i in tqdm(range(n_bootstrap)):
        resample_indices = np.random.choice(len_samples, len_samples, replace=True)
        resample = samples[:, resample_indices]
        dj, _, _ = gedmd(resample, Omega, n_env, beta, cut_svd)
        eigenvalues[i] = -dj

    eigenvalues_mean = np.mean(eigenvalues, axis=0)
    lower_bound = np.percentile(eigenvalues, 2.5, axis=0)
    upper_bound = np.percentile(eigenvalues, 97.5, axis=0)

    return eigenvalues_mean, lower_bound, upper_bound

if __name__ == "__main__":
    Ts = [300, 400, 500, 600, 700, 800, 900]
    k_B_kJ_per_mol_K = 0.008314462618

    # run model selection for below parameters
    p = 300
    sig_opt = 5.0
    n_env = 4
    cut_svd = 1e-4
    Omega = sample_rff_gaussian(6, p, sig_opt) # dimension of rff should align with the number of torsions

    eigenvalues_mean = []
    eigenvalues_lower_bound = []
    eigenvalues_upper_bound = []

    for T in tqdm(Ts):
        beta = 1 / (k_B_kJ_per_mol_K * T)
        ti_samples = np.load(
            f"torsions_{T}k.npy"
        )
        X = ti_samples.T
        eigenvalues_mean, eigenvalues_lower, eigenvalues_upper = bootstrap_eigenvalues(X, Omega, n_env, beta, cut_svd)
        eigenvalues_mean.append(eigenvalues_mean)
        eigenvalues_lower_bound.append(eigenvalues_lower)
        eigenvalues_upper_bound.append(eigenvalues_upper)
