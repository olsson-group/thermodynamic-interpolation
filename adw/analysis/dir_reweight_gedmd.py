import numpy as np
import sys
sys.path.append('../thermodynamic-interpolation')
from gedmd.rff_tools import sample_rff_gaussian
from gedmd.rff import spectral_analysis_rff_generator
import pandas as pd
from tqdm import tqdm

np.random.seed(0)

class AsymmetricDoubleWell:
    def __init__(self, a=4, b=0.5):
        self.a = a
        self.b = b

    def __call__(self, x):
        return self.a * (x**2 - 1) ** 2 + self.b * x

    def grad(self, x):
        return 4 * self.a * (x**3 - x) + self.b
    
def calculate_energy(sample, a=4, b=0.5):
    dw = AsymmetricDoubleWell(a=a, b=b)
    energy = dw(sample)
    return energy

def direct_weights(samples, initial_beta, target_beta):
    energies = calculate_energy(samples)
    log_weights = (initial_beta - target_beta) * energies
    return np.exp(log_weights)

def resample_with_weights(samples, weights, n_samples=None):
    if n_samples is None:
        n_samples = len(samples)
    normalized_weights = weights / np.sum(weights)
    resampled_indices = np.random.choice(len(samples), size=n_samples, replace=True, p=normalized_weights)
    resampled_samples = samples[resampled_indices]
    return resampled_samples

def gedmd(X, Omega, n_env, beta, cut_svd):
    dj, Wj, M = spectral_analysis_rff_generator(
        X, Omega, n_env, a=2/beta, tol=cut_svd, reversible=True
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
    # example from 0.5 to other betas, direct reweighting
    betas = [0.75, 1.0, 1.5, 1.75] # beta parameteres

    # run model selection to find below optimal parameters
    p = 50
    sig_opt = 0.6
    n_env = 4
    cut_svd = 1e-4

    Omega = sample_rff_gaussian(1, p, sig_opt)

    second_eigenvalues_mean = []
    second_eigenvalues_lowerbound = []
    second_eigenvalues_upperbound = []

    md_samples = pd.read_csv(
        "/adw_md_samples.csv"
    )
    md_samples = md_samples["0.50"].iloc[:25000] 
    md_samples = np.array(md_samples)

    for i, beta in enumerate(betas):
        beta = float(beta)
        print(f"Processing temperature {beta}")

        weights = direct_weights(md_samples, 0.50, beta)

        resampled_samples = resample_with_weights(md_samples, weights)

        eigenvalues_mean, eigenvalues_lowerbound, eigenvalues_upperbound = bootstrap_eigenvalues(
            resampled_samples.reshape(1, -1), Omega, n_env, beta, cut_svd
        )
        second_eigenvalues_mean.append(eigenvalues_mean[2])
        second_eigenvalues_lowerbound.append(eigenvalues_lowerbound[2])
        second_eigenvalues_upperbound.append(eigenvalues_upperbound[2])

    # save results for interpolation from 0.5 to other betas, direct reweighting