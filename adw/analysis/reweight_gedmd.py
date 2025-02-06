import numpy as np
import sys
sys.path.append('../thermodynamic-interpolation')
from gedmd.rff_tools import sample_rff_gaussian
from gedmd.rff import spectral_analysis_rff_generator
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

def calculate_weights(initial_samples, target_samples, dlogps, beta):

    dlogps_t = np.array(dlogps)[-1]

    target_samples_t = np.array(target_samples)[-1]
    initial_energy = np.array(calculate_energy(initial_samples))
    target_energy = np.array(calculate_energy(target_samples_t))

    log_weights = 1.0 * initial_energy - beta * target_energy - dlogps_t

    return target_samples_t, np.exp(log_weights)


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

def resample_with_weights(samples, weights, n_samples=None):
    if n_samples is None:
        n_samples = len(samples)
    normalized_weights = weights / np.sum(weights)
    resampled_indices = np.random.choice(len(samples), size=n_samples, replace=True, p=normalized_weights)
    resampled_samples = samples[resampled_indices]
    return resampled_samples

def weights_filter_iqr(weights):
    q1 = np.percentile(weights, 2)
    q3 = np.percentile(weights, 98)
    iqr = q3 - q1
    lower_bound = q1 - 10 * iqr
    upper_bound = q3 + 10 * iqr

    return (weights > lower_bound) & (weights < upper_bound)


if __name__ == "__main__":
    # example from 1.0 to other betas, extropolate, reweighting
    betas = [1.25, 1.5, 1.75, 2.0]

    # run model selection to find below optimal parameters
    p = 50
    sig_opt = 0.6
    n_env = 4
    cut_svd = 1e-4

    Omega = sample_rff_gaussian(1, p, sig_opt)

    second_eigenvalues_mean = []
    second_eigenvalues_lowerbound = []
    second_eigenvalues_upperbound = []

    for i, beta in enumerate(betas):
        target_samples = np.load(
            f"/beta_1.0_to_{beta}/samples.npy"
        )
        initial_samples = np.load(
            f"/beta_1.0_to_{beta}/initial_samples.npy"
        )
        dlogps = np.load(
            f"/beta_1.0_to_{beta}/dlogps.npy"
        )

        target_samples_t, weights = calculate_weights(initial_samples, target_samples, dlogps, beta)

        filter = weights_filter_iqr(weights)
        target_samples_filtered = target_samples_t[filter]
        print(f"Filtered out {len(weights) - len(target_samples_filtered)} samples")
        weights = weights[filter]

        # resample with weights
        resampled_samples = resample_with_weights(target_samples_filtered, weights)

        eigenvalues_mean, lowerbound, upperbound = bootstrap_eigenvalues(
            resampled_samples.reshape(1, -1), Omega, n_env, beta, cut_svd
        )
        second_eigenvalues_mean.append(eigenvalues_mean[2])
        second_eigenvalues_lowerbound.append(lowerbound[2])
        second_eigenvalues_upperbound.append(upperbound[2])
    
    # save results for extrapolation from 1.0 to other betas, reweighting