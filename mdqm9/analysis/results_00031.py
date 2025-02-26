import os
import sys
sys.path.append("../thermodynamic-interpolation")

import numpy as np
import torch

from rdkit import Chem

from mdqm9.thermo import utils #paper.mdqm9.utils as utils
from mdqm9.analysis.utils import sort_atoms, z_matrix, ess, free_energy, sensititvity
from mdqm9.data import mdqm9_ambient as mdqm9


def gen_z_matrix(mol, samples):
    atom_order, _, ref_atoms = sort_atoms.compute_atom_order_and_references_groups(mol)
    sorted_samples = torch.tensor(samples[:, atom_order, :], dtype=torch.float32)
    return z_matrix.construct_z_matrix_batch(sorted_samples, ref_atoms).numpy()


def get_importance_weights(z0s, Es, neg_dlogps_bg, neg_dlogps_ti):
    return ess.calc_importance_weights(z0s=z0s, E1s=Es, neg_dlogps_bg=neg_dlogps_bg, neg_dlogps_ti=neg_dlogps_ti)


def get_ti_weights(E0s, E1s, neg_dlogps_ti):
    return ess.calc_ti_weights(E0s=E0s, E1s=E1s, neg_dlogps_ti=neg_dlogps_ti)


def gen_free_energy_tfep_md_ti(E0s, E1s, neg_dlogps_ti, n_bootstrap=1000, k=None):
    phis, _ = free_energy.calc_phis_tfep(E0s=E0s, E1s=E1s, neg_dlogps_ti=neg_dlogps_ti, k=k)

    # bootstrap 95% confints
    bootstrap_estimates = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indexes = np.random.choice(np.arange(len(phis)), len(phis), replace=True)
        E0s_bootstrap = E0s[indexes]
        E1s_bootstrap = E1s[indexes]
        neg_dlogps_ti_bootstrap = neg_dlogps_ti[indexes]

        phis_bootstrap, _ = free_energy.calc_phis_tfep(E0s=E0s_bootstrap, E1s=E1s_bootstrap, neg_dlogps_ti=neg_dlogps_ti_bootstrap, k=k)
        bootstrap_estimates[i] = free_energy.calc_tfep_dF(phis=phis_bootstrap, weights=np.ones_like(phis_bootstrap))
    
    lower_bound = np.percentile(bootstrap_estimates, 2.5)
    upper_bound = np.percentile(bootstrap_estimates, 97.5)
    return free_energy.calc_tfep_dF(phis=phis, weights=np.ones_like(phis)), [lower_bound, upper_bound]
    

def gen_free_energy_bg(Es_T0, neg_dlogps_bg_T0, Es_T1, neg_dlogps_bg_T1, n_bootstrap=1000, k=None):
    phis0 = free_energy.calc_phis_bg(Es=Es_T0, neg_dlogps_bg=neg_dlogps_bg_T0, k=k)
    phis1 = free_energy.calc_phis_bg(Es=Es_T1, neg_dlogps_bg=neg_dlogps_bg_T1, k=k)

    dF = free_energy.calc_bg_dF(phis=phis1) - free_energy.calc_bg_dF(phis=phis0)

    # bootstrap 95% confints
    bootstrap_estimates = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indexes0 = np.random.choice(np.arange(len(phis0)), len(phis0), replace=True)
        indexes1 = np.random.choice(np.arange(len(phis1)), len(phis1), replace=True)

        Es_bg_ref_T0 = Es_T0[indexes0]
        neg_dlogps_bg_ref_T0 = neg_dlogps_bg_T0[indexes0]

        Es_bg_ref_T1 = Es_T1[indexes1]
        neg_dlogps_bg_ref_T1 = neg_dlogps_bg_T1[indexes1]

        phis0_bootstrap = free_energy.calc_phis_bg(Es=Es_bg_ref_T0, neg_dlogps_bg=neg_dlogps_bg_ref_T0, k=k)
        phis1_bootstrap = free_energy.calc_phis_bg(Es=Es_bg_ref_T1, neg_dlogps_bg=neg_dlogps_bg_ref_T1, k=k)

        bootstrap_estimates[i] = free_energy.calc_bg_dF(phis=phis1_bootstrap) - free_energy.calc_bg_dF(phis=phis0_bootstrap)

    lower_bound = np.percentile(bootstrap_estimates, 2.5)
    upper_bound = np.percentile(bootstrap_estimates, 97.5)
    return dF, [lower_bound, upper_bound]


def gen_free_energy_bg_tfep(Es_T0, neg_dlogps_bg_T0, Es_T1, neg_dlogps_bg_T1, n_bootstrap=1000, k=None):
    phis = free_energy.calc_phis_bg_tfep(E0s=Es_T0, neg_dlogps_bg_T0=neg_dlogps_bg_T0, E1s=Es_T1, neg_dlogps_bg_T1=neg_dlogps_bg_T1, k=k)
    dF = free_energy.calc_tfep_dF(phis=phis, weights=np.ones_like(phis))

    # bootstrap 95% confints
    bootstrap_estimates = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indexes = np.random.choice(np.arange(len(phis)), len(phis), replace=True)

        Es_bg_ref_T0 = Es_T0[indexes]
        Es_bg_ref_T1 = Es_T1[indexes]
        neg_dlogps_bg_ref_T0 = neg_dlogps_bg_T0[indexes]
        neg_dlogps_bg_ref_T1 = neg_dlogps_bg_T1[indexes]

        phis_bootstrap = free_energy.calc_phis_bg_tfep(E0s=Es_bg_ref_T0, neg_dlogps_bg_T0=neg_dlogps_bg_ref_T0, E1s=Es_bg_ref_T1, neg_dlogps_bg_T1=neg_dlogps_bg_ref_T1, k=k)
        bootstrap_estimates[i] = free_energy.calc_tfep_dF(phis=phis_bootstrap, weights=np.ones_like(phis_bootstrap))

    lower_bound = np.percentile(bootstrap_estimates, 2.5)
    upper_bound = np.percentile(bootstrap_estimates, 97.5)
    return dF, [lower_bound, upper_bound]


def gen_ess_bg(z0s, E1s, neg_dlogps_bg, neg_dlogps_ti, k=None, n_bootstrap=1000):
    weights = ess.calc_importance_weights(z0s=z0s, E1s=E1s, neg_dlogps_bg=neg_dlogps_bg, neg_dlogps_ti=neg_dlogps_ti)
    if k==None:
        ess_val = ess.calc_ESS(weights)
    else:
        indexes_to_keep = sensititvity.filter_iqr(weights, k=k)
        weights = weights[indexes_to_keep]
        ess_val = ess.calc_ESS(weights)

    # bootstrap 95% confints
    bootstrap_estimates = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indexes = np.random.choice(np.arange(len(weights)), len(weights), replace=True)
        weights_bootstrap = weights[indexes]
        bootstrap_estimates[i] = ess.calc_ESS(weights_bootstrap)

    lower_bound = np.percentile(bootstrap_estimates, 2.5)
    upper_bound = np.percentile(bootstrap_estimates, 97.5)
    return ess_val, [lower_bound, upper_bound]


def gen_ess_ti(E0s, E1s, neg_dlogps_ti, k=None, n_bootstrap=1000):
    weights = ess.calc_ti_weights(E0s=E0s, E1s=E1s, neg_dlogps_ti=neg_dlogps_ti)
    if k==None:
        ess_val = ess.calc_ESS(weights)
    else:
        indexes_to_keep = sensititvity.filter_iqr(weights, k=k)
        weights = weights[indexes_to_keep]
        ess_val = ess.calc_ESS(weights)

    # bootstrap 95% confints
    bootstrap_estimates = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indexes = np.random.choice(np.arange(len(weights)), len(weights), replace=True)
        weights_bootstrap = weights[indexes]
        bootstrap_estimates[i] = ess.calc_ESS(weights_bootstrap)

    lower_bound = np.percentile(bootstrap_estimates, 2.5)
    upper_bound = np.percentile(bootstrap_estimates, 97.5)
    return ess_val, [lower_bound, upper_bound]


def gen_torsions(z_matrix):
    return z_matrix[:, 2:, 2]


def gen_bond_angles(z_matrix):
    return z_matrix[:, 1:, 1]


def gen_bond_lengths(z_matrix):
    return z_matrix[:, :, 0]


if __name__ == '__main__':
    # basic settings
    config = utils.load_config('mdqm9/config/ambient/', '00031_settings_no_900.json')  # example, change config to run with different settings
    sdf_path = "../data/mols/"  # path to sdf
    sdf_filename = "mdqm9.sdf"

    traj_path = "../data/mols/rotated_replica_exchange_trajs/"
    ambient_md_path = "../samples/ambient_md/00031"
    ambient_lti_path = "../samples/ambient_lti/00031"
    latent_path = "../samples/latent/00031"

    results_save_path = f"../results/00031/{config.data_save_name}"
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)

    # load molecule
    file_id = 31
    suppl = Chem.SDMolSupplier(os.path.join(sdf_path, sdf_filename), removeHs=False, sanitize=True)
    mol = suppl[file_id]

    # TI model outputs
    x0s_md_ti = np.load(f"{ambient_md_path}/samples_{config.data_save_name}.npy")[:, 0, :, :]/mdqm9.SCALING_FACTOR_31
    x1s_md_ti = np.load(f"{ambient_md_path}/samples_{config.data_save_name}.npy")[:, -1, :, :]/mdqm9.SCALING_FACTOR_31

    E0s_md_ti = np.load(f"{ambient_md_path}/E0s_{config.data_save_name}.npy")
    E1s_md_ti = np.load(f"{ambient_md_path}/E1s_{config.data_save_name}.npy")

    neg_dlogps_md_ti = np.load(f"{ambient_md_path}/dlogps_{config.data_save_name}.npy")

    x0s_bg_ti = np.load(f"{ambient_lti_path}/samples_{config.data_save_name}.npy")[:, 0, :, :]/mdqm9.SCALING_FACTOR_31
    x1s_bg_ti = np.load(f"{ambient_lti_path}/samples_{config.data_save_name}.npy")[:, -1, :, :]/mdqm9.SCALING_FACTOR_31

    E0s_bg_ti = np.load(f"{ambient_lti_path}/E0s_{config.data_save_name}.npy")
    E1s_bg_ti = np.load(f"{ambient_lti_path}/E1s_{config.data_save_name}.npy")
    
    zs_bg_ti = np.load(f"{ambient_lti_path}/latent_noises_{config.data_save_name}.npy")  # noise samples used to produce the x0 fed to TI model
    neg_dlogps_bg = np.load(f"{ambient_lti_path}/latent_dlogps_{config.data_save_name}.npy")
    neg_dlogps_ti = np.load(f"{ambient_lti_path}/dlogps_{config.data_save_name}.npy")

    zs_bg_ref_T0 = np.load(f"{latent_path}/samples_{config.sampling_T0}K.npy")[:25_000, 0, :, :]  # noise samples used to produce the x used as BG reference
    zs_bg_ref_T1 = np.load(f"{latent_path}/samples_{config.sampling_T1}K.npy")[:25_000, 0, :, :]  # noise samples used to produce the x used as BG reference

    xs_bg_ref_T0 = np.load(f"{latent_path}/samples_{config.sampling_T0}K.npy")[:25_000, -1, :, :]/mdqm9.SCALING_FACTOR_31  
    xs_bg_ref_T1 = np.load(f"{latent_path}/samples_{config.sampling_T1}K.npy")[:25_000, -1, :, :]/mdqm9.SCALING_FACTOR_31

    neg_dlogps_bg_ref_T0 = np.load(f"{latent_path}/dlogps_{config.sampling_T0}K.npy")[:25_000]
    neg_dlogps_bg_ref_T1 = np.load(f"{latent_path}/dlogps_{config.sampling_T1}K.npy")[:25_000]  # negative change in log prob. for BG reference

    Es_bg_ref_T0 = np.load(f"{latent_path}/Es_{config.sampling_T0}K.npy")[:25_000]  # energy evaluated at BG reference x0
    Es_bg_ref_T1 = np.load(f"{latent_path}/Es_{config.sampling_T1}K.npy")[:25_000]  # energy evaluated at BG reference x1

    temp_index_dict = dict(zip(np.arange(300, 1001, step=100), list(range(8))))
    x0s_md = mdqm9.get_mdqm9_trajs(temp_index_dict[config.sampling_T0], config.mdqm9_traj_filename, config.traj_path, scale=False, split='train')
    x1s_md = mdqm9.get_mdqm9_trajs(temp_index_dict[config.sampling_T1], config.mdqm9_traj_filename, config.traj_path, scale=False, split='train')

    # generate z-matrix
    z_matrix_md_ti_0 = gen_z_matrix(mol, x0s_md_ti)
    z_matrix_md_ti_1 = gen_z_matrix(mol, x1s_md_ti)
    z_matrix_bg_ti_0 = gen_z_matrix(mol, x0s_bg_ti)
    z_matrix_bg_ti_1 = gen_z_matrix(mol, x1s_bg_ti)
    z_matrix_bg_ref_T0 = gen_z_matrix(mol, xs_bg_ref_T0)
    z_matrix_bg_ref_T1 = gen_z_matrix(mol, xs_bg_ref_T1)
    z_matrix_md_1 = gen_z_matrix(mol, x1s_md)
    z_matrix_md_0 = gen_z_matrix(mol, x0s_md)

    # get torsions
    torsions_md_ti_0 = gen_torsions(z_matrix_md_ti_0)
    torsions_md_ti_1 = gen_torsions(z_matrix_md_ti_1)
    torsions_bg_ti_0 = gen_torsions(z_matrix_bg_ti_0)
    torsions_bg_ti_1 = gen_torsions(z_matrix_bg_ti_1)
    torsions_bg_ref_T0 = gen_torsions(z_matrix_bg_ref_T0)
    torsions_bg_ref_T1 = gen_torsions(z_matrix_bg_ref_T1)
    torsions_md_1 = gen_torsions(z_matrix_md_1)
    torsions_md_0 = gen_torsions(z_matrix_md_0)

    # get bond angles
    bond_angles_md_ti_0 = gen_bond_angles(z_matrix_md_ti_0)
    bond_angles_md_ti_1 = gen_bond_angles(z_matrix_md_ti_1)
    bond_angles_bg_ti_0 = gen_bond_angles(z_matrix_bg_ti_0)
    bond_angles_bg_ti_1 = gen_bond_angles(z_matrix_bg_ti_1)
    bond_angles_bg_ref_T0 = gen_bond_angles(z_matrix_bg_ref_T0)
    bond_angles_bg_ref_T1 = gen_bond_angles(z_matrix_bg_ref_T1)
    bond_angles_md_1 = gen_bond_angles(z_matrix_md_1)
    bond_angles_md_0 = gen_bond_angles(z_matrix_md_0)

    # get bond lengths
    bond_lengths_md_ti_0 = gen_bond_lengths(z_matrix_md_ti_0)
    bond_lengths_md_ti_1 = gen_bond_lengths(z_matrix_md_ti_1)
    bond_lengths_bg_ti_0 = gen_bond_lengths(z_matrix_bg_ti_0)
    bond_lengths_bg_ti_1 = gen_bond_lengths(z_matrix_bg_ti_1)
    bond_lengths_bg_ref_T0 = gen_bond_lengths(z_matrix_bg_ref_T0)
    bond_lengths_bg_ref_T1 = gen_bond_lengths(z_matrix_bg_ref_T1)
    bond_lengths_md_1 = gen_bond_lengths(z_matrix_md_1)
    bond_lengths_md_0 = gen_bond_lengths(z_matrix_md_0)

    # get ESS
    ess_md_ti, ess_md_ti_ci = gen_ess_ti(E0s_md_ti, E1s_md_ti, neg_dlogps_md_ti, k=100)  # ESS for MD/TI map (OBS: map x0 -> x1)
    ess_md_ti_percentage = ess_md_ti/len(neg_dlogps_md_ti)*100
    ess_md_ti_ci_percentage = [ci/len(neg_dlogps_md_ti)*100 for ci in ess_md_ti_ci]

    ess_bg_ti, ess_bg_ti_ci = gen_ess_bg(zs_bg_ti, E1s_bg_ti, neg_dlogps_bg, neg_dlogps_ti, k=100)  # ESS for BG/TI map (OBS: complete map z -> x1)
    ess_bg_ti_percentage = ess_bg_ti/len(neg_dlogps_bg)*100
    ess_bg_ti_ci_percentage = [ci/len(neg_dlogps_bg)*100 for ci in ess_bg_ti_ci]

    ess_bg_T0, ess_bg_T0_ci = gen_ess_bg(zs_bg_ref_T0, Es_bg_ref_T0, neg_dlogps_bg_ref_T0, np.zeros_like(neg_dlogps_bg_ref_T0), k=100)  # ESS for BG reference map (OBS: map z -> x, i.e. ESS of what we give to TI)
    ess_bg_T0_percentage = ess_bg_T0/len(neg_dlogps_bg_ref_T0)*100
    ess_bg_T0_ci_percentage = [ci/len(neg_dlogps_bg_ref_T0)*100 for ci in ess_bg_T0_ci]

    # get free energy change
    df_md_ti, dF_md_ti_ci = gen_free_energy_tfep_md_ti(E0s_md_ti, E1s_md_ti, neg_dlogps_md_ti, k=100)
    df_bg_ti_tfep, dF_bg_ti_tfep_ci = gen_free_energy_bg_tfep(E0s_bg_ti, neg_dlogps_bg, E1s_bg_ti, neg_dlogps_bg+neg_dlogps_ti, k=100)
    dF_bg_ref, dF_bg_ref_ci = gen_free_energy_bg(Es_bg_ref_T0, neg_dlogps_bg_ref_T0, Es_bg_ref_T1, neg_dlogps_bg_ref_T1, k=100)
    dF_bg_ref_tfep, dF_bg_ref_tfep_ci = gen_free_energy_bg_tfep(Es_bg_ref_T0, neg_dlogps_bg_ref_T0, Es_bg_ref_T1, neg_dlogps_bg_ref_T1, k=100)

    # importance weights
    weights_md_ti = get_ti_weights(E0s_md_ti, E1s_md_ti, neg_dlogps_md_ti)  # weights for MD/TI map
    indexes_to_keep = sensititvity.filter_iqr(weights_md_ti, k=100)
    weights_md_ti = weights_md_ti[indexes_to_keep]
    torsions_md_ti_1 = torsions_md_ti_1[indexes_to_keep]
    bond_angles_md_ti_1 = bond_angles_md_ti_1[indexes_to_keep]
    bond_lengths_md_ti_1 = bond_lengths_md_ti_1[indexes_to_keep]

    weights_bg_ti_T1 = get_importance_weights(zs_bg_ti, E1s_bg_ti, neg_dlogps_bg, neg_dlogps_ti)  # weights for TI target temperature marginals
    indexes_to_keep = sensititvity.filter_iqr(weights_bg_ti_T1, k=100)
    weights_bg_ti_T1 = weights_bg_ti_T1[indexes_to_keep]
    torsions_bg_ti_1 = torsions_bg_ti_1[indexes_to_keep]
    bond_angles_bg_ti_1 = bond_angles_bg_ti_1[indexes_to_keep]
    bond_lengths_bg_ti_1 = bond_lengths_bg_ti_1[indexes_to_keep]
    
    weights_bg_ti_T0 = get_importance_weights(zs_bg_ti, E0s_bg_ti, neg_dlogps_bg, np.zeros_like(neg_dlogps_ti))  # weights for TI initial temperature marginals    
    weights_bg_ref_T0 = get_importance_weights(zs_bg_ref_T0, Es_bg_ref_T0, neg_dlogps_bg_ref_T0, np.zeros_like(neg_dlogps_bg_ref_T0))  # weights for BG reference marginals at initial temperature
    weights_bg_ref_T1 = get_importance_weights(zs_bg_ref_T1, Es_bg_ref_T1, neg_dlogps_bg_ref_T1, np.zeros_like(neg_dlogps_bg_ref_T1))  # weights for BG reference marginals at target temperature

    # print bg/ti stats
    print(f"ESS (BG at T0): {ess_bg_T0_percentage:.4f} -+ {ess_bg_T0_ci_percentage}, ESS (BG/TI): {ess_bg_ti_percentage:.4f} -+ {ess_bg_ti_ci_percentage}, ESS (MD/TI): {ess_md_ti_percentage:.4f} -+ {ess_md_ti_ci_percentage}")
    print(f"dF (BG/TI): {df_bg_ti_tfep:.4f} -+ {dF_bg_ti_tfep_ci}, dF (MD/TI): {df_md_ti:.4f} -+ {dF_md_ti_ci}")
    print(f"dF (BG Ref.): {dF_bg_ref:.4f} -+ {dF_bg_ref_ci}, dF (BG Ref. TFEP): {dF_bg_ref_tfep:.4f} -+ {dF_bg_ref_tfep_ci}")

    # save results
    np.save(os.path.join(results_save_path, "torsions_md_ti_0.npy"), torsions_md_ti_0)
    np.save(os.path.join(results_save_path, "torsions_md_ti_1.npy"), torsions_md_ti_1)
    np.save(os.path.join(results_save_path, "torsions_bg_ti_0.npy"), torsions_bg_ti_0)
    np.save(os.path.join(results_save_path, "torsions_bg_ti_1.npy"), torsions_bg_ti_1)
    np.save(os.path.join(results_save_path, "torsions_bg_ref_T0.npy"), torsions_bg_ref_T0)
    np.save(os.path.join(results_save_path, "torsions_bg_ref_T1.npy"), torsions_bg_ref_T1)
    np.save(os.path.join(results_save_path, "torsions_md_T1.npy"), torsions_md_1)
    np.save(os.path.join(results_save_path, "torsions_md_T0.npy"), torsions_md_0)

    np.save(os.path.join(results_save_path, "bond_angles_md_ti_0.npy"), bond_angles_md_ti_0)
    np.save(os.path.join(results_save_path, "bond_angles_md_ti_1.npy"), bond_angles_md_ti_1)
    np.save(os.path.join(results_save_path, "bond_angles_bg_ti_0.npy"), bond_angles_bg_ti_0)
    np.save(os.path.join(results_save_path, "bond_angles_bg_ti_1.npy"), bond_angles_bg_ti_1)
    np.save(os.path.join(results_save_path, "bond_angles_bg_ref_T0.npy"), bond_angles_bg_ref_T0)
    np.save(os.path.join(results_save_path, "bond_angles_bg_ref_T1.npy"), bond_angles_bg_ref_T1)
    np.save(os.path.join(results_save_path, "bond_angles_md_T1.npy"), bond_angles_md_1)
    np.save(os.path.join(results_save_path, "bond_angles_md_T0.npy"), bond_angles_md_0)

    np.save(os.path.join(results_save_path, "bond_lengths_md_ti_0.npy"), bond_lengths_md_ti_0)
    np.save(os.path.join(results_save_path, "bond_lengths_md_ti_1.npy"), bond_lengths_md_ti_1)
    np.save(os.path.join(results_save_path, "bond_lengths_bg_ti_0.npy"), bond_lengths_bg_ti_0)
    np.save(os.path.join(results_save_path, "bond_lengths_bg_ti_1.npy"), bond_lengths_bg_ti_1)
    np.save(os.path.join(results_save_path, "bond_lengths_bg_ref_T0.npy"), bond_lengths_bg_ref_T0)
    np.save(os.path.join(results_save_path, "bond_lengths_bg_ref_T1.npy"), bond_lengths_bg_ref_T1)
    np.save(os.path.join(results_save_path, "bond_lengths_md_1.npy"), bond_lengths_md_1)
    np.save(os.path.join(results_save_path, "bond_lengths_md_0.npy"), bond_lengths_md_0)

    np.save(os.path.join(results_save_path, "ess_md_ti_percentage.npy"), ess_md_ti_percentage)
    np.save(os.path.join(results_save_path, "ess_bg_ti_percentage.npy"), ess_bg_ti_percentage)
    np.save(os.path.join(results_save_path, "ess_bg_T0_percentage.npy"), ess_bg_T0_percentage)

    np.save(os.path.join(results_save_path, "ess_md_ti_ci_percentage.npy"), ess_md_ti_ci_percentage)
    np.save(os.path.join(results_save_path, "ess_bg_ti_ci_percentage.npy"), ess_bg_ti_ci_percentage)
    np.save(os.path.join(results_save_path, "ess_bg_T0_ci_percentage.npy"), ess_bg_T0_ci_percentage)

    np.save(os.path.join(results_save_path, "df_md_ti.npy"), df_md_ti)
    np.save(os.path.join(results_save_path, "dF_bg_ti_tfep.npy"), df_bg_ti_tfep)
    np.save(os.path.join(results_save_path, "dF_bg_ref.npy"), dF_bg_ref)
    np.save(os.path.join(results_save_path, "dF_bg_ref_tfep.npy"), dF_bg_ref_tfep)

    np.save(os.path.join(results_save_path, "dF_bg_ref_ci.npy"), dF_bg_ref_ci)
    np.save(os.path.join(results_save_path, "dF_bg_ti_tfep_ci.npy"), dF_bg_ti_tfep_ci)
    np.save(os.path.join(results_save_path, "dF_md_ti_ci.npy"), dF_md_ti_ci)
    np.save(os.path.join(results_save_path, "dF_bg_ref_tfep_ci.npy"), dF_bg_ref_tfep_ci)

    np.save(os.path.join(results_save_path, "weights_md_ti.npy"), weights_md_ti)
    np.save(os.path.join(results_save_path, "weights_bg_ti_T1.npy"), weights_bg_ti_T1)
    np.save(os.path.join(results_save_path, "weights_bg_ti_T0.npy"), weights_bg_ti_T0)
    np.save(os.path.join(results_save_path, "weights_bg_ref_T0.npy"), weights_bg_ref_T0)
    np.save(os.path.join(results_save_path, "weights_bg_ref_T1.npy"), weights_bg_ref_T1)

    print("Results saved.")

