import os
import sys
sys.path.append("../thermodynamic-interpolation")

import numpy as np
import h5py as h5

from rdkit import Chem

from mdqm9.analysis.results_00031 import gen_z_matrix, gen_torsions, gen_bond_angles, gen_bond_lengths
from mdqm9.thermo import utils
from mdqm9.data import mdqm9_ambient as mdqm9


if __name__ == '__main__':
    config = utils.load_config('mdqm9/config/ambient/', '10506_settings_no_300.json')  # example, change config to run with different settings
    sdf_path = "../data/mols/"  # path to sdf
    sdf_filename = "mdqm9.sdf"

    h5_filename = "mdqm9-nc.hdf5"
    h5_path = "../data/mols/"  # path to mdqm9-nc.hdf5
    h5_idx = 10504  # NOTE: Use 10504 for larger molecule, i.e. 3p2y1y (indexes differ from the mol id here, but this is the correct)

    traj_path = "../data/mols/rotated_replica_exchange_trajs/"
    ambient_md_path = "../samples/ambient_md/10506"
    ambient_lti_path = "../samples/ambient_lti/10506"
    latent_path = "../samples/latent/10506"

    # save
    results_save_path = f"../results/10506/{config.data_save_name}"
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)

    x0s_md_ti = np.load(f"{ambient_md_path}/samples_{config.data_save_name}.npy")[:, 0, :, :]/mdqm9.SCALING_FACTOR
    x1s_md_ti = np.load(f"{ambient_md_path}/samples_{config.data_save_name}.npy")[:, -1, :, :]/mdqm9.SCALING_FACTOR

    x0s_bg_ti = np.load(f"{ambient_lti_path}/samples_{config.data_save_name}.npy")[:, 0, :, :]/mdqm9.SCALING_FACTOR
    x1s_bg_ti = np.load(f"{ambient_lti_path}/samples_{config.data_save_name}.npy")[:, -1, :, :]/mdqm9.SCALING_FACTOR

    temp_index_dict = dict(zip(np.arange(300, 1001, step=100), list(range(8))))
    x0s_md = mdqm9.get_mdqm9_trajs(temp_index_dict[config.T0s[0]], config.mdqm9_traj_filename, traj_path, scale=False, split='train')
    x1s_md = mdqm9.get_mdqm9_trajs(temp_index_dict[config.T1s[0]], config.mdqm9_traj_filename, traj_path, scale=False, split='train')

    suppl = Chem.SDMolSupplier(os.path.join(sdf_path, sdf_filename), removeHs=False, sanitize=True)
    file_id = int(config.mdqm9_traj_filename.split('.')[0])
    mol = suppl[file_id]

    h5_file = h5.File(os.path.join(h5_path, h5_filename), 'r')
    h5_md_traj = h5_file[f"{h5_idx}"]["trajectories"]["md_0"][:]

    z_matrix_h5_md = gen_z_matrix(mol, h5_md_traj)
    torsions_h5_md = gen_torsions(z_matrix_h5_md)

    z_matrix_md_ti_0 = gen_z_matrix(mol, x0s_md_ti)
    z_matrix_md_ti_1 = gen_z_matrix(mol, x1s_md_ti)

    z_matrix_bg_ti_0 = gen_z_matrix(mol, x0s_bg_ti)
    z_matrix_bg_ti_1 = gen_z_matrix(mol, x1s_bg_ti)

    z_matrix_md_1 = gen_z_matrix(mol, x1s_md)
    z_matrix_md_0 = gen_z_matrix(mol, x0s_md)

    torsions_md_ti_0 = gen_torsions(z_matrix_md_ti_0)
    torsions_md_ti_1 = gen_torsions(z_matrix_md_ti_1)

    torsions_bg_ti_0 = gen_torsions(z_matrix_bg_ti_0)
    torsions_bg_ti_1 = gen_torsions(z_matrix_bg_ti_1)

    torsions_md_1 = gen_torsions(z_matrix_md_1)
    torsions_md_0 = gen_torsions(z_matrix_md_0)

    bond_angles_md_ti_0 = gen_bond_angles(z_matrix_md_ti_0)
    bond_angles_md_ti_1 = gen_bond_angles(z_matrix_md_ti_1)

    bond_angles_bg_ti_0 = gen_bond_angles(z_matrix_bg_ti_0)
    bond_angles_bg_ti_1 = gen_bond_angles(z_matrix_bg_ti_1)

    bond_angles_md_1 = gen_bond_angles(z_matrix_md_1)
    bond_angles_md_0 = gen_bond_angles(z_matrix_md_0)

    bond_lengths_md_ti_0 = gen_bond_lengths(z_matrix_md_ti_0)
    bond_lengths_md_ti_1 = gen_bond_lengths(z_matrix_md_ti_1)

    bond_lengths_bg_ti_0 = gen_bond_lengths(z_matrix_bg_ti_0)
    bond_lengths_bg_ti_1 = gen_bond_lengths(z_matrix_bg_ti_1)

    bond_lengths_md_1 = gen_bond_lengths(z_matrix_md_1)
    bond_lengths_md_0 = gen_bond_lengths(z_matrix_md_0)

    np.save(f"{results_save_path}/torsions_h5_md.npy", torsions_h5_md)

    np.save(f"{results_save_path}/z_matrix_md_ti_0.npy", z_matrix_md_ti_0)
    np.save(f"{results_save_path}/z_matrix_md_ti_1.npy", z_matrix_md_ti_1)
    np.save(f"{results_save_path}/z_matrix_bg_ti_0.npy", z_matrix_bg_ti_0)
    np.save(f"{results_save_path}/z_matrix_bg_ti_1.npy", z_matrix_bg_ti_1)
    np.save(f"{results_save_path}/z_matrix_md_0.npy", z_matrix_md_0)
    np.save(f"{results_save_path}/z_matrix_md_1.npy", z_matrix_md_1)

    np.save(f"{results_save_path}/torsions_md_ti_0.npy", torsions_md_ti_0)
    np.save(f"{results_save_path}/torsions_md_ti_1.npy", torsions_md_ti_1)
    np.save(f"{results_save_path}/torsions_bg_ti_0.npy", torsions_md_ti_0)
    np.save(f"{results_save_path}/torsions_bg_ti_1.npy", torsions_md_ti_1)
    np.save(f"{results_save_path}/torsions_md_0.npy", torsions_md_0)
    np.save(f"{results_save_path}/torsions_md_1.npy", torsions_md_1)

    np.save(f"{results_save_path}/bond_angles_md_ti_0.npy", bond_angles_md_ti_0)
    np.save(f"{results_save_path}/bond_angles_md_ti_1.npy", bond_angles_md_ti_1)
    np.save(f"{results_save_path}/bond_angles_bg_ti_0.npy", bond_angles_md_ti_0)
    np.save(f"{results_save_path}/bond_angles_bg_ti_1.npy", bond_angles_md_ti_1)
    np.save(f"{results_save_path}/bond_angles_md_0.npy", bond_angles_md_0)
    np.save(f"{results_save_path}/bond_angles_md_1.npy", bond_angles_md_1)

    np.save(f"{results_save_path}/bond_lengths_md_ti_0.npy", bond_lengths_md_ti_0)
    np.save(f"{results_save_path}/bond_lengths_md_ti_1.npy", bond_lengths_md_ti_1)
    np.save(f"{results_save_path}/bond_lengths_bg_ti_0.npy", bond_lengths_md_ti_0)
    np.save(f"{results_save_path}/bond_lengths_bg_ti_1.npy", bond_lengths_md_ti_1)
    np.save(f"{results_save_path}/bond_lengths_md_0.npy", bond_lengths_md_0)
    np.save(f"{results_save_path}/bond_lengths_md_1.npy", bond_lengths_md_1)

    np.save(f"{results_save_path}/x0s_md.npy", x0s_md)
    np.save(f"{results_save_path}/x1s_md.npy", x1s_md)

