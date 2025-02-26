import os
import sys
sys.path.append("../thermodynamic-interpolation")

import numpy as np
from tqdm import tqdm

import openmm
from openmm import unit
from openff.toolkit.topology import Molecule
from openff.toolkit.topology import Topology
from openmmforcefields.generators import GAFFTemplateGenerator
from openmm.app import ForceField

from mdqm9.analysis.utils import eval_dataset


""" 
Contains energy_evaluation for mdqm9 dataset. Adapted from: 
https://github.com/olsson-group/sma-md/blob/main/analysis/energy_solvent.py

NOTE: Due to package incompabilities, this file needs to be run using a separate environment. 
To install the required dependencies to run this file, make a clean conda environment from the file "ti_energy_env.yml"

"""


def eval_energy(mol_dict, T, confs):
    rd_mol = mol_dict["rdkit_mol"]
    partial_charges = mol_dict["partial_charges"]

    off_mol = Molecule.from_rdkit(rd_mol)
    off_mol.partial_charges = unit.Quantity(value=np.array(partial_charges), unit=unit.elementary_charge)
    gaff = GAFFTemplateGenerator(molecules=off_mol)

    topology = Topology.from_molecules(off_mol).to_openmm()
    forcefield = ForceField('amber/protein.ff14SB.xml')
    forcefield.registerTemplateGenerator(gaff.generator)
    system = forcefield.createSystem(topology)
    integrator = openmm.LangevinIntegrator(T * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond)
    context = openmm.Context(system, integrator)
    energies = []
    kB_NA = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    energy_factor = 1. / (integrator.getTemperature() * kB_NA) #0.4 mol/kj
    
    for conf in tqdm(confs, total=len(confs)):

        context.setPositions(conf)  # positions must be given in nm!
        state = context.getState(getEnergy=True)
        energy = state.getPotentialEnergy() * energy_factor
        energies.append(energy)

    return np.array(energies)


if __name__ == "__main__":
    T0 = 1000
    T1 = 300

    mol_indices = 31    #NOTE: index 31 corresponds to small molecule, i.e. N-Me. Use 10504 for larger molecule, i.e. 3p2y1y (indexes differ from the mol id here, but this is the correct)
    
    scaling_factor = eval_dataset.SCALING_FACTOR_31 if mol_indices == 31 else eval_dataset.SCALING_FACTOR_10506
    
    sdf_path = "../data/mols//mdqm9-nc.sdf"  # path to mdqm9-nc.sdf
    hdf5_path = "../data/mols/mdqm9-nc.hdf5"  # path to mdqm9-nc.hdf5

    filename = f"samples_00031_no_{T1}_{T0}to{T1}K.npy"  # name of samples file
    file_path = f"../samples/ambient_md/00031"    # name of directory where samples file is stored
    save_path = "../samples/ambient_md/"   # directory in which to save the results 

    dataset = eval_dataset.MDQM9EvalDataset(sdf_path, hdf5_path)
    mol_dic = dataset[mol_indices]

    # conformations
    confs = np.load(f"{file_path}/{filename}")

    confs_T0 = confs[:, 0, :, :]/scaling_factor
    confs_T1 = confs[:, -1, :, :]/scaling_factor

    E0 = eval_energy(mol_dic, T0, confs_T0)
    E1 = eval_energy(mol_dic, T1, confs_T1)

    if not os.path.exists("../results/"):
        os.makedirs("../results/")
    
    np.save(f"{save_path}/E0s_{filename}", E0)
    np.save(f"{save_path}/E1s_{filename}", E1)

