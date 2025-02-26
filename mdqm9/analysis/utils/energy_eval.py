import os
import numpy as np
import sys
sys.path.append("../thermointerpolants")
from tqdm import tqdm

from openmm import unit
from openff.toolkit.topology import Molecule
from openff.toolkit.topology import Topology
import openmm
from openmmforcefields.generators import GAFFTemplateGenerator
from openmm.app import ForceField


from paper.mdqm9.analyze.utils import eval_dataset


"""
Contains energy_evaluation for mdqm9 dataset. Adapted from: 
https://github.com/olsson-group/sma-md/blob/main/analysis/energy_solvent.py
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
    mol_indices = 10504
    sdf_path = "molecular_interpolants/replica_exchange_trajs/mdqm9-nc.sdf"  # path to mdqm9-nc.sdf
    hdf5_path = "molecular_interpolants/replica_exchange_trajs/mdqm9-nc.hdf5"  # path to mdqm9-nc.hdf5

    T0 = 1000
    T1 = 300
    filename = f"10506_no_{T1}_v2_256_new_scale_{T0}to{T1}K_forward.npy" #f"00031_no_{T1}_v2_{T0}to{T1}K_forward.npy"
    confs_path = f"../results/final/md_ti/samples_{filename}"

    dataset = eval_dataset.MDQM9EvalDataset(sdf_path, hdf5_path)
    mol_dic = dataset[mol_indices]

    # conformations
    confs = np.load(confs_path)

    confs_T0 = confs[:, 0, :, :]/eval_dataset.SCALING_FACTOR
    confs_T1 = confs[:, -1, :, :]/eval_dataset.SCALING_FACTOR

    E0 = eval_energy(mol_dic, T0, confs_T0)
    E1 = eval_energy(mol_dic, T1, confs_T1)

    if not os.path.exists("../results/"):
        os.makedirs("../results/")
    
    np.save(f"../results/final/md_ti/E0s_{filename}", E0)
    np.save(f"../results/final/md_ti/E1s_{filename}", E1)

