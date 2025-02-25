import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import wandb
import numpy as np
import torch

from torch_geometric.loader import DataLoader

from thermo.ambient.models.cpainn import cPaiNN
from data import mdqm9_ambient as mdqm9
from thermo import utils
from thermo.ambient import integrators


def sample(config: argparse.Namespace, b: torch.nn.Module) -> None:
    # set seeds for reproducibility
    torch.manual_seed(config.seed)  
    np.random.seed(config.seed)

    if not os.path.exists(config.data_save_path):
        os.makedirs(config.data_save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.use_wandb:
        wandb.init(project=f"{config.project_name}, sampling logs", name=config.model_save_name)

    # data related stuff
    test_dataset = mdqm9.MDQM9SamplerDataset(
        traj_filename=config.mdqm9_traj_filename,
        sdf_filename='mdqm9.sdf',
        traj_path=config.traj_path,
        sdf_path=config.sdf_path,
        split='test',
        T0=config.sampling_T0,
        T1=config.sampling_T1,
        scale=config.scale_trajs,
        cutoff=config.cutoff,
        use_latent_trajs=False,
        n_latent_samples=0,
    )

    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )

    # set integration settings
    integrator = integrators.MoleculeIntegrator(
        b=b,
        method="dopri5",
        rtol=config.rtol,
        atol=config.atol,
        n_step=config.n_steps,
        return_dlogp=config.return_dlogp,
        reverse_ode=False,
        )

    latent_noises = []
    latent_dlogps = []
    
    samples = []
    dlogps = []

    b.eval()
    b.to(device)
    for i, batch in enumerate(test_loader):
        batch = batch.to(device)

        latent_noise = batch.bg_z.detach().cpu().numpy()
        latent_dlogp = batch.bg_dlogp.detach().cpu().numpy()

        batch_idx = batch.batch.detach().cpu().numpy()
        latent_noise = np.array([latent_noise[batch_idx == i] for i in range(batch_idx.max() + 1)])

        latent_noises.append(latent_noise)
        latent_dlogps.append(latent_dlogp)

        np.save(os.path.join(config.data_save_path, f'latent_noises_{config.data_save_name}.npy'), np.concatenate(latent_noises, axis=0))
        np.save(os.path.join(config.data_save_path, f'latent_dlogps_{config.data_save_name}.npy'), np.concatenate(latent_dlogps, axis=0))

        sample, dlogp, n_steps, _ = integrator.rollout(batch)

        sample = sample.detach().cpu().numpy()
        dlogp = dlogp.detach().cpu().numpy()

        sample = np.array([sample[:, batch_idx == i] for i in range(batch_idx.max() + 1)])
        samples.append(sample)
        np.save(os.path.join(config.data_save_path, f'samples_{config.data_save_name}.npy'), np.concatenate(samples, axis=0))

        if config.return_dlogp:
            dlogp = dlogp[-1, :]
            dlogps.append(dlogp)

            np.save(os.path.join(config.data_save_path, f'dlogps_{config.data_save_name}.npy'), np.concatenate(dlogps, axis=0))

        if config.use_wandb:
            wandb.log({"batch_i": i+1, "n_batches": len(test_loader)})
        else:
            print(f"Batch {i+1}/{len(test_loader)}")
    print(f"Number sampling steps: {n_steps}")

    np.save(os.path.join(config.data_save_path, f'latent_noises_{config.data_save_name}.npy'), np.concatenate(latent_noises, axis=0))
    np.save(os.path.join(config.data_save_path, f'latent_dlogps_{config.data_save_name}.npy'), np.concatenate(latent_dlogps, axis=0))
    np.save(os.path.join(config.data_save_path, f'samples_{config.data_save_name}.npy'), np.concatenate(samples, axis=0))

    if config.return_dlogp:
        np.save(os.path.join(config.data_save_path, f'dlogps_{config.data_save_name}.npy'), np.concatenate(dlogps, axis=0))

    print("Finished forward sampling...\n")

    if config.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    config = utils.load_config("mdqm9/config/ambient/", "00031_settings_no_300.json")  # change this file to run with different settings

    b = cPaiNN(
        n_features=config.n_features,
        score_layers = config.score_layers,
        temp_length=config.temp_length,
    )

    b.load_state_dict(torch.load(f'{config.model_save_path}/{config.model_save_name}_{config.model_epoch}_weights.pt', map_location=torch.device('cpu') if not torch.cuda.is_available() else None))
    sample(config, b)
