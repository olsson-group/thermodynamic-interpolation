import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import wandb
import numpy as np
import torch

from torch_geometric.loader import DataLoader

from data import mdqm9_latent as mdqm9
from thermo import utils
from thermo.latent import integrators
from thermo.latent.models.cpainn import cPaiNN
from tqdm import tqdm


def sample(config: argparse.Namespace, b: torch.nn.Module) -> None:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if not os.path.exists(config.data_save_path):
        os.makedirs(config.data_save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.use_wandb:
        wandb.init(project=f"{config.project_name}, sampling logs", name=config.model_save_name)

    dataset = mdqm9.SamplerDataset(
        traj_filename=config.mdqm9_traj_filename, 
        sdf_filename="mdqm9.sdf", 
        traj_path=config.traj_path, 
        sdf_path=config.sdf_path, 
        split='test', 
        T=config.sampling_T, 
        n_samples=config.n_samples,
        scale=config.scale_trajs, 
        cutoff=config.cutoff,
        align=config.align,
    )

    test_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(config.seed),
    )

    integrator = integrators.MoleculeIntegrator(b=b,
                                                method="dopri5",
                                                rtol=config.rtol,
                                                atol=config.atol,
                                                n_step=config.n_steps,
                                                return_dlogp=config.return_dlogp,
                                                reverse_ode=False)
    samples = []
    dlogps = []

    b.eval()
    b.to(device)
    for i, batch in enumerate(tqdm(test_loader)):
        batch = batch.to(device)
        sample, dlogp, batch_idx = integrator.rollout(batch)

        sample = sample.detach().cpu().numpy()
        dlogp = dlogp.detach().cpu().numpy()
        batch_idx = batch_idx.cpu().numpy()

        sample = np.array([sample[:, batch_idx == i] for i in range(batch_idx.max() + 1)])
        samples.append(sample)
        np.save(f'{config.data_save_path}/samples_{config.data_save_name}_forward.npy', np.concatenate(samples, axis=0))

        if config.return_dlogp:
            dlogp = dlogp[-1, :]
            dlogps.append(dlogp)

            np.save(f'{config.data_save_path}/dlogps_{config.data_save_name}_forward.npy', np.concatenate(dlogps, axis=0))

        if config.use_wandb:
            wandb.log({"batch_i": i+1, "n_batches": len(test_loader)})
        else:
            print(f"Batch {i+1}/{len(test_loader)}")

    samples = np.concatenate(samples, axis=0)
    np.save(f'samples_{config.data_save_name}_forward.npy', samples)

    if config.return_dlogp:
        np.save(f'dlogps_{config.data_save_name}_forward.npy', np.concatenate(dlogps, axis=0))

    print("Finished forward sampling...\n")

    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    """config = utils.load_config("paper/bg_thermo", "settings.json")
    b = torch.load(f'{config.model_save_path}/{config.model_save_name}/{config.model_save_name}_{config.model_epoch}.pt', map_location=torch.device('cpu') if not torch.cuda.is_available() else None)
    sample(config, b)"""

    config = utils.load_config("mdqm9/config/latent/", "10506_latent_allTs_settings.json")  # change this file to run with different settings

    b = cPaiNN(
        n_features=config.n_features,
        score_layers = config.score_layers,
        temp_length=config.temp_length,
        temperatures=config.T,
    )

    b.load_state_dict(torch.load(f'{config.model_save_path}/{config.model_save_name}_{config.model_epoch}_weights.pt', map_location=torch.device('cpu') if not torch.cuda.is_available() else None))
    sample(config, b)
