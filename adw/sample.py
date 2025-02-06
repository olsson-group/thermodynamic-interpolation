import os
import wandb

import numpy as np
import torch

import thermo.integrators as integrators
import thermo.utils as utils

from data.dataset import ADWMultiTempDataset
import tqdm


def sample(config, b):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    if config.use_wandb:
        wandb.init(project=f"{config.project_name}, sampling logs", name=config.model_save_name)

    assert len(config.beta0s) == len(config.beta1s) == 1    # make sure sampling is done for only one beta0 and beta1

    dataset = ADWMultiTempDataset(n_samples=config.n_samples, betas=config.beta0s, traj_path=config.traj_path)
    _, _, test_loader = utils.get_loaders(dataset, config)

    integrator = integrators.StandardIntegrator(b=b,
                                                method=config.solver_type,
                                                rtol=config.rtol,
                                                atol=config.atol,
                                                n_step=config.n_step,
                                                return_dlogp=config.return_dlogp)

    initial_samples = []
    samples = []
    dlogps = []

    b.eval()
    for i, (x0s, beta0s) in tqdm.tqdm(enumerate(test_loader), desc='Sampling progress', leave=True, total=len(test_loader)):
        x0s, beta0s = x0s.to(device), beta0s.to(device)
        beta1s = torch.ones_like(beta0s) * config.beta1s[0]

        sample, dlogp = integrator.rollout(x0s, beta0s=beta0s, beta1s=beta1s)

        initial_samples.append(x0s.detach().cpu().numpy())
        samples.append(sample.detach().cpu().numpy())

        if config.return_dlogp:
            dlogps.append(dlogp.detach().cpu().numpy())

        if config.use_wandb:
            wandb.log({"batch_i": i+1, "n_batches": len(test_loader)})

    if not os.path.exists(os.path.join(config.data_save_path, config.model_save_name, f"beta_{config.beta0s[0]}_to_{config.beta1s[0]}")):
        os.makedirs(os.path.join(config.data_save_path, config.model_save_name, f"beta_{config.beta0s[0]}_to_{config.beta1s[0]}"))

    initial_samples = np.array(initial_samples)[:, :, 0].flatten()
    np.save(os.path.join(config.data_save_path, config.model_save_name, f"beta_{config.beta0s[0]}_to_{config.beta1s[0]}", f'initial_samples_epoch_{config.sampling_epoch}.npy'), initial_samples)

    samples = np.array(samples)
    samples_reshaped = []

    for i in range(samples.shape[1]):
        samples_reshaped.append(samples[:, i, :, 0].flatten())
    samples = np.array(samples_reshaped)
    np.save(os.path.join(config.data_save_path, config.model_save_name, f"beta_{config.beta0s[0]}_to_{config.beta1s[0]}", f'samples_epoch_{config.sampling_epoch}.npy'), samples)

    if config.return_dlogp:
        dlogps = np.array(dlogps)
        dlogps_reshaped = []
        for i in range(dlogps.shape[1]):
            dlogps_reshaped.append(dlogps[:, i, :, 0].flatten())
        dlogps = np.array(dlogps_reshaped)
        np.save(os.path.join(config.data_save_path, config.model_save_name, f"beta_{config.beta0s[0]}_to_{config.beta1s[0]}", f'dlogps_epoch_{config.sampling_epoch}.npy'), dlogps)

    print("\nFinished sampling...\n")

    if config.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    config = utils.load_config("adw/config", "settings.json")  # edit config file to edit run
    
    b = torch.load(config.sampling_model, map_location=torch.device('cpu') if not torch.cuda.is_available() else None, weights_only=False)  # # f'{config.model_save_path}/{config.model_save_name}/epoch_{config.sampling_epoch}.pt'
    sample(config, b) #samples, dlogp = sample(config, b)
