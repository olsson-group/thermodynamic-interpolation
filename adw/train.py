import os

import torch
import wandb
import numpy as np

import thermo.losses as losses
import thermo.interpolants as interpolants

import thermo.utils as utils
from thermo.models.simple import FCNetMultiBeta
from data.dataset import ADWMultiTempDataset

import warnings
warnings.filterwarnings("ignore")


def train(config):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if config.use_wandb:
        wandb.init(project=config.project_name, config=config, name=f'{config.model_save_name}')

    # models
    b = FCNetMultiBeta(in_size=1, out_size=1, hidden_size=config.hidden_size, num_layers=config.num_layers).to(device).to(torch.float64)

    # interpolant
    interpolant = interpolants.LinearInterpolant(a=config.a)

    base_data = ADWMultiTempDataset(n_samples=config.n_samples, betas=config.beta0s, traj_path=config.traj_path)
    target_data = ADWMultiTempDataset(n_samples=config.n_samples, betas=config.beta1s, traj_path=config.traj_path)

    train_base_loader, val_base_loader, _ = utils.get_loaders(base_data, config)
    train_target_loader, val_target_loader, _ = utils.get_loaders(target_data, config)

    # define loss etc.
    loss_fn = losses.StandardVelocityLoss(interpolant=interpolant)
    optim = torch.optim.Adam(b.parameters(), lr=config.lr, weight_decay=config.wd)
    lr_s = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=10, verbose=True)

    # main loop
    for epoch in range(config.epochs):
        epoch_train_losses = []
        epoch_val_losses = []

        # train
        b.train()
        for (x0, beta0), (x1, beta1) in zip(train_base_loader, train_target_loader):

            x0, x1, beta0, beta1 = x0.to(device), x1.to(device), beta0.to(device), beta1.to(device)

            optim.zero_grad()
            loss = loss_fn(b, x0, x1, beta0, beta1)

            # "safe" backprop
            if torch.isnan(loss).any():
                if config.use_wandb:
                    wandb.log({"NaN log": 1}, step=epoch)
                else:
                    print("NaN loss")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(b.parameters(), 1)

            optim.step()

            epoch_train_losses.append(loss.detach())

        epoch_train_loss = torch.stack(epoch_train_losses).mean()

        # eval
        b.eval()
        for (x0, beta0), (x1, beta1) in zip(val_base_loader, val_target_loader):
            x0, x1, beta0, beta1 = x0.to(device), x1.to(device), beta0.to(device), beta1.to(device)

            # compute the loss
            loss = loss_fn(b, x0, x1, beta0, beta1)
            epoch_val_losses.append(loss.detach())

        epoch_val_loss = torch.stack(epoch_val_losses).mean()
        lr_s.step(epoch_val_loss)
        
        if config.use_wandb:
            wandb.log({"train_loss": epoch_train_loss, "val_loss": epoch_val_loss}, step=epoch)
        else:
            print(f"Epoch: {epoch:03d} --- Loss: {epoch_train_loss:.8f} --- Val Loss: {epoch_val_loss:.8f}")

        # save models
        if not os.path.exists(os.path.join(config.model_save_path, config.model_save_name)):
            os.makedirs(os.path.join(config.model_save_path, config.model_save_name))
        
        torch.save(b, os.path.join(config.model_save_path, config.model_save_name, f"epoch_{epoch}.pt"))
    
    if config.use_wandb:
        wandb.finish()


def main(config):
    torch.manual_seed(config.seed)
    train(config)


if __name__ == "__main__":
    config = utils.load_config("adw/config", "settings.json")  # edit config file to edit run settings
    main(config)
 