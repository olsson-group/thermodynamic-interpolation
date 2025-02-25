import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import wandb
import numpy as np

import torch
from torch_geometric.loader import DataLoader

from thermo.latent import interpolants, losses
from thermo.latent.models import cpainn
from data import mdqm9_latent as mdqm9
from thermo import utils

import warnings
warnings.filterwarnings("ignore")

def trainer(config: argparse.Namespace) -> None:
    """## Train a model.

    ### Args:
        - `config`: configuration namespace.
    """

    if config.use_wandb:  # init wandb
        wandb.init(project=config.project_name, name=config.model_save_name)

    if not os.path.exists(os.path.join(config.model_save_path, config.model_save_name)):
        os.makedirs(os.path.join(config.model_save_path, config.model_save_name))
    
    # set seeds for reproducibility
    np.random.seed(config.seed)  
    torch.manual_seed(config.seed)

    model = cpainn.cPaiNN(
        n_features=config.n_features,
        score_layers = config.score_layers,
        temp_length=config.temp_length,
    )

    min_epoch = 0
    max_epoch = config.n_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    interpolant = interpolants.OneSidedLinearInterpolant()
    loss_fn = losses.OneSidedVelocityLoss(interpolant=interpolant)

    train_dataset = mdqm9.MDQM9MultiTempDataset(
        traj_filename=config.mdqm9_traj_filename, 
        sdf_filename="mdqm9.sdf", 
        traj_path=config.traj_path, 
        sdf_path=config.sdf_path, 
        split='train', 
        Ts=config.T, 
        scale=config.scale_trajs, 
        cutoff=config.cutoff,
        align=config.align,
    )

    # optimizer and lr scheduler
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    lr_s = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=10, verbose=True)

    # training loop
    for epoch in range(min_epoch, max_epoch):
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )
    
        epoch_train_loss = torch.tensor(0.0)
        last_train_loss = torch.tensor(0.0)

        model.train()
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            optim.zero_grad()
            loss = loss_fn(batch, model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optim.step()
            epoch_train_loss += loss.item()

            print(f"Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}", end="\r")

        for batch in train_loader:
            batch = batch.to(device)
            loss = loss_fn(batch, model)
            last_train_loss += loss.item()
    
        epoch_train_loss /= len(train_loader)
        last_train_loss /= len(train_loader)

        lr_s.step(epoch_train_loss)

        if config.use_wandb:
            wandb.log({"train_loss": epoch_train_loss, "last_model_train_loss": last_train_loss}, step=epoch)
        else:
            print(f"Epoch {epoch+1}/{config.n_epochs} - Train Loss: {epoch_train_loss:.4f} - Last Train Loss: {last_train_loss:.4f}")

        # torch.save(model, os.path.join(config.model_save_path, config.model_save_name, f"{config.model_save_name}_{epoch}.pt"))
        torch.save(model.state_dict(), os.path.join(config.model_save_path, config.model_save_name, f"{config.model_save_name}_{epoch}.pt"))
        
    if config.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    config = utils.load_config("mdqm9/config", "settings_latent.json")
    trainer(config)