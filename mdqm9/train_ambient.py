import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import copy
import argparse
import wandb
import numpy as np

import torch
from torch_geometric.loader import DataLoader

from thermo.ambient import interpolants, losses
from thermo.ambient.models import cpainn
from data import mdqm9_ambient as mdqm9
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

    # create/load model
    if config.use_pretrained:
        model = torch.load(f'{config.model_save_path}/{config.model_save_name}/{config.model_save_name}_{config.model_epoch}.pt', map_location=torch.device('cpu') if not torch.cuda.is_available() else None)
        
        min_epoch = int(config.model_epoch)  # sync epochs
        max_epoch = min_epoch + config.n_epochs

    else:
        model = cpainn.cPaiNN(
            n_features=config.n_features,
            score_layers = config.score_layers,
            temp_length=config.temp_length,
        )

        min_epoch = 0
        max_epoch = config.n_epochs

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # select interpolant function
    interpolant = interpolants.LinearInterpolant(
        a=config.a,
        gamma=config.gamma
    )

    # loss function
    loss_fn = losses.StandardVelocityLoss(
        interpolant=interpolant,
        t_distr=config.t_distr
    )

    # data related stuff
    train_dataset0 = mdqm9.MDQM9MultiTempDataset(
        traj_filename=config.mdqm9_traj_filename,
        sdf_filename='mdqm9.sdf',
        traj_path=config.traj_path,
        sdf_path=config.sdf_path,
        split='train',
        Ts=config.T0s,
        scale=config.scale_trajs,
        cutoff=config.cutoff,
    )

    train_dataset1 = mdqm9.MDQM9MultiTempDataset(
        traj_filename=config.mdqm9_traj_filename,
        sdf_filename='mdqm9.sdf',
        traj_path=config.traj_path,
        sdf_path=config.sdf_path,
        split='train',
        Ts=config.T1s,
        scale=config.scale_trajs,
        cutoff=config.cutoff,
    )

    # optimizer and lr scheduler
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    lr_s = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=10, verbose=True)

    # training loop
    for epoch in range(min_epoch, max_epoch):
        train_loader0 = DataLoader(  # reset loaders for each epoch to recombine data at different temperatures
            dataset=train_dataset0, 
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=config.num_workers,
            worker_init_fn = lambda _: np.random.seed(np.random.randint(0, 1000)),
            pin_memory=True,
        )

        train_loader1 = DataLoader(
            dataset=train_dataset1, 
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=config.num_workers,
            worker_init_fn = lambda _: np.random.seed(np.random.randint(0, 1000)),
            pin_memory=True,
        )

        epoch_train_loss = torch.tensor(0.0)
        epoch_best_loss = torch.tensor(float("inf"))
        last_train_loss = torch.tensor(0.0)

        model.train()
        for i, (batch0, batch1) in enumerate(zip(train_loader0, train_loader1)):
            batch0 = batch0.to(model.device)
            batch1 = batch1.to(model.device)

            optim.zero_grad()

            loss = loss_fn(batch0, batch1, model)

            # store best model
            epoch_best_model = copy.deepcopy(model) if loss < epoch_best_loss else epoch_best_model
            epoch_best_loss = loss if loss < epoch_best_loss else epoch_best_loss

            # safe backprop to avoid crashing model
            if torch.isnan(loss).any():
                if config.use_wandb:
                    wandb.log({"NaN log": 1}, step=epoch)
                else:
                    print("NaN loss")
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optim.step()
            epoch_train_loss += loss.item()

            print(f"Batch {i+1}/{len(train_loader0)} - Loss: {loss.item():.4f}", end="\r")
        
        for (batch0, batch1) in zip(train_loader0, train_loader1):  # eval train loader on last model
            batch0 = batch0.to(model.device)
            batch1 = batch1.to(model.device)

            loss = loss_fn(batch0, batch1, model)
            
            last_train_loss += loss.item()

        epoch_train_loss /= len(train_loader0)
        last_train_loss /= len(train_loader0)

        lr_s.step(epoch_train_loss)

        if config.use_wandb:
            wandb.log({"train_loss": epoch_train_loss, "last_model_train_loss": last_train_loss, "epoch_best_loss": epoch_best_loss}, step=epoch)
        else:
            print(f"Epoch {epoch+1}/{config.n_epochs} - Train Loss: {epoch_train_loss:.4f} - Last Train Loss: {last_train_loss:.4f} - Epoch Best Loss: {epoch_best_loss:.4f}")

        #torch.save(model, os.path.join(config.model_save_path, config.model_save_name, f"{config.model_save_name}_{epoch}.pt"))
        #torch.save(epoch_best_model, os.path.join(config.model_save_path, config.model_save_name, f"{config.model_save_name}_best{epoch}.pt"))

        # save state dicts
        torch.save(model.state_dict(), os.path.join(config.model_save_path, config.model_save_name, f"{config.model_save_name}_{epoch}_weights.pt"))
        torch.save(epoch_best_model.state_dict(), os.path.join(config.model_save_path, config.model_save_name, f"{config.model_save_name}_best{epoch}_weights.pt"))


    # save run configuration
    utils.clone_config(config.model_save_path, config.model_save_name, config)    

    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    config = utils.load_config("mdqm9/config/ambient", "00031_settings_no_300.json")  # change this file to run with different settings
    trainer(config)
    