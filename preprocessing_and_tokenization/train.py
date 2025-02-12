import argparse
import os
import re
import random
import sys
import time
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import wandb
from generative.networks.nets import VQVAE 
from sklearn.model_selection import train_test_split
from mrdataset import VolumeDataModule

def alphanum_key(s):
    """
    Helper for natural sorting. Splits string into a list of strings and numbers.
    """
    return [int(text) if text.isdigit() else text for text in re.split('([0-9]+)', s)]

def get_step(filename):
    """
    Extract the training step number from a checkpoint filename.
    """
    pattern = r"(\d+)\.pth"
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    else:
        return -1

def set_seed(seed):
    """
    Set seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """
    Load the YAML configuration file.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_model(config):
    """
    Main training loop. Reads all parameters from the provided config.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.set_float32_matmul_precision('high')

    # Setting reproducibility seed
    set_seed(config.get('seed', 42))

    paths_config = config.get('paths', {})
    base_tocho = paths_config.get('tocho')
    proj_name = paths_config.get('proj')
    current_subdir = paths_config.get('current')

    proj_path = os.path.join(base_tocho, proj_name)
    checkpoint_dir = os.path.join(proj_path, current_subdir)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # --- WandB configuration ---
    wandb_config = config.get('wandb', {})
    wandb_project = wandb_config.get('project', 'default_project')
    wandb_entity = wandb_config.get('entity', None)
    wandb.init(project=wandb_project, entity=wandb_entity)

    # --- Data configuration ---
    data_config = config.get('data', {})
    mri_csv_path = data_config.get('mri_csv_path')
    if mri_csv_path is None or not os.path.exists(mri_csv_path):
        print("MRI CSV path is not provided or does not exist in the config.")
        sys.exit(1)
    mr_df = pd.read_csv(mri_csv_path)

    # Create a train/validation split
    test_size = data_config.get('test_size', 0.2)
    train_df, val_df = train_test_split(mr_df, test_size=test_size, shuffle=True)

    # Save the validation set for future inference
    val_csv_path = os.path.join(checkpoint_dir, 'val_dataset.csv')
    val_df.to_csv(val_csv_path, index=False)

    # DataModule parameters (adjust as needed)
    patch_cat   = data_config.get('patch_cat', 64)
    batch_size  = data_config.get('batch_size', 20)
    token_limit = data_config.get('token_limit', 1600)
    gpus        = data_config.get('gpus', 1)
    num_workers = data_config.get('num_workers', 8)

    data_module = VolumeDataModule(train_df, val_df,
                                   patch_cat=patch_cat,
                                   batch_size=batch_size,
                                   token_limit=token_limit,
                                   gpus=gpus,
                                   num_workers=num_workers)
    data_module.setup()

    # --- Model configuration ---
    model_config = config.get('model', {})
    vqvae_model = VQVAE(
        spatial_dims          = model_config.get("spatial_dims", 3),
        in_channels           = model_config.get("in_channels", 1),
        out_channels          = model_config.get("out_channels", 1),
        num_res_layers        = model_config.get("num_res_layers", 2),
        downsample_parameters = model_config.get("downsample_parameters", []),
        upsample_parameters   = model_config.get("upsample_parameters", []),
        num_channels          = model_config.get("num_channels", 128),
        num_res_channels      = model_config.get("num_res_channels", 32),
        num_embeddings        = model_config.get("num_embeddings", 512),
        embedding_dim         = model_config.get("embedding_dim", 64)
    )
    vqvae_model = vqvae_model.to(device)

    # --- Optimizer configuration ---
    optimizer_config = config.get('optimizer', {})
    lr = optimizer_config.get('lr', 1e-4)
    optimizer = torch.optim.Adam(vqvae_model.parameters(), lr=lr)
    l1_loss = nn.L1Loss()

    # --- Training parameters ---
    train_config = config.get('train', {})
    n_epochs      = train_config.get('n_epochs', 10)
    train_interval = train_config.get('train_interval', 200)
    val_interval   = train_config.get('val_interval', 200)

    # --- Checkpoint load ---
    existing_ckpts = sorted(
        [fname for fname in os.listdir(checkpoint_dir) if fname.endswith(".pth")],
        key=alphanum_key
    )
    total_step = 0
    if existing_ckpts:
        last_ckpt = existing_ckpts[-1]
        ckpt_path = os.path.join(checkpoint_dir, last_ckpt)
        vqvae_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        total_step = get_step(last_ckpt) + 1


    total_start = time.time()

    # --- Training loop ---
    for epoch in range(n_epochs):
        epoch_start = time.time()
        vqvae_model.train()

        # Reset interval loss accumulators for this epoch
        interval_train_recons_loss = 0.0
        interval_train_quant_loss  = 0.0

        train_loader = data_module.train_dataloader()
        val_loader   = data_module.val_dataloader()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"), start=1):
            images = batch.to(device)
            # Permute the image axes in a random order
            axes = [2, 3, 4]
            random.shuffle(axes)
            images_t = images.permute(0, 1, *axes)

            optimizer.zero_grad(set_to_none=True)
            reconstruction, quantization_loss = vqvae_model(images=images_t)
            recons_loss = l1_loss(reconstruction.float(), images_t.float())
            loss = recons_loss + quantization_loss
            loss.backward()
            optimizer.step()

            interval_train_recons_loss += recons_loss.item()
            interval_train_quant_loss  += quantization_loss.detach().item()

            # Log training losses every train_interval mini-batches
            if (total_step + 1) % train_interval == 0:
                avg_train_recons = interval_train_recons_loss / train_interval
                avg_train_quant  = interval_train_quant_loss / train_interval

                wandb.log({
                    'avg_train_recon_loss': avg_train_recons,
                    'avg_train_quant_loss': avg_train_quant
                })

                elapsed = time.time() - epoch_start
        
                print(f"Step {total_step}: Elapsed {int(elapsed//60)} min ({int(elapsed)} sec)")
                sys.stdout.flush()

                interval_train_recons_loss = 0.0
                interval_train_quant_loss  = 0.0

            # Run validation and save checkpoint every val_interval mini-batches
            if (total_step + 1) % val_interval == 0:
                vqvae_model.eval()
                recon_val_loss = 0.0
                quant_val_loss = 0.0
                val_batches = 0

                # Save current checkpoint
                ckpt_save_path = os.path.join(checkpoint_dir, f"vqvae_model_step{total_step}.pth")
                torch.save(vqvae_model.state_dict(), ckpt_save_path)

                with torch.no_grad():
                    for val_step, val_batch in enumerate(val_loader, start=1):
                        if val_step > 2:  # Limit validation to a few batches
                            break
                        images_val = val_batch.to(device)
                        recon_val, quant_val = vqvae_model(images=images_val)
                        recon_loss_val = l1_loss(recon_val.float(), images_val.float())
                        recon_val_loss += recon_loss_val.item()
                        quant_val_loss += quant_val.detach().item()
                        val_batches += 1

                if val_batches > 0:
                    recon_val_loss /= val_batches
                    quant_val_loss /= val_batches

                wandb.log({
                    'avg_val_recon_loss': recon_val_loss,
                    'avg_val_quant_loss': quant_val_loss
                })

                vqvae_model.train()

            total_step += 1

        epoch_elapsed = time.time() - epoch_start
        print(f"Epoch {epoch+1} completed in {int(epoch_elapsed//60)} min ({int(epoch_elapsed)} sec).")
        sys.stdout.flush()

    total_time = time.time() - total_start
    total_minutes = total_time // 60
    total_hours = total_minutes // 60
    print(f"Training completed in {int(total_minutes)} min ({int(total_hours)} hrs).")
    print("End")
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="Train VQVAE Model")
    parser.add_argument(
        '--config',
        type=str,
        default='/model_run/train_vqvae_params.yaml',
        help='Path to configuration YAML file.'
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train_model(config)

if __name__ == "__main__":
    main()
