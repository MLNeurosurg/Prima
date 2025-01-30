from pathlib import Path
import random
import time
import pandas as pd
import numpy as np

import sys
import os
import yaml
import re
from tqdm import tqdm

from generative.networks.nets import VQVAE
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from torchmr.data.mrdataset import VolumeDataModule
import torch
import torch.nn as nn
import wandb

def alphanum_key(s):
    # Split the string into a list of numbers and non-number parts
    return [int(text) if text.isdigit() else text for text in re.split('([0-9]+)', s)]

def get_step(fn):
    pattern = r"(\d+)\.pth"
    match = re.search(pattern, fn)
    if match:
        step_number = int(match.group(1))
        return step_number
    else:
        return -1

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# wandb_logger = WandbLogger(project='<project_name>', log_model = 'all')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.set_float32_matmul_precision('high')
TOCHO_PATH = "/nfs/turbo/umms-tocho-snr"
PATH_PROJ  = f"{TOCHO_PATH}/exp/eharakej"
PATH_CUR = f"{PATH_PROJ}/TOKEN-MODEL-8-32-32_ablation_4096_2024-10-02-1322PM"

def main(params, data_module):
    print(f"Using {device}")
    sys.stdout.flush()
    # get dataloaders
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    vqvae_model = VQVAE(
        spatial_dims   = params["spatial_dims"],
        in_channels    = params["in_channels"],
        out_channels   = params["out_channels"],
        num_res_layers = params["num_res_layers"],
        downsample_parameters = params["downsample_parameters"],
        upsample_parameters = params["upsample_parameters"],
        num_channels          = params["num_channels"],
        num_res_channels      = params["num_res_channels"],
        num_embeddings = params["num_embeddings"],
        embedding_dim  = params["embedding_dim"],
    )
    vqvae_model = vqvae_model.to(device)

    optimizer = torch.optim.Adam(params=vqvae_model.parameters(), lr=1e-4)
    l1_loss = nn.L1Loss()
    
    n_epochs       = params["n_epochs"]
    val_interval   = 200 # every 200 mini-batches save model and statistics
    train_interval = 200

    interval_train_recons_loss_list = []
    interval_valid_recons_loss_list = []
    interval_train_quant_loss_list = []
    interval_valid_quant_loss_list = []

    # save model checkpoints
    Path(PATH_CUR).mkdir(parents=True, exist_ok=True)
    items = [x for x in os.listdir(PATH_CUR) if x.endswith(".pth")]
    if len(items) > 0:
        checkpoints = sorted(items, key=alphanum_key)
        CHECK_PATH = os.path.join(PATH_CUR, checkpoints[-1])
        vqvae_model.load_state_dict(torch.load(CHECK_PATH, map_location=torch.device(device)))
        total_step = get_step(checkpoints[-1]) + 1
    else:
        total_step = 0

    #initialize wandb library
    wandb.init(project='<project name>', entity = '<username>')
                
    total_start = time.time()
    #total_step = get_step(checkpoints[-1]) + 1
    #total_step = 0

    for epoch in tqdm(range(n_epochs)):
        start_epoch_time = time.time()
        s = start_epoch_time
        
        vqvae_model.train()

        interval_train_recons_loss = 0
        interval_train_quant_loss = 0

        for step, batch in tqdm(enumerate(train_loader)):
            
            images = batch.to(device)
            axes = [2, 3, 4]  
            random.shuffle(axes)
            images_t = images.permute(0, 1, *axes)
            optimizer.zero_grad(set_to_none=True)

            # model outputs reconstruction and the quantization error
            reconstruction, quantization_loss = vqvae_model(images=images_t)
            
            recons_loss = l1_loss(reconstruction.float(), images_t.float())

            loss = recons_loss + quantization_loss

            loss.backward()
            optimizer.step()

            interval_train_recons_loss += recons_loss.item()
            interval_train_quant_loss += quantization_loss.detach().item()

            # save training loss every 1% of training mini-batches (analagous to every 1/100 epochs)
            if (total_step + 1) % train_interval == 0:
                avg_train_recons_loss = interval_train_recons_loss / train_interval
                interval_train_recons_loss_list.append(avg_train_recons_loss)
                avg_train_quant_loss = interval_train_quant_loss/train_interval
                interval_train_quant_loss_list.append(avg_train_quant_loss)
                wandb.log({'avg_train_recon_loss': avg_train_recons_loss})
                wandb.log({'avg_train_quant_loss': avg_train_quant_loss})
                interval_train_recons_loss = 0
                interval_train_quant_loss = 0

                try:               
                    recon_train_file_path = f"{PATH_CUR}/interval_train_recons_loss_list.npy" 
                    np.save(recon_train_file_path, np.array(interval_train_recons_loss_list))
                    quant_train_file_path = f"{PATH_CUR}/interval_train_quant_loss_list.npy"
                    np.save(quant_train_file_path, np.array(interval_train_quant_loss_list))    
                except Exception as e:
                    print(f"Couldn't save data at step {total_step}.")
                    print(e)
                    sys.stdout.flush()
                                
                # Time; GPU usage
                end     = time.time()
                elapsed = end - s 
                s       = end
                gpu_usage = 50
                print(f"Training step {total_step}. Elapsed: {elapsed//60} minutes ({int(elapsed)} seconds). GPU Usage: {gpu_usage}%")
                sys.stdout.flush()
            
            # save model, validation, every 1,000 mini-batches
            if (total_step + 1) % val_interval == 0:
                            
                # Save model & train/valid loss stats
                vqvae_model.eval()
                recon_val_loss = 0
                quant_val_loss = 0

                with torch.no_grad():
                
                    torch.save(
                        vqvae_model.state_dict(),
                        os.path.join(PATH_CUR, f"vqvae_model_step{total_step}.pth"),
                    )
                
                    k = 0
                    for val_step, batch in enumerate(val_loader, start=1):
                        k += 1
                        if k == 3:
                            break
                        images = batch.to(device)
                        images_t = images

                        reconstruction, quantization_loss = vqvae_model(images=images_t)
                        recons_loss = l1_loss(reconstruction.float(), images_t.float())
                        recon_val_loss += recons_loss.item()
                        quant_val_loss += quantization_loss.detach().item()

                recon_val_loss /= val_step
                quant_val_loss /= val_step
                interval_valid_recons_loss_list.append(recon_val_loss)
                interval_valid_quant_loss_list.append(quant_val_loss)
                wandb.log({'avg_val_recon_loss': recon_val_loss})
                wandb.log({'avg_val_quant_loss': quant_val_loss})
                
                try:               
                    recon_valid_file_path = f"{PATH_CUR}/interval_valid_recons_loss_list.npy"                 
                    np.save(recon_valid_file_path, np.array(interval_valid_recons_loss_list))
                    quant_valid_file_path = f"{PATH_CUR}/interval_valid_quant_loss_list.npy"
                    np.save(quant_valid_file_path, np.array(interval_valid_quant_loss_list))

                except Exception as e:
                    print(f"Couldn't save data at step {total_step}.")
                    print(e)
                    sys.stdout.flush()
                
                # Reset training again
                vqvae_model.train()
        
            total_step +=1
        
        elapsed = time.time() - start_epoch_time
        print(f"Epoch {epoch}. Elapsed: {elapsed//60} minutes ({int(elapsed)} seconds).")
        sys.stdout.flush()
        
    total_time = time.time() - total_start
    total_min = total_time//60
    total_hr  = total_min//60
    print(f"train completed, total time: {total_min} minutes {total_hr} hours.")
    print("End")
    sys.stdout.flush()

    # Save
    try:
        recon_train_file_path = f"{PATH_CUR}/interval_train_recons_loss_list.npy"                 
        np.save(recon_train_file_path, np.array(interval_train_recons_loss_list))
        quant_train_file_path = f"{PATH_CUR}/interval_train_quant_loss_list.npy"
        np.save(quant_train_file_path, np.array(interval_train_quant_loss_list))
        
        recon_valid_file_path = f"{PATH_CUR}/interval_valid_recons_loss_list.npy"                 
        np.save(recon_valid_file_path, np.array(interval_valid_recons_loss_list))
        quant_valid_file_path = f"{PATH_CUR}/interval_valid_quant_loss_list.npy"
        np.save(quant_valid_file_path, np.array(interval_valid_quant_loss_list))

    except Exception as e:
        print(f"Couldn't save data at end.")
        print(e)
        sys.stdout.flush()
    


if __name__ == "__main__":
   
    set_seed(42)

    # read in config file with VQVAE model parameters
    with open('/model_run/train_vqvae_params.yaml', 'r') as f:
        params=yaml.safe_load(f)

    patch_cat = params['patch_size'][1]
    
    mr_df = pd.read_csv('/path/to/mri_data')

    # create a train test split
    train_data, val_data = train_test_split(mr_df, test_size=0.2, shuffle= True)

    # store validation set for future inferencing
    val_data.to_csv(f'{PATH_CUR}/val_dataset.csv')

    data_module = VolumeDataModule(train_data,
                                   val_data,
                                   patch_cat=patch_cat,
                                   batch_size=20,
                                   token_limit=1600,
                                   gpus=1,
                                   num_workers=8)
    
    main(params, data_module=data_module)

