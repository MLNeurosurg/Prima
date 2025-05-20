import sys
import time

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from monai.data import Dataset
from tools.VolUtils import tokenize_volume, \
                            load_series_sitk, \
                            resize_tokens_batch


class VolumeDataset(Dataset):

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        volume_path = self.data.iloc[idx]['series_path']
        volume = load_series_sitk(volume_path)

        tokens, _, _, _, patch_shape, z_idx = tokenize_volume(volume,
                                                              mask_perc=50)

        if not tokens:
            # print(f"No tokens found for {volume_path}")
            sys.exit(1)

        patch_shape[z_idx] = 8  #upsacling due to vqvae
        tokens = resize_tokens_batch(tokens, patch_shape)
        return tokens, patch_shape


class ConcatDataset(Dataset):

    def __init__(self, dataset, batch_size=32, token_limit=2048):
        self.dataset = dataset
        self.batch_size = batch_size
        self.token_limit = token_limit

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, idx):
        start = time.time()
        all_tokens = []
        shape_buckets = {}

        # Load tokens from a batch of series
        for i in range(self.batch_size):
            series_idx = idx * self.batch_size + i
            tokens, patch_shape = self.dataset[series_idx]
            shape_str = f'{patch_shape[0]}x{patch_shape[1]}x{patch_shape[2]}'

            if shape_str not in shape_buckets:
                shape_buckets[shape_str] = []

            shape_buckets[shape_str].extend(tokens)

        # Stack tokens in each bucket
        for key in shape_buckets:
            shape_buckets[key] = np.stack(shape_buckets[key])

        # Find the bucket with the maximum number of tokens or randomly select
        max_bucket_key = max(shape_buckets,
                             key=lambda k: shape_buckets[k].shape[0])
        selected_tokens = shape_buckets[max_bucket_key]
        selected_shape = [int(dim) for dim in max_bucket_key.split('x')]

        # Pad tokens if they are less than the limit
        if selected_tokens.shape[0] < self.token_limit:
            pad_size = self.token_limit - selected_tokens.shape[0]
            pad_token = np.zeros([pad_size] + selected_shape,
                                 dtype=selected_tokens.dtype)
            selected_tokens = np.concatenate((selected_tokens, pad_token),
                                             axis=0)

        # Ensure the selected_tokens has exactly token_limit tokens
        selected_tokens = selected_tokens[:self.token_limit]

        end = time.time()
        print(f"Time taken to collate the batch: {end - start}")
        return torch.tensor(selected_tokens, dtype=torch.float32)


def custom_collate_fn(batch):
    batch = [torch.unsqueeze(i, 1) for i in batch]
    return batch[0]


class VolumeDataModule(pl.LightningDataModule):
    '''
    This is custom datamodule that deals with varied mr dataset shapes
    '''

    def __init__(self,
                 train_data,
                 val_data,
                 batch_size=32,
                 token_limit=2048,
                 gpus=1,
                 num_workers=8):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.token_limit = token_limit
        self.num_workers = num_workers
        self.inner_batch = gpus

    def setup(self, stage=None):
        self.train_dataset = VolumeDataset(self.train_data)
        self.val_dataset = VolumeDataset(self.val_data)

        self.train_concat_dataset = ConcatDataset(self.train_dataset,
                                                  batch_size=self.batch_size,
                                                  token_limit=self.token_limit)
        self.val_concat_dataset = ConcatDataset(self.val_dataset,
                                                batch_size=self.batch_size,
                                                token_limit=self.token_limit)

    def train_dataloader(self):
        return DataLoader(self.train_concat_dataset,
                          batch_size=self.inner_batch,
                          shuffle=True,
                          collate_fn=custom_collate_fn,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_concat_dataset,
                          batch_size=self.inner_batch,
                          collate_fn=custom_collate_fn,
                          num_workers=self.num_workers)


if __name__ == '__main__':
    # volume_path = ''
    # volume = load_series_sitk(volume_path)
    # print(volume.shape)
    # tokens, pad_vol_shape, patch_size, z_idx = chop_up_volume_into_patches_modified(volume, mask_perc=50)
    # print(len(tokens))
    # print(pad_vol_shape)

    breakpoint()
