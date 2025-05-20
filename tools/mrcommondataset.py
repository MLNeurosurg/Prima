import sys
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from VolUtils import tokenize_volume, resize_tokens_batch


class MrVoxelDataset(Dataset):

    def __init__(self, series_volumes, transform=None):
        self.series_volumes = series_volumes
        self.transform = transform

    def __len__(self):
        return len(self.series_volumes)

    def __getitem__(self, idx):
        volume = self.series_volumes[idx]

        tokens, _, _, _, patch_shape, z_idx = tokenize_volume(volume,
                                                              mask_perc=50)

        if not tokens:
            sys.exit("No tokens found")

        patch_shape[z_idx] = 8  #upsacling due to vqvae
        tokens = resize_tokens_batch(tokens, patch_shape)
        return torch.stack(tokens)


if __name__ == "__main__":
    # Create sample series volumes (3D tensors)
    sample_volumes = [
        torch.randn(256, 256, 32),
        torch.randn(256, 256, 120),
        torch.randn(256, 256, 40)
    ]
    
    # Initialize dataset
    dataset = MrVoxelDataset(sample_volumes)
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Test a few iterations
    print("\nTesting DataLoader outputs:")
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i+1}:")
        print(f"Batch shape: {batch.shape}")
        print(f"Batch type: {batch.dtype}")


