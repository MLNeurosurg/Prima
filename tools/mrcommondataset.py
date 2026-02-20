import sys
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .VolUtils import tokenize_volume, resize_tokens_batch
import scipy.ndimage as ndi
try:
    import SimpleITK as sitk
except ImportError:
    sitk = None


class MrVoxelDataset(Dataset):

    def __init__(self, series_volumes, transform=None):
        self.series_volumes = series_volumes
        self.transform = transform

    def __len__(self):
        return len(self.series_volumes)

    def __getitem__(self, idx):
        volume = self.series_volumes[idx]
        # tokenize_volume expects numpy/torch with .shape; convert SimpleITK Image if needed
        if sitk is not None and hasattr(volume, "GetSize"):
            volume = np.asarray(sitk.GetArrayFromImage(volume), dtype=np.float64)
        
        tokens, coords, otsu, pad_shape, patch_shape, z_idx = tokenize_volume(volume,
                                                              mask_perc=50)

        otsu_thresholds = generate_otsu_thresholds(coords, otsu, pad_shape, patch_shape)

        ser_emb_meta = {
                    'PaddedVolShape': pad_shape,
                    'PatchShape': patch_shape,
                    'OtsuThresholds': otsu_thresholds,
                    'emb_index': {idx: coord for idx, coord in enumerate(coords)}
                }

        if not tokens:
            print("No tokens found for a certain sequence in the study.")
            return torch.tensor([]), ser_emb_meta

        patch_shape[z_idx] = 8  #upsacling due to vqvae
        try:
            tokens = resize_tokens_batch(tokens, patch_shape)
        except Exception as e:
            print(f"Error resizing tokens for volume {idx}: {e}")
            return torch.tensor([]), ser_emb_meta
        return torch.stack(tokens), ser_emb_meta


# Generate otsu thresholds dictionary TODO: add filling hole coords upto threhold of 20
def generate_otsu_thresholds(coordinates,
                             otsu,
                             vol_shape,
                             patch_shape,
                             find_holes=True,
                             find_holes_threshold=20,
                             step=1):
    thresholds = list(range(0, 102, step))
    otsu_dict = {}

    for threshold in thresholds:
        threshold_coords = [(idx, coordinates[idx]) for idx, val in \
                            enumerate(otsu) if val>=threshold \
                            and val<threshold+step]
        otsu_dict[threshold] = {}
        otsu_dict[threshold]['OutfillCoords'] = threshold_coords

        highlight_coords = [coord[1] for coord in threshold_coords]
        if find_holes:
            if threshold <= find_holes_threshold:
                filled_coords = find_fully_filled_patches(
                    create_filled_mask(vol_shape,
                                       highlight_coords,
                                       patch_size=patch_shape),
                    patch_size=patch_shape)
                otsu_dict[threshold]['InfillCoords'] = [
                    (z, y, x) for (z, y, x) in filled_coords
                    if (z, y, x) not in highlight_coords
                ]

    return otsu_dict

def create_filled_mask(original_shape,
                       highlight_coords,
                       patch_size=(4, 32, 32)):
    # Step 1: Create the initial 3D mask
    mask = np.zeros(original_shape, dtype=np.bool_)
    for z, y, x in highlight_coords:
        mask[z:z + patch_size[0], y:y + patch_size[1],
             x:x + patch_size[2]] = True

    # Step 2: Fill holes in the mask
    filled_mask = ndi.binary_fill_holes(mask)

    return filled_mask


def find_fully_filled_patches(filled_mask, patch_size=(4, 32, 32)):
    original_shape = filled_mask.shape
    # assert original_shape[0] % patch_size[0] == 0
    # assert original_shape[1] % patch_size[1] == 0
    # assert original_shape[2] % patch_size[2] == 0
    new_coords = []

    # Iterate through the volume, slicing it into patches of the specified size
    for z in range(0, original_shape[0], patch_size[0]):
        for y in range(0, original_shape[1], patch_size[1]):
            for x in range(0, original_shape[2], patch_size[2]):
                # Extract the patch
                patch = filled_mask[z:z + patch_size[0], y:y + patch_size[1],
                                    x:x + patch_size[2]]

                # Check if the patch is fully filled by calculating the average
                # (if the patch is smaller than patch_size at edges, consider it not fully filled)
                if np.average(patch) == 1:
                    new_coords.append((z, y, x))

    return new_coords


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


