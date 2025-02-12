import os

import json

import time
import logging
import concurrent.futures
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import pydicom as pyd
import SimpleITK as sitk
from glob import glob
from monai.transforms import (
    Compose,
    Spacingd,
    EnsureChannelFirstd,
    Resized,
    ToTensord,
    LoadImage,
    Resize,
)
import monai.transforms as montransform
import nibabel as nib

MAX_SLICETHICKNESS_THRESHOLD = 4
DEFAULT_PATCH_SHAPE = [32, 32, 32]

def load_series_sitk(series_path):
    image = sitk.ReadImage(series_path)
    np_image = sitk.GetArrayFromImage(image)
    return np_image


def percentile_mask(image, mask_threshold=50):
    # this conditional statement checks to calculated sequences, such as e/dACD
    if image.max() < 5:
        mask = image > mask_threshold / 100
    else:
        mask = image > mask_threshold
    return mask


def adjusted_patch_shape(
    image_shape: Tuple[int, int, int],
    patch_shape: Optional[List[int]] = None,
    z_val: int = 4,
) -> Tuple[List[int], Optional[int]]:
    """Adjust the patch shape based on the image shape."""
    if patch_shape is None:
        patch_shape = DEFAULT_PATCH_SHAPE.copy()

    z_idx = None

    for idx, dim_size in enumerate(image_shape):
        if dim_size != 256:
            z_idx = idx
            patch_shape[z_idx] = z_val
            break

    return patch_shape, z_idx


def pad_volume_for_patches(
    volume: Union[np.ndarray, torch.Tensor], patch_size: List[int]
) -> torch.Tensor:
    """Pad the volume so that it can be evenly divided into patches."""
    if isinstance(volume, np.ndarray):
        volume = torch.from_numpy(volume.astype(np.float32))

    pad_sizes = [(ps - s % ps) % ps for s, ps in zip(volume.shape, patch_size)]
    pad = []
    for p in pad_sizes[::-1]:
        pad.extend([p // 2, p - p // 2])

    padded_volume = F.pad(volume, pad)
    return padded_volume


def scale(x):
    max_x = x.max().item()
    if max_x > 0:
        return x / max_x  # or 2^16
    return x


def tokenize_volume(
    volume: Union[np.ndarray, torch.Tensor], mask_perc: int = 50
) -> Tuple[
    List[torch.Tensor],
    List[Tuple[int, int, int]],
    List[float],
    Tuple[int, int, int],
    List[int],
    Optional[int],
]:
    """Chop a volume into patches and collect relevant information."""
    start = time.time()
    img = volume
    patch_size, z_idx = adjusted_patch_shape(img.shape)
    logging.info(f"Patch shape is {patch_size}")
    padded_volume = pad_volume_for_patches(img, patch_size)
    z_patches = padded_volume.shape[0] // patch_size[0]
    y_patches = padded_volume.shape[1] // patch_size[1]
    x_patches = padded_volume.shape[2] // patch_size[2]

    mask_ = percentile_mask(padded_volume, mask_perc)
    scaled_padded_vol = scale(padded_volume)
    patches = []
    coordinates = []
    values_ = []

    for z in range(z_patches):
        for y in range(y_patches):
            for x in range(x_patches):
                z_start = z * patch_size[0]
                y_start = y * patch_size[1]
                x_start = x * patch_size[2]
                patch = scaled_padded_vol[
                    z_start : z_start + patch_size[0],
                    y_start : y_start + patch_size[1],
                    x_start : x_start + patch_size[2],
                ]
                otsu_test = mask_[
                    z_start : z_start + patch_size[0],
                    y_start : y_start + patch_size[1],
                    x_start : x_start + patch_size[2],
                ]
                patches.append(patch)
                coordinates.append((z_start, y_start, x_start))
                values_.append(np.mean(otsu_test.numpy()) * 100)

    elapsed_time = time.time() - start
    logging.info(f"Finished chopping volume into patches in {elapsed_time:.2f} seconds")

    return patches, coordinates, values_, padded_volume.shape, patch_size, z_idx


def resize_tokens_batch(tensor_list, patch_shape):
    # Assuming Resize can handle batch tensors
    resize = Resize(spatial_size=patch_shape)
    batch_tensor = np.stack(tensor_list)  # Stack tensors to create a batch
    resized_batch = resize(batch_tensor)
    return list(resized_batch)
