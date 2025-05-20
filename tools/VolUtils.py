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

# Constants
MAX_SLICETHICKNESS_THRESHOLD = 4
DEFAULT_PATCH_SHAPE = [32, 32, 32]


def load_series_sitk(series_path: str) -> np.ndarray:
    """
    Load a series using SimpleITK and convert to numpy array.
    
    Args:
        series_path: Path to the series file
        
    Returns:
        Numpy array containing the image data
    """
    try:
        image = sitk.ReadImage(series_path)
        return sitk.GetArrayFromImage(image)
    except Exception as e:
        raise RuntimeError(f"Failed to load series from {series_path}: {str(e)}")


def percentile_mask(image: Union[np.ndarray, torch.Tensor], mask_threshold: int = 50) -> Union[np.ndarray, torch.Tensor]:
    """
    Create a binary mask based on percentile threshold.
    
    Args:
        image: Input image as numpy array or torch tensor
        mask_threshold: Threshold value for masking
        
    Returns:
        Binary mask of same type as input
    """
    try:
        # Handle both numpy arrays and torch tensors
        if isinstance(image, torch.Tensor):
            if image.max() < 5:
                mask = image > (mask_threshold / 100)
            else:
                mask = image > mask_threshold
        else:
            if image.max() < 5:
                mask = image > (mask_threshold / 100)
            else:
                mask = image > mask_threshold
        return mask
    except Exception as e:
        raise RuntimeError(f"Failed to create percentile mask: {str(e)}")


def adjusted_patch_shape(
    image_shape: Tuple[int, int, int],
    patch_shape: Optional[List[int]] = None,
    z_val: int = 4,
) -> Tuple[List[int], Optional[int]]:
    """
    Adjust the patch shape based on the image shape.
    
    Args:
        image_shape: Shape of the input image
        patch_shape: Optional initial patch shape
        z_val: Value to use for z dimension
        
    Returns:
        Tuple of (adjusted patch shape, z dimension index)
    """
    try:
        if patch_shape is None:
            patch_shape = DEFAULT_PATCH_SHAPE.copy()

        z_idx = None

        for idx, dim_size in enumerate(image_shape):
            if dim_size != 256:
                z_idx = idx
                patch_shape[z_idx] = z_val
                break

        return patch_shape, z_idx
    except Exception as e:
        raise RuntimeError(f"Failed to adjust patch shape: {str(e)}")


def pad_volume_for_patches(volume: Union[np.ndarray, torch.Tensor],
                           patch_size: List[int]) -> torch.Tensor:
    """
    Pad the volume so that it can be evenly divided into patches.
    
    Args:
        volume: Input volume as numpy array or torch tensor
        patch_size: Size of patches to create
        
    Returns:
        Padded volume as torch tensor
    """
    try:
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume.astype(np.float32))

        pad_sizes = [(ps - s % ps) % ps for s, ps in zip(volume.shape, patch_size)]
        pad = []
        for p in pad_sizes[::-1]:
            pad.extend([p // 2, p - p // 2])

        padded_volume = F.pad(volume, pad)
        return padded_volume
    except Exception as e:
        raise RuntimeError(f"Failed to pad volume: {str(e)}")


def scale(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Scale input to [0, 1] range.
    
    Args:
        x: Input array or tensor
        
    Returns:
        Scaled array or tensor
    """
    try:
        max_x = x.max().item() if isinstance(x, torch.Tensor) else x.max()
        if max_x > 0:
            return x / max_x
        return x
    except Exception as e:
        raise RuntimeError(f"Failed to scale input: {str(e)}")


def tokenize_volume(
    volume: Union[np.ndarray, torch.Tensor],
    mask_perc: int = 50
) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int]], List[float], Tuple[int, int, int], List[int], Optional[int]]:
    """
    Chop a volume into patches and collect relevant information.
    
    Args:
        volume: Input volume as numpy array or torch tensor
        mask_perc: Percentage threshold for masking
        
    Returns:
        Tuple containing:
        - List of patches
        - List of patch coordinates
        - List of patch values
        - Volume shape
        - Patch shape
        - Z dimension index
    """
    try:
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
                    patch = scaled_padded_vol[z_start:z_start + patch_size[0],
                                            y_start:y_start + patch_size[1],
                                            x_start:x_start + patch_size[2]]
                    otsu_test = mask_[z_start:z_start + patch_size[0],
                                    y_start:y_start + patch_size[1],
                                    x_start:x_start + patch_size[2]]
                    patches.append(patch)
                    coordinates.append((z_start, y_start, x_start))
                    values_.append(np.mean(otsu_test.numpy()) * 100)

        elapsed_time = time.time() - start
        logging.info(f"Finished chopping volume into patches in {elapsed_time:.2f} seconds")

        return patches, coordinates, values_, padded_volume.shape, patch_size, z_idx
    except Exception as e:
        raise RuntimeError(f"Failed to tokenize volume: {str(e)}")


def resize_tokens_batch(tensor_list: List[torch.Tensor], patch_shape: List[int]) -> List[torch.Tensor]:
    """
    Resize a batch of tokens to a target shape.
    
    Args:
        tensor_list: List of input tensors
        patch_shape: Target shape for resizing
        
    Returns:
        List of resized tensors
    """
    try:
        resize = Resize(spatial_size=patch_shape)
        batch_tensor = np.stack(tensor_list)  # Stack tensors to create a batch
        resized_batch = resize(batch_tensor)
        return list(resized_batch)
    except Exception as e:
        raise RuntimeError(f"Failed to resize tokens batch: {str(e)}")
