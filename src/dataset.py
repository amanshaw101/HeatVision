import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import glob
import os

class LandsatDataset(Dataset):
    def __init__(self, data_dirs, transform=None, return_physics=False):
        """
        Args:
            data_dirs (list): List of directories containing processed .npy files 
                              (e.g., ['data/London/processed', 'data/Dubai/processed'])
            transform (callable, optional): Optional transform to be applied on a sample.
            return_physics (bool): If True, also return NDVI and LST arrays for physics-informed loss.
        """
        self.samples = []
        self.transform = transform
        self.return_physics = return_physics
        
        for d in data_dirs:
            # We look for *B4.npy files as the anchor
            b4_files = glob.glob(os.path.join(d, "*_B4.npy"))
            for b4_path in b4_files:
                # Construct other paths
                base = b4_path.replace("_B4.npy", "")
                sample = {
                    'b4': b4_path,
                    'b5': base + "_B5.npy",
                    'lst': base + "_LST.npy",
                    'weak': base + "_WeakLabel.npy",
                    'ndvi': base + "_NDVI.npy",
                }
                # Check if all exist (NDVI optional for backward compat)
                required = ['b4', 'b5', 'lst', 'weak']
                if all(os.path.exists(sample[k]) for k in required):
                    self.samples.append(sample)
                else:
                    print(f"Warning: Missing files for {base}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths = self.samples[idx]
        
        # Load arrays
        b4 = np.load(paths['b4']).astype(np.float32)
        b5 = np.load(paths['b5']).astype(np.float32)
        lst = np.load(paths['lst']).astype(np.float32)
        weak = np.load(paths['weak']).astype(np.float32)
        
        # Load NDVI if available and requested
        ndvi = None
        if self.return_physics and os.path.exists(paths['ndvi']):
            ndvi = np.load(paths['ndvi']).astype(np.float32)
        
        # Normalize to 0-1 range for stability (Robust)
        def normalize(arr):
            min_val = np.nanpercentile(arr, 2)
            max_val = np.nanpercentile(arr, 98)
            
            if max_val - min_val == 0:
                return np.zeros_like(arr)
            
            out = (arr - min_val) / (max_val - min_val)
            return np.clip(out, 0, 1)

        b4_norm = normalize(b4)
        b5_norm = normalize(b5)
        lst_norm = normalize(lst)
        
        # Stack inputs: Channel 0=Red, 1=NIR, 2=LST
        # Input shape: (3, H, W)
        image = np.stack([b4_norm, b5_norm, lst_norm], axis=0)
        
        # Mask shape: (1, H, W)
        mask = weak[np.newaxis, ...]
        
        # Physics arrays shape: (1, H, W) - un-normalized for physics loss
        if self.return_physics:
            lst_raw = lst[np.newaxis, ...]  # Keep raw LST in Kelvin
            ndvi_raw = ndvi[np.newaxis, ...] if ndvi is not None else np.zeros((1, *lst.shape), dtype=np.float32)
        
        # --- Random Crop ---
        crop_size = 512
        _, h, w = image.shape
        
        # Pad if smaller than crop size
        if h < crop_size or w < crop_size:
            pad_h = max(crop_size - h, 0)
            pad_w = max(crop_size - w, 0)
            image = np.pad(image, ((0,0), (0, pad_h), (0, pad_w)), mode='reflect')
            mask = np.pad(mask, ((0,0), (0, pad_h), (0, pad_w)), mode='reflect')
            if self.return_physics:
                lst_raw = np.pad(lst_raw, ((0,0), (0, pad_h), (0, pad_w)), mode='reflect')
                ndvi_raw = np.pad(ndvi_raw, ((0,0), (0, pad_h), (0, pad_w)), mode='reflect')
            _, h, w = image.shape
            
        # Random coordinates
        top = np.random.randint(0, h - crop_size + 1)
        left = np.random.randint(0, w - crop_size + 1)
        
        image = image[:, top:top+crop_size, left:left+crop_size]
        mask = mask[:, top:top+crop_size, left:left+crop_size]
        if self.return_physics:
            lst_raw = lst_raw[:, top:top+crop_size, left:left+crop_size]
            ndvi_raw = ndvi_raw[:, top:top+crop_size, left:left+crop_size]
        
        # --- Augmentation (Random Flip/Rotate) ---
        # 1. Random Horizontal Flip
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=2)
            mask = np.flip(mask, axis=2)
            if self.return_physics:
                lst_raw = np.flip(lst_raw, axis=2)
                ndvi_raw = np.flip(ndvi_raw, axis=2)
            
        # 2. Random Vertical Flip
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
            if self.return_physics:
                lst_raw = np.flip(lst_raw, axis=1)
                ndvi_raw = np.flip(ndvi_raw, axis=1)
            
        # 3. Random Rotation (0, 90, 180, 270)
        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k, axes=(1, 2))
            mask = np.rot90(mask, k, axes=(1, 2))
            if self.return_physics:
                lst_raw = np.rot90(lst_raw, k, axes=(1, 2))
                ndvi_raw = np.rot90(ndvi_raw, k, axes=(1, 2))
        
        # Make sure arrays are contiguous in memory
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        
        sample = {'image': torch.from_numpy(image), 'mask': torch.from_numpy(mask)}
        
        if self.return_physics:
            sample['lst'] = torch.from_numpy(np.ascontiguousarray(lst_raw))
            sample['ndvi'] = torch.from_numpy(np.ascontiguousarray(ndvi_raw))

        if self.transform:
            sample = self.transform(sample)

        return sample
