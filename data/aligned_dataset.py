### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

### This script was modified based on the pix2pixHD official implementation (see license above)
### https://github.com/NVIDIA/pix2pixHD

import os
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.base_dataset import BaseDataset
from data.data_util import *
import torch
import nibabel as nib
import numpy as np
import random

import monai
from monai.data import PersistentDataset, CacheDataset
from monai.metrics import SSIMMetric
from monai import transforms as monai_transforms
from pathlib import Path
import pandas as pd


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        ###
        ### input A (source domain)
        dir_A = '_A'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A, opt.extension))
        assert self.A_paths, 'modality A can not find files with extension ' + opt.extension
        ### input B (target domain)
        ### if you are converting T1w to T2w, please put training T1w scans into train_A and training T2w scans into train_B
        dir_B = '_B'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B, opt.extension))
        assert self.B_paths, 'modality B can not find files with extension ' + opt.extension

        self.dataset_size = len(self.A_paths)
        
        pix_dim = (1,1,-1)
        self.transform = monai_transforms.Compose([
            monai_transforms.LoadImaged(keys=["A", "B"]),
            monai_transforms.EnsureChannelFirstd(keys=["A", "B"]),
            monai_transforms.EnsureTyped(keys=["A", "B"], dtype=torch.float32),
            monai_transforms.Orientationd(keys=["A", "B"], axcodes="RAS"),
            monai_transforms.Spacingd(keys=["A", "B"], pixdim=pix_dim, mode=("bilinear")),
            monai_transforms.CenterSpatialCropd(keys=["A", "B"], roi_size=(256, 256, -1)),
            monai_transforms.SpatialPadd(keys=["A", "B"], spatial_size=(256, 256, -1), mode="constant", constant_values=0),
            monai_transforms.ThresholdIntensityd(keys=["A", "B"], threshold=0.0, above=True, cval=0.0),
            monai_transforms.ScaleIntensityRangePercentilesd(keys=["A", "B"], lower=0, upper=self.opt.norm_perc, b_min=0, b_max=1, clip=True),
        ])
        self.data_dicts = [{'A': a, 'B': b} for a, b in zip(self.A_paths, self.B_paths)]
        # cache_dir = Path('./cache') / f'persistent_cache_{opt.phase}'
        # cache_dir.mkdir(parents=True, exist_ok=True)
        self.monai_ds = CacheDataset(
            data=self.data_dicts,
            transform=self.transform,
            # cache_dir=cache_dir,
            cache_rate=1.0,
            num_workers=4,
        )
        self._calculate_valid_slices()
        
        self.ssim = SSIMMetric(data_range=1.0, spatial_dims=2)
        
    def _calculate_valid_slices(self):
        """Calculate the total number of valid slices for each domain."""
        min_nonzero_pixels = 1000  # Minimum number of non-zero pixels to consider a slice valid
        slice_list = []
        slices_per_volume = {i: [] for i in range(len(self.monai_ds))}
        
        for volume_idx in range(len(self.monai_ds)):
            volume_dict = self.monai_ds[volume_idx]
            data_A = volume_dict["A"]  # shape: (C, H, W, D)
            data_B = volume_dict["B"]
            thresh_A = torch.quantile(data_A[data_A > 0], 0.001)  # 非零值的0.1%分位数
            thresh_B = torch.quantile(data_B[data_B > 0], 0.001)
            
            mask_A = (data_A > thresh_A)
            mask_B = (data_B > thresh_B)
            mask_and = mask_A & mask_B
            
            # valid slices based on non-zero pixels in the AND mask
            valid_z =[z for z in range(data_A.shape[-1]) if mask_and[..., z].sum().item() >= min_nonzero_pixels]
            
            if not valid_z:
                valid_z = [data_A.shape[-1] // 2]
                
            slice_list.extend([(volume_idx, z) for z in valid_z])
            slices_per_volume[volume_idx] = valid_z
        
        self.slices_per_volume = slices_per_volume
        self.slice_list = slice_list
        print(f'[AlignedDataset] {self.opt.phase} set: total volumes: {len(self.monai_ds)}, total valid slices: {len(self.slice_list)}')

    def __getitem__(self, index):
        # index is the slice index in the entire dataset
        volume_idx, slice_idx = self.slice_list[index]
        data_dict = self.monai_ds[volume_idx]
        
        tmpA: torch.Tensor = data_dict["A"]  # shape: (C, H, W, D)
        tmpB: torch.Tensor = data_dict["B"]
        assert tmpA.shape == tmpB.shape, 'paired scans must have the same shape'

        # 3. 先得到前景边界（在 pad 之前做）
        if self.opt.remove_bg: 
            bound = get_bounds(tmpA[0].numpy())          # [x_min, x_max, y_min, y_max, z_min, z_max] # tmpA[0]  shape=(H,W,D)
        else:
            # 用整图边界
            bound = [0, tmpA.shape[0], 0, tmpA.shape[1], 0, tmpA.shape[2]]


        # 6. 采样
        if self.opt.dimension.startswith('2'):
            slice_A = tmpA[..., slice_idx]          # (1, H, W)
            slice_B = tmpB[..., slice_idx]
            thresh_A = torch.quantile(slice_A[slice_A > 0], 0.001)  # 非零值的0.1%分位数
            thresh_B = torch.quantile(slice_B[slice_B > 0], 0.001)
            
            mask_A = (slice_A > thresh_A)
            mask_B = (slice_B > thresh_B)
            mask_and = mask_A & mask_B
            slice_A = slice_A * mask_and
            slice_B = slice_B * mask_and
            roi_A = slice_A[:, bound[0]:bound[1], bound[2]:bound[3]].unsqueeze(0)  # (1, C, H, W)
            roi_B = slice_B[:, bound[0]:bound[1], bound[2]:bound[3]].unsqueeze(0)
            ssim_val = self.ssim(roi_A, roi_B).item()
            
            # print(f"using 2d, shape: {slice_A.shape}, {slice_B.shape}")
            
            # convert to [-1, 1]
            slice_A = (slice_A * 2.0) - 1.0
            slice_B = (slice_B * 2.0) - 1.0

            return {'img_A': slice_A, # (1, H, W)
                    'img_B': slice_B,
                    'ssim': ssim_val,
                    'slice_idx': slice_idx,
                    'A_path': self.A_paths[volume_idx],
                    'B_path': self.B_paths[volume_idx],
                    }
        

        elif self.opt.dimension.startswith('3'):
            # 3D 采样
            x, y, z = self.opt.patch_size
            x_idx = random.randint(bound[0], bound[1] - x)
            y_idx = random.randint(bound[2], bound[3] - y)
            z_idx = random.randint(bound[4], bound[5] - z)
            return {'img_A': tmpA[:, x_idx:x_idx+x, y_idx:y_idx+y, z_idx:z_idx+z],
                    'img_B': tmpB[:, x_idx:x_idx+x, y_idx:y_idx+y, z_idx:z_idx+z],
                    'A_path': self.A_paths[index],
                    'B_path': self.B_paths[index],
                    }

    def name(self):
        return 'Paired/Aligned Dataset'
    
    @property
    def num_volumes(self):
        return len(self.A_paths)
    
    @property
    def num_slices(self):
        return len(self.slice_list)

    def __len__(self):
        return len(self.slice_list)


def test():
    # /public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/K2E2_pred/CUT_monai_K2E2/test_latest/K_060_t1_skullstripped_real_A.nii.gz
    # /public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/K2E2_pred/CUT_monai_K2E2/test_latest/K_060_t1_skullstripped_fake_B.nii.gz
    # cal ssim for each slice
    transform = monai_transforms.Compose([
            monai_transforms.LoadImaged(keys=["A", "B"]),
            monai_transforms.EnsureChannelFirstd(keys=["A", "B"]),
            monai_transforms.EnsureTyped(keys=["A", "B"], dtype=torch.float32),
            monai_transforms.Orientationd(keys=["A", "B"], axcodes="RAS"),
            monai_transforms.Spacingd(keys=["A", "B"], pixdim=(1.0, 1.0, -1), mode=("bilinear")),
            monai_transforms.CenterSpatialCropd(keys=["A", "B"], roi_size=(256, 256, -1)),
            monai_transforms.SpatialPadd(keys=["A", "B"], spatial_size=(256, 256, -1), mode="constant", constant_values=0),
            monai_transforms.ThresholdIntensityd(keys=["A", "B"], threshold=0.0, above=True, cval=0.0),
            monai_transforms.ScaleIntensityRangePercentilesd(keys=["A", "B"], lower=0, upper=99.95, b_min=0, b_max=1, clip=True),
        ])
    
    path_A = "/public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/K2E2_pred/CUT_monai_K2E2/test_latest/K_060_t1_skullstripped_real_A.nii.gz"
    path_B = "/public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/K2E2_pred/CUT_monai_K2E2/test_latest/K_060_t1_skullstripped_fake_B.nii.gz"
    
    data = transform({"A": path_A, "B": path_B})
    data_A: torch.Tensor = data["A"]  # shape: (1, X, Y, Z)
    data_B: torch.Tensor = data["B"]
    z = data_A.shape[-1]
    bound = get_bounds(data_A.numpy())
    print(f"bound: {bound}")
    for slice_idx in range(z):
        slice_A = data_A[..., slice_idx] # ()
        slice_B = data_B[..., slice_idx]
        roi_A = slice_A[...,bound[0]:bound[1], bound[2]:bound[3]].unsqueeze(0) # (1, 1, H, W)
        roi_B = slice_B[...,bound[0]:bound[1], bound[2]:bound[3]].unsqueeze(0)
        mask_A = (roi_A > 1e-4)
        mask_B = (roi_B > 1e-4)
        mask_and = mask_A & mask_B
        ratio = mask_and.sum().item() / max(mask_A.sum().item(), mask_B.sum().item(), 1)
        ssim_val = SSIMMetric(data_range=1.0, spatial_dims=2)(roi_A, roi_B).item()
        print(f"slice {slice_idx}: ssim={ssim_val:.4f}, overlap ratio={ratio:.4f}")
        
    
if __name__ == '__main__':
    from options.train_options import TrainOptions
    opt = TrainOptions().parse()
    opt.dimension = '2D'
    opt.phase = 'test'
    opt.dataroot = '/public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/ptnet2d/'
    print(opt)
    dataset = AlignedDataset()
    dataset.initialize(opt)
    for i in range(100):
        data = dataset[0]
        print(i, data['A_path'], data['B_path'], data['ssim'], data['slice_idx'])
    # visualize one sample
    
    # test()
