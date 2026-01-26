#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, nibabel as nib, numpy as np, torch, lightning as pl
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import Trainer
from options.test_options import TestOptions
from models.models import create_model
from data.data_util import norm_img, patch_slicer, get_bounds


class NiftiPredictDataset(Dataset):
    """仅返回待推理的 3D/2D 扫描与文件名"""
    def __init__(self, opt):
        self.opt = opt
        self.test_dir = os.path.join(opt.dataroot, 'test_A')
        self.flist = [f for f in os.listdir(self.test_dir) if f.endswith(opt.extension)]

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, idx):
        fname = self.flist[idx]
        try:
            img_nii = nib.load(os.path.join(self.test_dir, fname))
        except ValueError:          # 兼容旧版 nibabel
            nib.Nifti1Header.quaternion_threshold = -1e-6
            img_nii = nib.load(os.path.join(self.test_dir, fname))

        scan = np.squeeze(img_nii.get_fdata()).astype(np.float32)
        scan[scan < 0] = 0
        return {'scan': scan, 'fname': fname, 'affine': img_nii.affine, 'header': img_nii.header}


class PTNetPredictor(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.save_dir = os.path.join(opt.dataroot, f'{opt.name}_{opt.whichmodel}')
        os.makedirs(self.save_dir, exist_ok=True)

        # 1. 先建网络（与原脚本一致）
        self.PTNet, _, _ = create_model(opt)

        # 2. 加载 Lightning 保存的 .ckpt
        ckpt_path = os.path.join(opt.checkpoints_dir, opt.name, opt.whichmodel)
        lightning_ckpt = torch.load(ckpt_path, map_location='cpu')

        # 3. 只取 PTNet 的权重
        state = {k.replace('PTNet.', ''): v
                 for k, v in lightning_ckpt['state_dict'].items()
                 if k.startswith('PTNet.')}
        self.PTNet.load_state_dict(state, strict=True)
        self.PTNet.eval().requires_grad_(False)

    def forward(self, x):          # 仅用于推理
        return self.PTNet(x)

    def predict_step(self, batch, batch_idx):
        scan, fname = batch['scan'], batch['fname']
        affine, header = batch['affine'], batch['header']

        # 统一原脚本预处理
        scan_norm = norm_img(scan, self.opt.norm_perc)
        pred = np.zeros_like(scan_norm)
        norm_cnt = np.zeros_like(scan_norm)

        # ===== 3D patch-wise =====
        if self.opt.dimension.startswith('3'):
            patches, _, idx = patch_slicer(
                scan_norm, scan_norm, self.opt.patch_size,
                tuple(d // 2 for d in self.opt.patch_size),
                remove_bg=self.opt.remove_bg, test=True, ori_path=None
            )
            for patch, coord in zip(patches, idx):
                tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)
                if self.device.type != 'cpu':
                    tensor = tensor.cuda()
                with torch.no_grad():
                    out = self.forward(tensor)
                out = out.squeeze().cpu().numpy()
                sl = tuple(slice(c[0], c[1]) for c in coord.reshape(3, 2))
                pred[sl] += out
                norm_cnt[sl] += 1
            pred[norm_cnt > 0] /= norm_cnt[norm_cnt > 0]

        # ===== 2D slice-wise =====
        elif self.opt.dimension.startswith('2'):
            if self.opt.remove_bg:
                b = get_bounds(scan_norm)
            else:
                b = [0, scan_norm.shape[0], 0, scan_norm.shape[1], 0, scan_norm.shape[2]]
            for z in range(b[4], b[5]):
                sl = scan_norm[:, :, z]
                tensor = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0)
                if self.device.type != 'cpu':
                    tensor = tensor.cuda()
                with torch.no_grad():
                    out = self.forward(tensor)
                pred[:, :, z] = out.squeeze().cpu().numpy()

        # 后处理与原脚本一致
        pred = (pred + 1) / 2
        pred_nii = nib.Nifti1Image(pred, affine, header)
        out_name = fname.replace(self.opt.extension,
                                 f'_PTNetSynth{self.opt.extension}')
        nib.save(pred_nii, os.path.join(self.save_dir, out_name))
        return {'saved': out_name}

def identity_collate(batch):
    """batch_size=1 时直接返回单个元素"""
    assert len(batch) == 1
    return batch[0]

# -------------------- main --------------------
def main():
    opt = TestOptions().parse(save=False)
    dataset = NiftiPredictDataset(opt)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=4, pin_memory=True,collate_fn=identity_collate,)

    model = PTNetPredictor(opt)

    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,                    # 单卡即可；如需多卡推理可改
        precision='16-mixed',         # ← 关键：FP16 混合精度
        enable_progress_bar=True,        
        logger=False,                 # 推理无需日志
    )
    trainer.predict(model, dataloaders=loader)


if __name__ == '__main__':
    main()