# convert cyclegan predict results to ptnet dataset format
# cyclegan output format
# ./opt.results
#     - test_latest
#       - SITE*_real_A.nii.gz # source domain (e.g., site A)
#       - _fake_B.nii.gz # harmonized to target domain (e.g., site B)
#       - _real_B.nii.gz # target domain (e.g., site B)
#     - train_latest
#     - val_latest
#
# PTNET format
# ./opt.dataroot 
#     - train_A # your source domain scans
#     - train_B # your target domain scans
#     - test_A # will be used for inference
#     - test_B
#     - val_A
#     - val_B
#     - external
# use symlink to save space

#!/usr/bin/env python3
"""
CycleGAN 预测结果 → PTNet 目录格式（软链接版）
python make_ptnet_dataset.py \
    --results /public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/BCP2CBCP/BCP2CBCP_pred/CUT_monai_BCP2CBCP/ \
    --dataroot /public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/ptnet2d/BCP2CBCP_pred --phase all
"""

'''
python make_ptnet_dataset.py \
    --results /public/home_data/home/songhy2024/data/PVWMI/T1w/k2E-PHILIPS-INGENIA-3.0T/K2E2_pred/T1w_K2E2_npu_resnet_9blocks_patchnce/ \
    --dataroot /public/home_data/home/songhy2024/data/PVWMI/T1w/ptnet2d/K2E2_pred/T1w_K2E2_npu_resnet_9blocks_patchnce/ \
    --phase all \
    --epoch 100
'''

'''
python make_ptnet_dataset.py \
    --results /public/home_data/home/songhy2024/data/PVWMI/T2w/k2E-PHILIPS-INGENIA-3.0T/K2E2_pred/T2w_K2E2_npu_resnet_9blocks_patchnce/ \
    --dataroot /public/home_data/home/songhy2024/data/PVWMI/T2w/ptnet2d/K2E2_pred/T2w_K2E2_npu_resnet_9blocks_patchnce/ \
    --phase all 
'''

'''
python make_ptnet_dataset.py \
    --results /public/home_data/home/songhy2024/data/PVWMI/T1w/k2I-SIEMENS-SKYRA-3.0T/K2I_pred/T1w_K2I_npu_resnet_9blocks_patchnce/ \
    --dataroot /public/home_data/home/songhy2024/data/PVWMI/T1w/ptnet2d/K2I_pred/T1w_K2I_npu_resnet_9blocks_patchnce/ \
    --phase all
'''
import os
import shutil
import argparse
from pathlib import Path

def symlink_force(target, link_name):
    """Create symlink, overwrite if exists."""
    link_name = Path(link_name)
    if link_name.is_symlink() or link_name.exists():
        link_name.unlink()
    link_name.symlink_to(target)

def copy_force(source_file, target_file):
    """Copy file, overwrite if exists."""
    target_file = Path(target_file)
    # 如果目标文件已存在，先删除
    if target_file.exists():
        target_file.unlink()
    # 使用shutil.copy2复制文件并保留元数据
    shutil.copy2(source_file, target_file)
    
def main(args):
    results = Path(args.results)
    dataroot = Path(args.dataroot)
    phase   = args.phase          # test / train / val / all

    # 支持 all，按顺序处理 train, val, test（与常见流程一致）
    phases = ['train', 'val', 'test'] if phase == 'all' else [phase]

    # external 目录统一放在 dataroot 下
    external = dataroot / 'external'
    external.mkdir(parents=True, exist_ok=True)

    for ph in phases:
        if args.epoch is not None:
            latest = f'{ph}_{args.epoch}'
        else:
            latest = f'{ph}_latest'
        src_dir = results / latest
        if not src_dir.exists():
            print(f'Warning: {src_dir} not found, skipping {ph}')
            continue

        # PTNet 子目录（按 phase 区分）
        out_A = dataroot / f'{ph}_A'
        out_B = dataroot / f'{ph}_B'
        for d in (out_A, out_B):
            d.mkdir(parents=True, exist_ok=True)

        # 遍历 CycleGAN 输出
        for f in src_dir.glob('*_real_A.nii.gz'):
            pid = f.with_suffix('').stem
            pid = pid.replace('_real_A', '')

            real_A = Path(src_dir) / (pid + '_real_A.nii.gz')
            fake_B = Path(src_dir) / (pid + '_fake_B.nii.gz')

            if not (real_A.exists() and fake_B.exists()):
                print(f'Skip {pid} in {ph}: missing fake_B or real_B')
                continue

            # 保留原始后缀（例如: PID_real_A.nii.gz / PID_fake_B.nii.gz）
            link_A = out_A / (pid + '_fake_B.nii.gz')   # out_A 里放 fake_B 的链接（保持原名）
            link_B = out_B / (pid + '_real_A.nii.gz')   # out_B 里放 real_A 的链接（保持原名）
            
            copy_force(real_A.resolve(), link_B)
            copy_force(fake_B.resolve(), link_A)
            print(f'[{ph}] Linked {pid}  ->  {link_A}  |  {link_B}')

        # external：把该 phase 中的 real_B 聚合到 dataroot/external（后者会覆盖同名链接）
        for f in src_dir.glob('*_real_B.nii.gz'):
            pid = f.with_suffix('').stem
            pid = pid.replace('_real_B', '')

            real_B = Path(src_dir) / (pid + '_real_B.nii.gz')

            if not real_B.exists():
                print(f'Skip {pid} in {ph}: missing real_B')
                continue

            link_B = external / (pid + '_real_B.nii.gz')
            copy_force(real_B.resolve(), link_B)
            print(f'[{ph}] Linked external {pid}  ->  {link_B}')
    
    print('✅ Done! PTNet format ready at', dataroot.absolute())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CycleGAN results -> PTNet dataset (symlink)')
    parser.add_argument('--results', required=True, help='CycleGAN results folder')
    parser.add_argument('--dataroot', required=True, help='Output PTNet dataset folder')
    parser.add_argument('--phase', default='test', choices=['test', 'train', 'val', 'all'])
    parser.add_argument('--epoch', type=int, default=None, help='如果指定，使用该 epoch 的结果（例如 epoch_100）而不是 latest')
    args = parser.parse_args()
    main(args)
