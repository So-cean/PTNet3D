#!/bin/bash
#SBATCH --job-name=ptnet
#SBATCH --output=./slurm_logs/out_%j.log
#SBATCH --error=./slurm_logs/err_%j.log
#SBATCH --time=5-00:00:00
#SBATCH --partition=bme_a10080g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA10080GBPCIe:1

conda activate kaolin
cd /public_bme2/bme-wangqian2/songhy2024/PTNet3D

python test_2d.py \
    --name ptnet2d_K2E2 \
    --checkpoints_dir ./checkpoints/dist \
    --dataroot /public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/ptnet2d/K2E2_pred/ \
    --dimension 2D \
    --batchSize 8 \
    --nThreads 8 \
    --patch_size 256 256 \
    --whichmodel PTNet_latest.pth

# python test_2d.py \
#     --name ptnet2d_K2I \
#     --checkpoints_dir ./checkpoints \
#     --dataroot /public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/ptnet2d/K2I_pred/ \
#     --dimension 2D \
#     --batchSize 8 \
#     --nThreads 8 \
#     --patch_size 256 256 \
#     --whichmodel PTNet_latest.pth