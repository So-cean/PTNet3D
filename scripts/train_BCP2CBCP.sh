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


python train.py \
    --name ptnet2d_BCP2CBCP \
    --checkpoints_dir ./checkpoints \
    --dataroot /public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/ptnet2d/BCP2CBCP_pred/ \
    --dimension 2D \
    --nThreads 8 \
    --batchSize 32 \
    --niter 4000 \
    --niter_decay 4000 \
    --lr 0.0002 \
    --patch_size 256 256 \
    --save_epoch_freq 1000
