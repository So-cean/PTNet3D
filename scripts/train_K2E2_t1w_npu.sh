#!/bin/bash
#SBATCH --job-name=K2E2
#SBATCH --output=./slurm_logs/out_%j.log
#SBATCH --error=./slurm_logs/err_%j.log
#SBATCH --time=120:00:00
#SBATCH --partition=bme_npu        # NPU 分区
#SBATCH --nodes=1                  # 只用 1 个节点
#SBATCH --nodelist=bme-npu02
#SBATCH --ntasks=1                 
#SBATCH --cpus-per-task=8         
#SBATCH --exclusive              

conda activate kaolin
cd /public/home_data/home/songhy2024/PTNet3D

torchrun --nproc_per_node=8 train.py \
    --dataroot /public/home_data/home/songhy2024/data/PVWMI/T1w/ptnet2d/K2E2_pred/T1w_K2E2_npu_resnet_9blocks_patchnce \
    --name T1w_K2E2_npu_ptnet \
    --checkpoints_dir ./checkpoints \
    --dimension 2D \
    --nThreads 8 \
    --batchSize 32 \
    --niter 100 \
    --niter_decay 100 \
    --lr 0.0002 \
    --patch_size 256 256 \
    --save_epoch_freq 20 \
    --display_freq 100 \
    --lambda_smooth_l1 1000 \
    --lambda_vgg 10 \
    --lambda_gan 0.5 \
    --display_id 0