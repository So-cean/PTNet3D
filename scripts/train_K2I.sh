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

mkdir -p ./slurm_logs

GPU_LOG="./slurm_logs/gpu_${SLURM_JOB_ID}.log"
(
  while kill -0 $$ 2>/dev/null; do
    {
      stdbuf -oL echo "=== $(date '+%F %T') ==="
      stdbuf -oL -eL nvidia-smi
      sleep 1
    } > "$GPU_LOG" 2>&1
  done
) &


python train.py \
    --name ptnet2d_K2I \
    --checkpoints_dir ./checkpoints \
    --dataroot /public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/ptnet2d/K2I_pred/ \
    --dimension 2D \
    --nThreads 8 \
    --batchSize 64 \
    --niter 2000 \
    --niter_decay 2000 \
    --lr 0.0002 \
    --patch_size 256 256 \
    --save_epoch_freq 500
