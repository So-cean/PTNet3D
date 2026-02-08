torchrun --nproc_per_node=1 test_2d.py \
    --name T1w_K2E2_npu_ptnet \
    --checkpoints_dir ./checkpoints \
    --dataroot /public/home_data/home/songhy2024/data/PVWMI/T1w/ptnet2d/K2E2_pred/T1w_K2E2_npu_resnet_9blocks_patchnce \
    --dimension 2D \
    --batchSize 8 \
    --nThreads 8 \
    --patch_size 256 256 \
    --whichmodel PTNet_latest.pth