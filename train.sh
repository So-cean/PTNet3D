python train_lightning.py --name PTNet3D-flash --checkpoints_dir ./checkpoints --dataroot /public_bme2/bme-wangqian2/songhy2024/data/BCP_PTNet3D --niter 500 --niter_decay 500 --batchSize 8 --lr 0.0005 --save_epoch_freq 20 --nThreads 8 --dimension 3D

python train.py \
    --name CUT \
    --checkpoints_dir ./checkpoints \
    --dataroot /public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/ptnet2d/ \
    --dimension 2D \
    --nThreads 8 \
    --batchSize 32 \
    --niter 10000 \
    --niter_decay 10000 \
    --lr 0.0005 \
    --save_epoch_freq 2000


python test.py --name CUT --checkpoints_dir ./checkpoints --dataroot /public_bme2/bme-wangqian2/songhy2024/data/PVWMI/T1w/ptnet2d/ --batchSize 8 --nThreads 8 --whichmodel PTNet_epoch=199.ckpt