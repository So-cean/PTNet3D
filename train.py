### This code is largely borrowed from pix2pixHD pytorch implementation
### https://github.com/NVIDIA/pix2pixHD

import time
import os
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd import Variable
from collections import OrderedDict
import math
from models.models import create_model
import torch.nn as nn
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
import util.util as util
from util.visualizer import Visualizer
from util.image_pool import ImagePool
from models.networks import GANLoss, feature_loss, discriminate
from models.networks import VGGLoss

import importlib
import shutil
if shutil.which("npu-smi") and importlib.util.find_spec("torch_npu") is not None:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu  
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    
import lightning as pl
pl.seed_everything(42, workers=True)


def lcm(a, b): return abs(a * b) / math.gcd(a, b) if a and b else 0

##############################################################################
# DDP Setup
##############################################################################

def setup_ddp():
    """Initialize DDP environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # Single GPU fallback
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='hccl', init_method='env://')
    
    return rank, world_size, local_rank

def cleanup_ddp():
    """Clean up DDP"""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """Check if current process is the main process"""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True

##############################################################################
# Initialize options
##############################################################################

rank, world_size, local_rank = setup_ddp()

opt = TrainOptions().parse(save=is_main_process())  # Only save config on main process
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
start_epoch, epoch_iter = 1, 0
opt.print_freq = lcm(opt.print_freq, opt.batchSize)
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10
ler = opt.lr

# Store DDP info in opt for data loader
opt.rank = rank
opt.world_size = world_size
opt.local_rank = local_rank

##############################################################################
# Initialize dataloader
##############################################################################

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
if is_main_process():
    print('#training batches per GPU = %d' % dataset_size)
    print('#total global batch size = %d' % (opt.batchSize * world_size))
    print('#total training samples = %d' % (dataset_size * opt.batchSize * world_size))
# ========================

##############################################################################
# Initialize networks
##############################################################################

PTNet, D, ext_discriminator = create_model(opt)
device = torch.device(f'cuda:{local_rank}')
PTNet = PTNet.to(device)
D = D.to(device)
ext_discriminator = ext_discriminator.to(device)

# Wrap models with DDP
if world_size > 1:
    PTNet = DDP(PTNet, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    D = DDP(D, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    # ext_discriminator is usually frozen (VGG), no need to wrap if not training
    # If it needs gradients, wrap it too:
    # ext_discriminator = DDP(ext_discriminator, device_ids=[local_rank], output_device=local_rank)

##############################################################################
# Initialize util components
##############################################################################

optimizer_PTNet = torch.optim.Adam(PTNet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0)
optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
fake_pool = ImagePool(0)

CE = nn.CrossEntropyLoss()
criterionGAN = GANLoss(use_lsgan=True, reduction='mean').to(device)
mse = torch.nn.MSELoss(reduction='mean')
mae = torch.nn.L1Loss(reduction='mean')
smooth_l1 = torch.nn.SmoothL1Loss(reduction='mean')

criterionVGG = VGGLoss(gpu_ids=opt.gpu_ids).to(device)  
lambda_smooth_l1 = getattr(opt, 'lambda_smooth_l1')
lambda_vgg = getattr(opt, 'lambda_vgg')  
lambda_gan = getattr(opt, 'lambda_gan')

# training/display parameter
visualizer = Visualizer(opt) if is_main_process() else None

steps_per_epoch = dataset_size  # 每个 epoch 的 step 数 = batch 数
total_steps = (start_epoch - 1) * steps_per_epoch + (epoch_iter // (opt.batchSize * world_size))

# 用于对齐打印频率的 delta 值
display_delta = total_steps % opt.display_freq if opt.display_freq > 0 else 0
print_delta = total_steps % opt.print_freq if opt.print_freq > 0 else 0
save_delta = total_steps % opt.save_latest_freq if opt.save_latest_freq > 0 else 0

##############################################################################
# Training code
##############################################################################

# Get the raw D module for forward (handle DDP wrapper)
D_module = D.module if hasattr(D, 'module') else D

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if is_main_process():
        visualizer.reset()
    
    # 每个 epoch 重置 epoch_iter（样本计数）
    if epoch != start_epoch:
        epoch_iter = 0
    
    # Set epoch for DistributedSampler to shuffle data differently each epoch
    if hasattr(data_loader, 'sampler') and isinstance(data_loader.sampler, torch.utils.data.distributed.DistributedSampler):
        data_loader.sampler.set_epoch(epoch)
        
    
    for i, data in enumerate(dataset, start=epoch_iter):
        iter_start_time = time.time()
        
        # 关键修复：step-based 计数
        total_steps += 1
        local_step = i + 1  # 当前 epoch 内的 step
        
        # 计算全局已处理样本数（用于显示）
        global_samples_seen = (epoch - 1) * dataset_size * opt.batchSize * world_size + local_step * opt.batchSize * world_size
        epoch_iter = local_step * opt.batchSize * world_size

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta if opt.display_freq > 0 else False

        ##############################################################################
        # Forward Pass
        ##############################################################################

        input_image = Variable(data['img_A'].to(device))
        target_image = Variable(data['img_B'].to(device))
        
        # Synthesize and MSE loss
        generated = PTNet(input_image) # [B, C, H, W] or [B, C, D, H, W]
        # loss_mse = mse(generated, target_image)
        loss_smooth_l1 = smooth_l1(generated, target_image)

        loss_vgg = criterionVGG(generated, target_image)
        
        # Fake Detection and Loss
        pred_fake_pool = discriminate(D_module, fake_pool, input_image, generated, use_pool=True)
        loss_D_fake = criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = discriminate(D_module, fake_pool, input_image, target_image)
        loss_D_real = criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        pred_fake = D(torch.cat((input_image, generated), dim=1))
        loss_G_GAN = criterionGAN(pred_fake, True)

        # GAN feature matching loss
        loss_G_GAN_Feat, loss_G_GAN_Feat_ext = feature_loss(opt, target_image, generated, pred_real, pred_fake,
                                                            ext_discriminator)

        # Compute overall loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_G = loss_smooth_l1 * lambda_smooth_l1 + loss_vgg * lambda_vgg + loss_G_GAN * lambda_gan + loss_G_GAN_Feat_ext + loss_G_GAN_Feat
        loss_dict = {
            'SMOOTH_L1': loss_smooth_l1.item(),
            'VGG': loss_vgg.item(),
            'G_GAN': loss_G_GAN.item(),
            'G_GAN_Feat_ext': loss_G_GAN_Feat_ext.item(),
            'G_GAN_Feat': loss_G_GAN_Feat.item(),
            'D_fake': loss_D_fake.item(),
            'D_real': loss_D_real.item()
        }
        ##############################################################################
        # Backward
        ##############################################################################

        # update generator weights
        optimizer_PTNet.zero_grad()
        loss_G.backward()
        optimizer_PTNet.step()

        # update discriminator weights
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        ##############################################################################
        # Display results, print out loss, and save latest model
        ##############################################################################
        
        # 同步 loss 到所有卡（可选，为了打印更准确）
        if world_size > 1:
            for k in loss_dict:
                tensor = torch.tensor(loss_dict[k], device=device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                loss_dict[k] = tensor.item() / world_size
                
        # print out loss (only main process)
        if is_main_process() and total_steps % opt.print_freq == print_delta:
            errors = {k: v if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.print_freq
            # 计算数据加载时间
            t_data = 0  # 简化，或者你可以重新计算
            visualizer.print_current_losses(epoch, epoch_iter, errors, t, t_data)

        # display output images (only main process)
        if is_main_process() and save_fake:
            visuals = OrderedDict([('input_image', util.tensor2im(input_image[0], normalize=True)),
                                    ('target_image', util.tensor2im(target_image[0], normalize=True)),
                                    ('synthesized_image', util.tensor2im(generated[0], normalize=True))])
            visualizer.display_current_results(visuals, epoch, total_steps)

        # save latest model (only main process)
        if is_main_process() and total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            # Save the underlying module (not the DDP wrapper)
            PTNet_state = PTNet.module.state_dict() if hasattr(PTNet, 'module') else PTNet.state_dict()
            D_state = D.module.state_dict() if hasattr(D, 'module') else D.state_dict()
            torch.save(PTNet_state, os.path.join(opt.checkpoints_dir, opt.name, 'PTNet_latest.pth'))
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
            torch.save(D_state, os.path.join(opt.checkpoints_dir, opt.name, 'D_latest.pth'))

        if epoch_iter >= dataset_size:
            break

    # Synchronize all processes at epoch end
    if world_size > 1:
        dist.barrier()

    # end of epoch
    if is_main_process():
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    # save model for this epoch (only main process)
    if is_main_process() and epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        PTNet_state = PTNet.module.state_dict() if hasattr(PTNet, 'module') else PTNet.state_dict()
        D_state = D.module.state_dict() if hasattr(D, 'module') else D.state_dict()
        torch.save(PTNet_state, os.path.join(opt.checkpoints_dir, opt.name, 'PTNet_ckpt%d%d.pth' % (epoch, total_steps)))
        np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')
        torch.save(D_state, os.path.join(opt.checkpoints_dir, opt.name, 'D_ckpt%d%d.pth' % (epoch, total_steps)))

    # linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        ler -= (opt.lr) / (opt.niter_decay)
        ler = max(ler, 0)
        for param_group in optimizer_PTNet.param_groups:
            param_group['lr'] = ler
        for param_group in optimizer_D.param_groups:
            param_group['lr'] = ler
        
        if is_main_process():
            print(f'Epoch {epoch}: Learning rate decayed to {ler:.6f}')

# Clean up DDP
cleanup_ddp()
