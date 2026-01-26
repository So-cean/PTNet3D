### This code is largely borrowed from pix2pixHD pytorch implementation
### https://github.com/NVIDIA/pix2pixHD
from collections import Counter
import re
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
        dist.init_process_group(backend='nccl', init_method='env://')
    
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
    print('#training images = %d' % dataset_size)

# ===== 统计 CP 比例 =====
cp_labels = [data_loader.dataset.data_dicts[i].get('CP') if isinstance(data_loader.dataset.data_dicts[i], dict) else None for i in range(len(data_loader.dataset))]
counts = Counter([c for c in cp_labels if c in (1, 2)])
n1 = counts.get(1, 0)
n2 = counts.get(2, 0)
# 如果没有提供 csv（没有有效的 CP 标签），则按照 1:1 加权
if n1 + n2 == 0:
    lambda_cp1 = 0.5
    lambda_cp2 = 0.5
else:
    lambda_cp1 = n2 / (n1 + n2)
    lambda_cp2 = 1 - lambda_cp1
if is_main_process():
    print(f'[PTNet]  CP1:{n1}  CP2:{n2}  →  loss weight λ={lambda_cp1:.2f}')
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
    D = DDP(D, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
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
# Update: networks.GANLoss no longer accepts a `tensor=` argument. Use the new signature.
criterionGAN = GANLoss(use_lsgan=True, reduction='none').to(device)
mse = torch.nn.MSELoss(reduction='none')

# training/display parameter
visualizer = Visualizer(opt) if is_main_process() else None
total_steps = (start_epoch - 1) * dataset_size + epoch_iter
display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

##############################################################################
# Training code
##############################################################################

# Get the raw D module for forward (handle DDP wrapper)
D_module = D.module if hasattr(D, 'module') else D

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    
    # Set epoch for DistributedSampler to shuffle data differently each epoch
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)
    elif hasattr(dataset, 'sampler') and hasattr(dataset.sampler, 'set_epoch'):
        dataset.sampler.set_epoch(epoch)
    
    for i, data in enumerate(dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize * world_size  # Account for all GPUs
        epoch_iter += opt.batchSize * world_size

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ##############################################################################
        # Forward Pass
        ##############################################################################

        input_image = Variable(data['img_A'].to(device))
        target_image = Variable(data['img_B'].to(device))
        # prepare CP labels robustly (handle missing csv or None values)
        batch_size = input_image.size(0)
        raw_cp = data.get('CP', None)
        if isinstance(raw_cp, torch.Tensor):
            cp_tensor = raw_cp.to(device)
        elif isinstance(raw_cp, (list, tuple)):
            cp_list = [0 if c is None else c for c in raw_cp]
            cp_tensor = torch.tensor(cp_list, dtype=torch.long, device=device)
        elif raw_cp is None:
            cp_tensor = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            # scalar or other -> broadcast
            cp_tensor = torch.full((batch_size,), int(raw_cp), dtype=torch.long, device=device)
        cp_labels = Variable(cp_tensor)  # shape: (B,)
        # create weight tensor using computed lambdas (use tensors on same device)
        lambda_t1 = torch.tensor(lambda_cp1, dtype=torch.float32, device=device)
        lambda_t2 = torch.tensor(lambda_cp2, dtype=torch.float32, device=device)
        weight = torch.where(cp_labels == 1, lambda_t1, lambda_t2)  # shape: (B,)

        # Synthesize and MSE loss
        generated = PTNet(input_image)
        loss_mse = mse(generated, target_image).flatten(1).mean(dim=1)  # shape: (B,)

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
                                                            ext_discriminator, reduction='none')

        # Compute overall loss
        loss_D = (loss_D_fake + loss_D_real).mean() * 0.5
        loss_G = (loss_mse * weight).mean() * 100.0 + (loss_G_GAN * weight).mean() + (loss_G_GAN_Feat_ext * weight).mean() * 10.0 + (loss_G_GAN_Feat * weight).mean() * 10.0
        # loss_G = loss_mse * 100.0 + loss_G_GAN + loss_G_GAN_Feat_ext * 10.0 + loss_G_GAN_Feat * 10.0
        loss_dict = dict(
            zip(['MSE', 'G_GAN', 'G_GAN_Feat_ext', 'G_GAN_Feat', 'D_fake', 'D_real'],
                [loss_mse.mean().item(),
                loss_G_GAN.mean().item(),
                loss_G_GAN_Feat_ext.mean().item(),
                loss_G_GAN_Feat.mean().item(),
                loss_D_fake.mean().item(),
                loss_D_real.mean().item()])
        )
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

        # print out loss (only main process)
        if is_main_process() and total_steps % opt.print_freq == print_delta:
            errors = {k: v if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)

        # display output images (only main process)
        if is_main_process() and save_fake:
            if 'label' in data:
                visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0, :, :, :, 15], opt.label_nc)),
                                       ('synthesized_image', util.tensor2im(generated.data[0, :, :, :, 15])),
                                       ('real_image', util.tensor2im(data['image'][0, :, :, :, 15]))])
                visualizer.display_current_results(visuals, epoch, total_steps)
            else:
                print("Warning: 'label' key not found in data. Skipping visualization.")

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
        for param_group in optimizer_PTNet.param_groups:
            param_group['lr'] = ler
            if is_main_process():
                print('change lr to ')
                print(param_group['lr'])
        for param_group in optimizer_D.param_groups:
            param_group['lr'] = ler

# Clean up DDP
cleanup_ddp()
