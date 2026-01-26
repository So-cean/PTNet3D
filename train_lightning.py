#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, math, time, numpy as np
import torch, torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from options.train_options import TrainOptions
from data.data_loader      import CreateDataLoader
from models.models         import create_model
from models.networks       import GANLoss, feature_loss, discriminate
from util.image_pool       import ImagePool
#For debugging consider passing CUDA_LAUNCH_BLOCKING=1
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from torch.nn import HuberLoss

class PTNet3DSystem(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.PTNet, self.D, self.ext_disc = create_model(opt)
        self.criterionGAN = GANLoss(use_lsgan=True)
        self.mse = nn.MSELoss()
        self.fake_pool = ImagePool(0)
        self.automatic_optimization = False        # 手动更新 G/D

    def forward(self, x):
        return self.PTNet(x)

    def training_step(self, batch, batch_idx):
        img_A = batch['img_A']
        img_B = batch['img_B']

        # 优化器
        opt_g, opt_d = self.optimizers()

        # ---------------------
        #  1. 生成器
        # ---------------------
        fake_B = self.PTNet(img_A)
        loss_mse = self.mse(fake_B, img_B) * 100.0

        # GAN loss
        pred_fake = self.D(torch.cat([img_A, fake_B], dim=1))
        loss_g_gan = self.criterionGAN(pred_fake, True)

        # 特征匹配
        pred_real = self.D(torch.cat([img_A, img_B], dim=1))
        loss_g_feat, loss_g_feat_ext = feature_loss(
            self.opt, img_B, fake_B, pred_real, pred_fake, self.ext_disc)

        loss_G = loss_mse + loss_g_gan + loss_g_feat*10.0 + loss_g_feat_ext*10.0

        opt_g.zero_grad()
        self.manual_backward(loss_G)
        opt_g.step()

        # ---------------------
        #  2. 判别器
        # ---------------------
        pred_fake_pool = discriminate(self.D, self.fake_pool, img_A, fake_B.detach(), use_pool=True)
        loss_d_fake = self.criterionGAN(pred_fake_pool, False)
        loss_d_real = self.criterionGAN(pred_real, True)
        loss_D = (loss_d_fake + loss_d_real) * 0.5

        opt_d.zero_grad()
        self.manual_backward(loss_D)
        opt_d.step()

        # 日志
        self.log_dict({'MSE': loss_mse, 'G_GAN': loss_g_gan,
                       'G_GAN_Feat': loss_g_feat, 'G_GAN_Feat_ext': loss_g_feat_ext,
                       'D_fake': loss_d_fake, 'D_real': loss_d_real,
                       'loss_G': loss_G, 'loss_D': loss_D},
                      prog_bar=True, logger=True, batch_size=img_A.size(0))

    def configure_optimizers(self):
        lr = self.opt.lr
        opt_g = torch.optim.Adam(self.PTNet.parameters(),      lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.D.parameters(),          lr=lr, betas=(0.5, 0.999))
        return [opt_g, opt_d], []

    def lr_scheduler_step(self, scheduler, metric):
        # 原代码线性衰减
        if self.current_epoch > self.opt.niter:
            lr = self.opt.lr * (1 - (self.current_epoch - self.opt.niter) / self.opt.niter_decay)
            for opt in self.optimizers():
                for pg in opt.param_groups:
                    pg['lr'] = lr

# -------------------- main --------------------
def main():
    opt = TrainOptions().parse()
    # 复用原 data loader
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    train_loader = dataset          # 本身就是 DataLoader

    model = PTNet3DSystem(opt)

    # 常用回调
    ckpt_cb = ModelCheckpoint(dirpath=os.path.join(opt.checkpoints_dir, opt.name),
                              filename='PTNet_{epoch:03d}',
                              save_top_k=-1, every_n_epochs=opt.save_epoch_freq,
                              save_last=True)
    lr_cb   = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=opt.niter + opt.niter_decay,
        accelerator='gpu',
        devices='auto',                     
        strategy='auto',
        precision='bf16-mixed',               # ← 关键：BF16 混合精度
        callbacks=[ckpt_cb, lr_cb],
        log_every_n_steps=opt.print_freq,
        num_sanity_val_steps=0,
        gradient_clip_val=None
    )
    trainer.fit(model, train_loader)

if __name__ == '__main__':
    main()