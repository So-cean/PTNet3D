from multiprocessing import reduction
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np


###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_init3D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm3d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'instance3D':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    print(netD)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netD.cuda(gpu_ids[0])

    netD.apply(weights_init)
    return netD


def define_D_3D(input_nc, ndf, n_layers_D, norm='instance3D', use_sigmoid=False, num_D=1, getIntermFeat=False,
                gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator3D(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    print(netD)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init3D)
    return netD

def discriminate(D, fake_pool, input_label, test_image, use_pool=False):
    input_concat = torch.cat((input_label, test_image.detach()), dim=1)
    if use_pool:
        fake_query = fake_pool.query(input_concat)
        return D.forward(fake_query)
    else:
        return D.forward(input_concat)

##############################################################################
# Losses
##############################################################################
# class GANLoss(nn.Module):
#     def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
#                  tensor=torch.FloatTensor):
#         super(GANLoss, self).__init__()
#         self.real_label = target_real_label
#         self.fake_label = target_fake_label
#         self.real_label_var = None
#         self.fake_label_var = None
#         self.Tensor = tensor
#         if use_lsgan:
#             self.loss = nn.MSELoss()
#         else:
#             self.loss = nn.BCELoss()

#     def get_target_tensor(self, input, target_is_real):
#         target_tensor = None
#         if target_is_real:
#             create_label = ((self.real_label_var is None) or
#                             (self.real_label_var.numel() != input.numel()))
#             if create_label:
#                 real_tensor = self.Tensor(input.size()).fill_(self.real_label)
#                 self.real_label_var = Variable(real_tensor, requires_grad=False)
#             target_tensor = self.real_label_var
#         else:
#             create_label = ((self.fake_label_var is None) or
#                             (self.fake_label_var.numel() != input.numel()))
#             if create_label:
#                 fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
#                 self.fake_label_var = Variable(fake_tensor, requires_grad=False)
#             target_tensor = self.fake_label_var
#         return target_tensor

#     def __call__(self, input, target_is_real):
#         if isinstance(input[0], list):
#             loss = 0
#             for input_i in input:
#                 pred = input_i[-1]
#                 target_tensor = self.get_target_tensor(pred, target_is_real)
#                 loss += self.loss(pred, target_tensor)
#             return loss
#         else:
#             target_tensor = self.get_target_tensor(input[-1], target_is_real)
#             return self.loss(input[-1], target_tensor)
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real=1.0, target_fake=0.0, reduction='mean'):
        """
        reduction : 'mean' | 'none'
                    'mean' -> 返回标量（兼容旧代码）
                    'none' -> 返回 (B,) 向量（用于逐样本加权）
        """
        super().__init__()
        self.reduction = reduction
        self.register_buffer("real_label", torch.tensor(target_real))
        self.register_buffer("fake_label", torch.tensor(target_fake))
        # 核心 loss：不聚合版本
        self.loss = nn.MSELoss(reduction='none') if use_lsgan else nn.BCEWithLogitsLoss(reduction='none')

    def get_target_tensor(self, pred: torch.Tensor, target_is_real: bool):
        target_val = self.real_label if target_is_real else self.fake_label
        return torch.full_like(pred, target_val)

    def forward(self, input, target_is_real: bool):
        """
        input : 判别器输出
                · 单尺度：Tensor
                · 多尺度：list[list[Tensor]]  (pix2pixHD 风格)
        target_is_real : True  -> 真实标签
                         False -> 假标签
        返回：
            reduction='mean' -> 标量
            reduction='none' -> (B,) 向量（逐样本）
        """
        # ----------- 多尺度判别器 -----------
        if isinstance(input[0], list):
            batch_size = input[0][-1].size(0)          # 取第一张图 batch 维度
            loss_vec = torch.zeros(batch_size, device=input[0][-1].device)
            for input_i in input:
                pred = input_i[-1]                     # 最后一层特征图
                target_tensor = self.get_target_tensor(pred, target_is_real)
                # 先在空间上平均，再返回 (B,)
                loss_vec += self.loss(pred, target_tensor).mean(dim=[1, 2, 3])
            return loss_vec if self.reduction == 'none' else loss_vec.mean()

        # ----------- 单尺度 -----------
        pred = input[-1] if isinstance(input, list) else input
        target_tensor = self.get_target_tensor(pred, target_is_real)
        loss_vec = self.loss(pred, target_tensor).mean(dim=[1, 2, 3])  # [B]
        return loss_vec if self.reduction == 'none' else loss_vec.mean()

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


def feature_loss(opt, ori_img, syn_img, pred_real, pred_fake, ext_discriminator, reduction='none'):
    criterionFeat = torch.nn.L1Loss(reduction=reduction)

    if opt.dimension.startswith('2'):
        B = ori_img.size(0)
        loss_G_GAN_Feat = torch.zeros(B, device=ori_img.device)
        D_weights = 1.0 / 2
        for i in range(2):
            for j in range(len(pred_fake[i]) - 1):
                # 逐样本 L1，再在空间上平均 → [B]
                feat_diff = criterionFeat(pred_fake[i][j], pred_real[i][j].detach())  # [B,C,H,W]
                loss_G_GAN_Feat += D_weights * feat_diff.mean(dim=[1, 2, 3])

        # 外部 VGG 特征
        ori_img = ori_img.expand(-1, 3, -1, -1)
        syn_img = syn_img.expand(-1, 3, -1, -1)
        feat_resize = nn.Upsample(size=(224, 224))
        feat_res_real = ext_discriminator(feat_resize(ori_img))
        feat_res_fake = ext_discriminator(feat_resize(syn_img))
        loss_G_GAN_Feat_ext = torch.zeros(B, device=ori_img.device)
        vgg_weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        for tmp_i in range(len(feat_res_fake)):
            feat_diff = criterionFeat(feat_res_real[tmp_i].detach(), feat_res_fake[tmp_i])  # [B,C,H,W]
            loss_G_GAN_Feat_ext += (feat_diff.mean(dim=[1, 2, 3]) * vgg_weights[tmp_i])

        # 返回逐样本向量
        return loss_G_GAN_Feat, loss_G_GAN_Feat_ext
    elif opt.dimension.startswith('3'):
        loss_G_GAN_Feat = 0
        D_weights = 1.0 / 3
        for i in range(3):
            for j in range(len(pred_fake[i]) - 1):
                loss_G_GAN_Feat += D_weights * \
                                   criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
        ori_img = ori_img.expand(-1, 3, -1, -1, -1)
        syn_img = syn_img.expand(-1, 3, -1, -1, -1)
        feat_res_real = ext_discriminator(ori_img)
        feat_res_fake = ext_discriminator(syn_img)
        loss_G_GAN_Feat_ext = 0
        res_weights = [1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        feature_level = ['layer1', 'layer2', 'layer3', 'layer4']
        for tmp_i in range(len(feature_level)):
            loss_G_GAN_Feat_ext += criterionFeat(feat_res_real[feature_level[tmp_i]].detach(),
                                               feat_res_fake[feature_level[tmp_i]]) * res_weights[tmp_i]

        return loss_G_GAN_Feat, loss_G_GAN_Feat_ext
    else:
        raise NotImplementedError


##############################################################################
# Discriminator
##############################################################################


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


class MultiscaleDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator3D, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator3D(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool3d(3, stride=2, padding=[1, 1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
