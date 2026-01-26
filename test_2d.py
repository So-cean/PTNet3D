import os
import csv
from options.test_options import TestOptions
from models.models import create_model
import nibabel as nib
import numpy as np
import torch
import monai.transforms as monai_transforms
from monai.metrics import PSNRMetric, SSIMMetric
import SimpleITK as sitk

opt = TestOptions().parse(save=False)

PTNet, _, _ = create_model(opt)
PTNet.cuda()
PTNet.eval()
PTNet.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, opt.whichmodel)))

des = os.path.join(opt.dataroot, opt.name + '_' + opt.whichmodel)
os.makedirs(des, exist_ok=True)

# -----------  MONAI 2-D transform（与训练一致） -----------
pix_dim = (0.8, 0.8, -1)
# pix_dim = (1.0, 1.0, -1)
transform = monai_transforms.Compose([
    monai_transforms.LoadImaged(keys=["image"]),
    monai_transforms.EnsureChannelFirstd(keys=["image"]),
    monai_transforms.EnsureTyped(keys=["image"], dtype=torch.float32),
    monai_transforms.Orientationd(keys=["image"], axcodes="RAS"),
    monai_transforms.Spacingd(keys=["image"], pixdim=pix_dim, mode="bilinear"),
    monai_transforms.CenterSpatialCropd(keys=["image"], roi_size=(256, 256, -1)),
    monai_transforms.SpatialPadd(keys=["image"], spatial_size=(256, 256, -1),
                                 mode="constant", constant_values=0),
    monai_transforms.ThresholdIntensityd(keys=["image"], threshold=0.0, above=True, cval=0.0),
    monai_transforms.ScaleIntensityRangePercentilesd(
        keys=["image"], lower=0, upper=opt.norm_perc, b_min=0, b_max=1, clip=True),
])

# -----------  指标计算器（2-D） -----------
psnr_fun = PSNRMetric(max_val=1.0)
ssim_fun = SSIMMetric(spatial_dims=2, data_range=1.0)

def calc_metrics(y_pred, y_true):
    """输入 (H,W) numpy"""
    y_pred = torch.from_numpy(y_pred).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    y_true = torch.from_numpy(y_true).unsqueeze(0).unsqueeze(0)
    # ensure float32 tensors
    y_pred = y_pred.to(dtype=torch.float32)
    y_true = y_true.to(dtype=torch.float32)
    psnr = psnr_fun(y_pred, y_true).item()
    ssim = ssim_fun(y_pred, y_true).item()
    # compute MAE and MSE using torch to avoid NumPy<->Torch dispatch issues
    mae  = torch.mean(torch.abs(y_pred - y_true)).item()
    mse  = torch.mean((y_pred - y_true) ** 2).item()
    return psnr, ssim, mae, mse

def inference_and_save(vol_path, prefix, save=True):
    """
    推理 + 保存 fake & real（transform 后空间）
    返回：transform 后的 tensor（用于后续配对计算）
    """
    fname = os.path.basename(vol_path)
    filename = fname.split(opt.extension)[0]
    print(f'Inference: {fname}')
    processed = transform({"image": vol_path})
    vol_tensor = processed["image"] # shape: (1, H, W, D)
    affine  = processed["image"].affine
    pixdim  = processed["image"].pixdim[:3]

    pred_vol = np.zeros(vol_tensor[0].shape, dtype=np.float32)
    for z in range(vol_tensor.shape[-1]):
        slc = vol_tensor[..., z]                      # (1, H, W)
        ipt = (slc.unsqueeze(0).to(dtype=torch.float32, device='cuda'))        # (1,1,H,W)
        out = PTNet(ipt)
        pred_vol[:, :, z] = torch.squeeze(out).detach().cpu().numpy()

    # # --- 使用 SimpleITK 做闭运算得到更稳健的 brain mask ---
    # # 将数据从 (H, W, D) -> (D, H, W) 以适配 sitk.GetImageFromArray
    # np_vol = vol_tensor[0].numpy().astype(np.float32)   # (H, W, D)
    # sitk_img = sitk.GetImageFromArray(np.transpose(np_vol, (2, 0, 1)))  # (z,y,x)
    # # 设置 spacing 为 (z_spacing, y_spacing, x_spacing)
    # sitk_img.SetSpacing((float(pixdim[2]), float(pixdim[1]), float(pixdim[0])))

    # # 二值化（阈值 > 0），再做闭运算去除小孔/连通缺失
    # bin_img = sitk.BinaryThreshold(sitk_img, lowerThreshold=1e-6, upperThreshold=1.0, insideValue=1, outsideValue=0)

    # # 设定闭运算半径（以 mm 为基准，转换为体素）；这里默认 5mm 的结构元素大小（仅在体素平面上主要作用）
    # rad_mm = 5.0
    # # 取 in-plane 最小像素间距计算半径（避免 z 轴过度膨胀）
    # px = max(pixdim[0], 1e-6); py = max(pixdim[1], 1e-6)
    # rad_x = max(1, int(round(rad_mm / px)))
    # rad_y = max(1, int(round(rad_mm / py)))
    # # 取较大值作为 scalar radius（SimpleITK 接受 scalar 半径）
    # radius_vox = max(rad_x, rad_y, 1)

    # closed = sitk.BinaryMorphologicalClosing(bin_img, radius_vox)

    # # 转回 numpy 并复原轴 (D,H,W) -> (H,W,D)
    # mask_np = np.transpose(sitk.GetArrayFromImage(closed), (1, 2, 0)).astype(bool)

    # pred_vol *= mask_np

    # 保存 fake
    if save:    
        fake_nii = nib.Nifti1Image(pred_vol, affine)
        fake_nii.header.set_zooms(pixdim)
        nib.save(fake_nii, os.path.join(des, f'{filename}_PTNetSynth_{prefix}'+opt.extension))

        # 保存 real（transform 后）
        real_nii = nib.Nifti1Image(vol_tensor[0].numpy(), affine)
        real_nii.header.set_zooms(pixdim)
        nib.save(real_nii, os.path.join(des, f'{filename}__PTNetSynth_{prefix}'+opt.extension))

    return vol_tensor[0], pred_vol  # 返回用于配对计算

# -----------  1. external 推理（无 GT） -----------
external_path = os.path.join(opt.dataroot, 'external')
ext_lst = [i for i in os.listdir(external_path) if i.endswith(opt.extension)]
with torch.no_grad():
    for f in ext_lst:
        inference_and_save(os.path.join(external_path, f), 'external')

# -----------  2. test_A 推理 + 与 test_B 配对计算指标 -----------
testA_path = os.path.join(opt.dataroot, 'test_A')
testB_path = os.path.join(opt.dataroot, 'test_B')
testA_lst = [i for i in os.listdir(testA_path) if i.endswith(opt.extension)]

csv_f = open(os.path.join(des, 'metrics_testAB.csv'), 'w', newline='')
writer = csv.writer(csv_f)
writer.writerow(['file', 'PSNR', 'SSIM', 'MAE', 'MSE'])

with torch.no_grad():
    for fname in testA_lst:
        gt_path = os.path.join(testB_path, fname)
        if not os.path.exists(gt_path):
            print(f'Skip {fname}: no paired GT in test_B')
            continue

        # 推理 testA 并拿到 fake
        tensA, fake_vol = inference_and_save(os.path.join(testA_path, fname), 'A')
        # 处理 GT
        tensB, _ = inference_and_save(gt_path, 'B', save=False)  # 仅为了保存 GT，不重复保存 fake

        # 逐切片计算指标
        psnr_l, ssim_l, mae_l, mse_l = [], [], [], []
        for z in range(tensA.shape[-1]):
            y_pred = fake_vol[..., z]
            y_true = tensB[..., z].numpy()
            mask = (y_true > 0)
            if mask.sum() == 0:
                continue
            
            # print(f"y_pred shape: {y_pred.shape}, y_true shape: {y_true.shape}")
            # print(f"y_pred dtype: {y_pred.dtype}, y_true dtype: {y_true.dtype}")
            p, s, m, ms = calc_metrics(y_pred, y_true)
            psnr_l.append(p); ssim_l.append(s); mae_l.append(m); mse_l.append(ms)
            # print(f"p type : {type(p)}, s type: {type(s)}, m type: {type(m)}, ms type: {type(ms)}")
            
        writer.writerow([fname,
                         np.mean(psnr_l),
                         np.mean(ssim_l),
                         np.mean(mae_l),
                         np.mean(mse_l)])
        print(f'{fname}  PSNR={np.mean(psnr_l):.3f}  SSIM={np.mean(ssim_l):.3f}  '
              f'MAE={np.mean(mae_l):.4f}  MSE={np.mean(mse_l):.4f}')

csv_f.close()
print('All done. Metrics saved to:', os.path.join(des, 'metrics_testAB.csv'))