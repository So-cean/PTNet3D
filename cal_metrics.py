from collections import defaultdict
from functools import lru_cache
import os
from pathlib import Path
from unittest import result
from matplotlib import pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import lpips  # 引入 LPIPS 库
from sympy import im
import torch
from tqdm import tqdm
import SimpleITK as sitk

# npu 
# import torch_npu
# from torch_npu.contrib import transfer_to_npu  

# torch.npu.config.allow_internal_format = False
# torch.npu.set_compile_mode(jit_compile=False)


def create_brain_mask(image, kernel_radius=3, fill_holes=True):
    """生成大脑掩膜（可选小孔填充）"""
    # 初始Otsu阈值分割来生成mask
    initialMask = sitk.OtsuThreshold(image, 0, 1, 200)
 
    # 形态学膨胀操作
    dilateFilter = sitk.BinaryDilateImageFilter()
    dilateFilter.SetKernelType(sitk.sitkBall)
    dilateFilter.SetKernelRadius(kernel_radius)
    dilatedMask = dilateFilter.Execute(initialMask)
 
    # 形态学腐蚀操作
    erodeFilter = sitk.BinaryErodeImageFilter()
    erodeFilter.SetKernelType(sitk.sitkBall)
    erodeFilter.SetKernelRadius(kernel_radius)
    erodedMask = erodeFilter.Execute(dilatedMask)
 
    # 连通域分析
    labelFilter = sitk.ConnectedComponentImageFilter()
    labeledMask = labelFilter.Execute(erodedMask)
 
    # 计算连通域的大小，并找出最大的连通域
    labelShapeStats = sitk.LabelShapeStatisticsImageFilter()
    labelShapeStats.Execute(labeledMask)
    largestLabel = None
    largestSize = 0
    for label in labelShapeStats.GetLabels():
        size = labelShapeStats.GetPhysicalSize(label)
        if size > largestSize:
            largestLabel = label
            largestSize = size
 
    # 创建一个只包含最大连通域的掩码
    mask = sitk.BinaryThreshold(labeledMask, largestLabel, largestLabel, 1, 0)
    
    # 可选的小孔填充步骤
    if fill_holes:
        fillHolesFilter = sitk.BinaryFillholeImageFilter()
        mask = fillHolesFilter.Execute(mask)
    
    return mask


def create_mask_threshold(data, threshold=1e-4):
    """基于阈值创建掩膜"""
    mask = data > threshold
    return mask

def create_roi_mask(data, kernel_radius=3):
    """创建ROI掩膜"""
    # 将numpy数组转换为SimpleITK图像
    image = sitk.GetImageFromArray(data)
    mask_image = create_mask_threshold(image, threshold=0)
    mask = sitk.GetArrayFromImage(mask_image).astype(bool)
    return mask

def percentile_normalize(image, mask, lower=0.1, upper=99.9):
    """
    基于百分位数的归一化，使用掩膜处理非零背景
    
    参数:
        image: 输入图像 (numpy数组)
        mask: 脑组织掩膜 (布尔数组)
        lower: 下百分位 (0-100)
        upper: 上百分位 (0-100)
    
    返回:
        归一化后的图像 (float32)
    """
    # 验证输入
    if image.shape != mask.shape:
        raise ValueError("图像和掩膜形状不匹配")
    
    # 提取ROI区域内的数据 (脑组织区域)
    roi_data = image[mask]
    
    # 计算ROI区域的百分位
    if len(roi_data) == 0:
        # 如果没有ROI数据，使用整个图像的统计量
        print("警告: ROI区域为空，使用整个图像计算百分位")
        roi_data = image.flatten()
        if len(roi_data) == 0:
            return np.zeros_like(image, dtype=np.float32)  # 空图像
    
    # 计算百分位
    v_min, v_max = np.percentile(roi_data, [lower, upper])
    
    # 防止无效范围
    if v_max - v_min < 1e-6:
        v_max = v_min + 1e-6
    
    # 归一化所有值
    volume_clipped = np.clip(image, v_min, v_max)
    volume_normalized = (volume_clipped - v_min) / (v_max - v_min)
    volume_normalized = np.clip(volume_normalized, 0, 1)
    
    # 将背景设为0
    volume_normalized[~mask] = 0
    
    return volume_normalized.astype(np.float32)

def calculate_ssim(file1, file2, kernel_radius=3, lower_percentile=0.1, upper_percentile=99.9, win_size=7, use_mask=True):
    """
    计算两个 NIfTI 文件的 SSIM
    步骤：
    1. 读取两个 NIfTI 文件
    2. 创建大脑掩膜
    3. 使用掩膜进行百分位归一化
    4. 计算归一化后图像的 SSIM
    
    参数:
        file1, file2: NIfTI 文件路径
        kernel_radius: 形态学操作核半径
        lower_percentile: 归一化下限百分位
        upper_percentile: 归一化上限百分位
        win_size: SSIM 计算窗口大小
        use_mask: 是否使用掩膜
        
    返回:
        ssim_value: 计算的 SSIM 值
    """
    # 读取 NIfTI 文件
    nii1 = nib.load(file1).get_fdata()
    nii2 = nib.load(file2).get_fdata()
    
    # 创建掩膜
    mask = create_roi_mask(nii1, kernel_radius) if use_mask else np.ones_like(nii1, dtype=bool)
    
    # 百分位归一化
    nii1_norm = percentile_normalize(nii1, mask, lower_percentile, upper_percentile)
    nii2_norm = percentile_normalize(nii2, mask, lower_percentile, upper_percentile)
    
    # 计算 SSIM (使用掩膜区域)
    ssim_value = masked_ssim(
        nii1_norm, 
        nii2_norm, 
        mask=mask,
        win_size=win_size,
        data_range=1.0  # 数据范围 [0, 1]
    )
        
    return ssim_value

def masked_ssim(img1, img2, mask, win_size=7, **kwargs):
    """
    在掩膜区域内计算 SSIM（切片级计算）
    
    参数:
        img1, img2: 输入图像 (3D numpy 数组)
        mask: 掩膜区域 (3D boolean 数组)
        win_size: SSIM 窗口大小
        **kwargs: 传递给 structural_similarity 的其他参数
        
    返回:
        ssim: 平均切片 SSIM 值
    """
    ssim_values = []
    
    # 遍历每个切片
    for z in range(img1.shape[0]):
        slice1 = img1[z]
        slice2 = img2[z]
        slice_mask = mask[z]
        
        # 跳过完全空的切片
        if np.sum(slice_mask) == 0:
            continue
            
        # 计算当前切片的 SSIM
        ssim_val = structural_similarity(
            slice1, slice2,
            win_size=win_size,
            data_range=kwargs.get('data_range', 1.0),
            gaussian_weights=True,
            full=False
        )
        ssim_values.append(ssim_val)
    
    # 计算平均 SSIM
    return np.mean(ssim_values) if ssim_values else 0

def calculate_mae(nii1_path, nii2_path, use_mask=True):
    """计算MAE指标"""
    # 加载图像
    nii1 = nib.load(nii1_path).get_fdata()
    nii2 = nib.load(nii2_path).get_fdata()
    
    if nii1.shape != nii2.shape:
        raise ValueError("Shape mismatch")
    
    # 创建掩膜
    mask = create_roi_mask(nii1) if use_mask else np.ones_like(nii1,dtype=bool)
    
    # 归一化整个图像
    nii1_norm = percentile_normalize(nii1, mask)
    nii2_norm = percentile_normalize(nii2, mask)
    
    # 应用Mask去除背景
    img1_roi = nii1_norm[mask > 0]
    img2_roi = nii2_norm[mask > 0]
    
    mae = np.mean(np.abs(img1_roi - img2_roi))
    
    return mae

def calculate_nmse(nii1_path, nii2_path, use_mask=True):
    """计算NMSE指标"""
    # 加载图像
    nii1 = nib.load(nii1_path).get_fdata()
    nii2 = nib.load(nii2_path).get_fdata()
    
    if nii1.shape != nii2.shape:
        raise ValueError("Shape mismatch")
    
    # 创建掩膜
    mask = create_roi_mask(nii1) if use_mask else np.ones_like(nii1,dtype=bool)
    
    # 归一化整个图像
    nii1_norm = percentile_normalize(nii1, mask)
    nii2_norm = percentile_normalize(nii2, mask)
    
    # 应用Mask去除背景
    img1_roi = nii1_norm[mask > 0]
    img2_roi = nii2_norm[mask > 0]
    
    mse = np.mean((img1_roi - img2_roi) ** 2)
    nmse = mse / (np.var(img1_roi) + 1e-8)
    
    return nmse

def calculate_mse(nii1_path, nii2_path, use_mask=True):
    """计算MSE指标"""
    # 加载图像
    nii1 = nib.load(nii1_path).get_fdata()
    nii2 = nib.load(nii2_path).get_fdata()
    
    if nii1.shape != nii2.shape:
        raise ValueError("Shape mismatch")
    
    # 创建掩膜
    mask = create_roi_mask(nii1) if use_mask else np.ones_like(nii1,dtype=bool)
    
    # 归一化整个图像
    nii1_norm = percentile_normalize(nii1, mask)
    nii2_norm = percentile_normalize(nii2, mask)
    
    # 应用Mask去除背景
    img1_roi = nii1_norm[mask > 0]
    img2_roi = nii2_norm[mask > 0]
    
    mse = np.mean((img1_roi - img2_roi) ** 2)
    
    return mse

def calculate_psnr(nii1_path, nii2_path, use_mask=True):
    """计算PSNR指标"""
    # 加载图像
    nii1 = nib.load(nii1_path).get_fdata()
    nii2 = nib.load(nii2_path).get_fdata()
    
    if nii1.shape != nii2.shape:
        raise ValueError("Shape mismatch")
    
    # 创建掩膜
    mask = create_roi_mask(nii1) if use_mask else np.ones_like(nii1,dtype=bool)
    
    # 归一化整个图像
    nii1_norm = percentile_normalize(nii1)
    nii2_norm = percentile_normalize(nii2)
    
    # 应用Mask去除背景
    img1_roi = nii1_norm[mask > 0]
    img2_roi = nii2_norm[mask > 0]
    
    mse = np.mean((img1_roi - img2_roi) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 1.0  # 归一化后最大值为 1
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr
    

# 初始化 LPIPS 计算器
lpips_loss = lpips.LPIPS(net='alex')  # 使用 AlexNet 作为 backbone
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
lpips_loss.to(device)
lpips_loss.eval()
def calculate_lpips(nii1_path, nii2_path, use_mask=True):
    """计算LPIPS感知相似度指标"""
    # 加载图像
    nii1 = nib.load(nii1_path).get_fdata()
    nii2 = nib.load(nii2_path).get_fdata()
    
    if nii1.shape != nii2.shape:
        raise ValueError("Shape mismatch")
    
    # 创建掩膜
    mask = create_roi_mask(nii1) if use_mask else np.ones_like(nii1,dtype=bool)
    valid_slices = [z for z in range(nii1.shape[2]) if np.any(mask[:, :, z] > 0)]
    
    total_lpips, processed_slices = 0.0, 0
    
    # 对整个体积进行归一化（确保一致对比度）
    v_min1, v_max1 = np.percentile(nii1[mask & (nii1 > 0)], [0.1, 99.9])
    v_min2, v_max2 = np.percentile(nii2[mask & (nii2 > 0)], [0.1, 99.9])
    
    # 使用平均值确保一致对比度
    v_min = (v_min1 + v_min2) / 2
    v_max = (v_max1 + v_max2) / 2
    
    # 归一化整个体积
    nii1_norm = (np.clip(nii1, v_min, v_max) - v_min) / (v_max - v_min + 1e-10)
    nii2_norm = (np.clip(nii2, v_min, v_max) - v_min) / (v_max - v_min + 1e-10)
    
    # 背景设为0
    nii1_norm[nii1 == 0] = 0
    nii2_norm[nii2 == 0] = 0
    
   
    
    for z in valid_slices:
        slice1 = nii1_norm[:, :, z]
        slice2 = nii2_norm[:, :, z]
        slice_mask = mask[:, :, z]
        
        if np.sum(slice_mask) < 10:  # 跳过无效切片
            continue
            
        # 应用掩膜
        slice1_masked = slice1 * slice_mask
        slice2_masked = slice2 * slice_mask
        
        # 转换到 [-1, 1] 范围
        slice1_norm = (slice1_masked * 2) - 1
        slice2_norm = (slice2_masked * 2) - 1

        # 转换为RGB张量
        slice1_rgb = np.stack([slice1_norm]*3, axis=-1).transpose(2, 0, 1)
        slice2_rgb = np.stack([slice2_norm]*3, axis=-1).transpose(2, 0, 1)
        tensor1 = torch.tensor(slice1_rgb, device=device).unsqueeze(0).float()
        tensor2 = torch.tensor(slice2_rgb, device=device).unsqueeze(0).float()
        
        with torch.no_grad():
            total_lpips += lpips_loss(tensor1, tensor2).item()
        processed_slices += 1
        
    return total_lpips / max(processed_slices, 1)  # 避免除零


def calculate_metrics(nii1_path, nii2_path, use_mask=True):
    """计算所有指标"""
    # mask coverage
    mask = create_roi_mask(nib.load(nii1_path).get_fdata()) if use_mask else np.ones_like(nib.load(nii1_path).get_fdata(),dtype=bool)
    coverage = np.sum(mask) / mask.size
    print(f"Mask coverage: {coverage:.4f}")
    # 计算 MAE
    mae = calculate_mae(nii1_path, nii2_path, use_mask=use_mask)
    # print(f"MAE: {mae:.4f}")
    
    # 计算 NMSE
    nmse = calculate_nmse(nii1_path, nii2_path, use_mask=use_mask)
    # print(f"NMSE: {nmse:.6f}")
    
    # 计算 MSE
    mse = calculate_mse(nii1_path, nii2_path, use_mask=use_mask)
    # print(f"MSE: {mse:.6f}")
    
    # 计算 PSNR
    psnr = calculate_psnr(nii1_path, nii2_path, use_mask=use_mask)
    # print(f"PSNR: {psnr:.2f} dB")
    
    # 计算 SSIM (按切片计算)
    ssim_val = calculate_ssim(nii1_path, nii2_path, kernel_radius=3, lower_percentile=0.1, upper_percentile=99.9, win_size=7, use_mask=use_mask)
    # print(f"SSIM: {ssim_val:.4f}")
    
    # 计算 LPIPS
    lpips_val = calculate_lpips(nii1_path, nii2_path, use_mask=use_mask)
    # print(f"LPIPS: {lpips_val:.4f}")
    
    return  mae, nmse, mse, psnr, ssim_val, lpips_val

def plot_psnr_age(df, save_path='plots/psnr_vs_age.png'):
    plt.figure(figsize=(10, 6))
    sns.regplot(x='Age', y='PSNR', data=df, 
                scatter_kws={'alpha':0.7, 'color':'#1f77b4'},
                line_kws={'color':'#ff7f0e', 'lw':2})
    plt.title('PSNR vs Age Trend', fontsize=14)
    plt.xlabel('Age (months)', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_ssim_age(df, save_path='plots/ssim_vs_age.png'):
    plt.figure(figsize=(10, 6))
    sns.regplot(x='Age', y='SSIM', data=df, 
                scatter_kws={'alpha':0.7, 'color':'#2ca02c'},
                line_kws={'color':'#d62728', 'lw':2})
    plt.title('SSIM vs Age Trend', fontsize=14)
    plt.xlabel('Age (months)', fontsize=12)
    plt.ylabel('SSIM', fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_lpips_distribution(df, save_path='plots/lpips_distribution.png'):
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(x='Age', y='LPIPS', data=df, 
                    hue='LPIPS', palette='viridis', 
                    size='LPIPS', sizes=(50, 200))
    plt.title('Perceptual Quality (LPIPS) Distribution by Age', fontsize=14)
    plt.xlabel('Age (months)', fontsize=12)
    plt.ylabel('LPIPS (lower is better)', fontsize=12)
    plt.colorbar(scatter.collections[0], label='LPIPS Intensity')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_mae_group(df,save_path='plots/mae_by_agegroup.png'):
    # Add age group categories 
    df['AgeGroup'] = pd.cut(df['Age'], 
                            bins=[0, 3, 6, 9, 12,15,18,24],
                            labels=['0-3m', '4-6m', '7-9m', '10-12m','13-15m','16-18m','19-24m'])

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='AgeGroup', y='MAE', data=df,
                hue='AgeGroup', palette='Pastel1', legend=False,
                showfliers=False,  # Hide outliers
                width=0.6)
    plt.title('MAE Distribution by Age Group', fontsize=14)
    plt.xlabel('Age Group', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def collect_matched_paths(bash_path):
    """
    确保test_A、test_B、PTNet3D_pred中的文件正确匹配
    """
    # 使用字典按受试者ID组织文件路径
    subject_files = defaultdict(dict)
    
    bash_path = Path(bash_path)
    
    # 定义目录映射
    dir_mapping = {
        'test_A': 'source',
        'test_B': 'target', 
        'PTNet3D_pred': 'pred'
    }
    
    # 遍历所有目录，按受试者ID分类
    for root, dirs, files in os.walk(bash_path):
        for dir_name, file_type in dir_mapping.items():
            if dir_name in root:
                for file in files:
                    if file.endswith('.nii.gz'):
                        # 提取受试者ID（去除月份和后缀）
                        if 'PTNetSynth' in file:
                            # 处理预测文件: MNBCP000178_15mo_PTNetSynth.nii.gz
                            subject_id = file.split('_PTNetSynth')[0]
                        else:
                            # 处理真实文件: MNBCP000178_15mo.nii.gz  
                            subject_id = file.replace('.nii.gz', '')
                        
                        file_path = Path(root) / file
                        subject_files[subject_id][file_type] = file_path
    
    # 转换为匹配的列表，确保顺序一致
    img_list_0, img_list_1, pred_list, age_list = [], [], [], []
    
    for subject_id, files in sorted(subject_files.items()):
        # 检查是否所有必需文件都存在
        if 'source' in files and 'target' in files and 'pred' in files:
            img_list_0.append(files['source'])
            img_list_1.append(files['target']) 
            pred_list.append(files['pred'])
            
            # 从文件名提取年龄
            age_str = subject_id.split('_')[-1].replace('mo', '')
            try:
                age_list.append(int(age_str))
            except ValueError:
                age_list.append(None)
    
    return img_list_0, img_list_1, pred_list, age_list


if __name__ == "__main__":
    # 配置参数
    bash_path = "/public_bme2/bme-wangqian2/songhy2024/data/BCP_PTNet3D"
    img_list_0, img_list_1, pred_list, age_list = collect_matched_paths(bash_path)

    
    print(f"Found {len(img_list_0)} T1 images, {len(img_list_1)} T2 images, and {len(pred_list)} predicted images.")
    print(f"first T1 image: {img_list_0[0]}")
    print(f"first T2 image: {img_list_1[0]}")
    print(f"first predicted image: {pred_list[0]}")
    
    results = []
    for i, (nii1, nii2, nii3) in tqdm(enumerate(zip(img_list_0, img_list_1, pred_list)), total=len(img_list_1)):
        print(f"Processing {nii3}")
        if not os.path.exists(nii3):
            print(f"Warning: {nii3} does not exist, skipping...")
            continue
        


        mae, nmse, mse, psnr, ssim_val, lpips_val = calculate_metrics(nii2, nii3, use_mask=True)
        print(f"PSNR: {psnr:.2f} dB, SSIM: {ssim_val:.4f}, LPIPS: {lpips_val:.4f}, MAE: {mae:.4f}")
        
    #     # 记录结果
    #     results.append({
    #         'Age': age_list[i],
    #         'PSNR': psnr,
    #         'SSIM': ssim_val,
    #         'LPIPS': lpips_val,
    #         'MAE': mae,
    #         'File': Path(nii3).name
    #     })
        

    # # # 转换为DataFrame
    # df = pd.DataFrame(results)
    
    # plot_dir = "./plots_PTNet3D"
    # os.makedirs(plot_dir, exist_ok=True)
    # plot_psnr_age(df, save_path=os.path.join(plot_dir, 'psnr_vs_age.png'))
    # plot_ssim_age(df, save_path=os.path.join(plot_dir, 'ssim_vs_age.png'))
    # plot_lpips_distribution(df, save_path=os.path.join(plot_dir, 'lpips_distribution.png'))
    # plot_mae_group(df, save_path=os.path.join(plot_dir, 'mae_by_agegroup.png'))
    
    # # save results to csv
    # df.to_csv(os.path.join(plot_dir, 'metrics_results.csv'), index=False)
