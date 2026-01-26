# convert my bcp dataset into PTNet3D format
# input:
# dataroot

# PTNet3D structure:
# ./opt.dataroot 
#     - train_A # your source domain scans
#     - train_B # your target domain scans
#     - test_A # will be used for inference
#     - test_B # ground truth for evaluation

# input dataroor: ./BCP_Registered_Modality
# BCP_Registered_Modality/T1w # source domain # subject_month.nii.gz
# BCP_Registered_Modality/T2w # target domain

# split the dataset by subjects, filter month 6~24 month

import os
import shutil
import json
import random
from pathlib import Path
from collections import defaultdict

def convert_to_ptnet3d(data_root, output_root, train_ratio=0.8):
    # 创建输出目录结构
    output_root = Path(output_root)
    dirs = ["train_A", "train_B", "test_A", "test_B"]
    for d in dirs:
        (output_root / d).mkdir(parents=True, exist_ok=True)
    
    # 收集所有有效文件（6-24个月）
    t1_dir = Path(data_root) / "T1w"
    t2_dir = Path(data_root) / "T2w"
    
    # 按受试者组织文件
    subject_files = defaultdict(list)
    
    # 处理T1w文件
    for t1_file in t1_dir.glob("*.nii.gz"):
        filename = t1_file.name.replace('.nii.gz', '')  # 移除完整后缀
        parts = filename.split('_')  # 正确分割为["NCBCP161437", "11mo"]
        if len(parts) < 2:
            continue
        
        subject_id = parts[0]
        month_str = parts[1]
        # print(f"Processing file: {t1_file}, Subject: {subject_id}, Month: {month_str}")
        # 提取月份数字并过滤
        try:
            month = int(month_str.replace("mo", ""))
            if not (6 <= month <= 24):
                continue
        except ValueError:
            continue
        
        # 查找对应的T2w文件
        t2_file = t2_dir / f"{subject_id}_{month_str}.nii.gz"
        if not t2_file.exists():
            continue
        
        subject_files[subject_id].append({
            "subject": subject_id,
            "month": month,
            "month_str": month_str,
            "t1_path": t1_file,
            "t2_path": t2_file
        })
    
    # 按受试者划分数据集
    subjects = list(subject_files.keys())
    random.shuffle(subjects)
    split_idx = int(len(subjects) * train_ratio)
    train_subjects = subjects[:split_idx]
    test_subjects = subjects[split_idx:]
    
    # 复制文件到PTNet3D格式目录
    train_count = 0
    test_count = 0
    
    for subject_id, files in subject_files.items():
        for file_info in files:
            month_str = file_info["month_str"]
            t1_file = file_info["t1_path"]
            t2_file = file_info["t2_path"]
            
            # 使用原始文件名作为新文件名
            new_filename = f"{subject_id}_{month_str}.nii.gz"
            
            if subject_id in train_subjects:
                shutil.copy2(t1_file, output_root / "train_A" / new_filename)
                shutil.copy2(t2_file, output_root / "train_B" / new_filename)
                train_count += 1
            else:
                shutil.copy2(t1_file, output_root / "test_A" / new_filename)
                shutil.copy2(t2_file, output_root / "test_B" / new_filename)
                test_count += 1
    
    # 创建split.json文件
    split_data = {
        "train": train_subjects,
        "test": test_subjects,
        "description": "PTNet3D dataset split for BCP data (6-24 months)",
        "train_count": len(train_subjects),
        "test_count": len(test_subjects),
        "total_subjects": len(subjects),
        "total_scans": train_count + test_count
    }
    
    with open(output_root / "split.json", "w") as f:
        json.dump(split_data, f, indent=2)
    
    return split_data

if __name__ == "__main__":
    # 输入数据目录（包含T1w和T2w文件夹）
    input_data_root = "/public_bme2/bme-wangqian2/songhy2024/data/BCP"
    
    # 输出目录（PTNet3D格式）
    output_data_root = "/public_bme2/bme-wangqian2/songhy2024/data/BCP_PTNet3D"
    
    # 转换数据集
    split_info = convert_to_ptnet3d(input_data_root, output_data_root)
    
    # 打印结果
    print("数据集转换完成！")
    print(f"输出目录: {output_data_root}")
    print(f"训练集受试者数量: {split_info['train_count']}")
    print(f"测试集受试者数量: {split_info['test_count']}")
    print(f"总受试者数量: {split_info['total_subjects']}")
    print(f"总扫描数量: {split_info['total_scans']}")
    print(f"split.json 已保存在: {Path(output_data_root) / 'split.json'}")