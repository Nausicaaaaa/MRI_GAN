#!/usr/bin/env python3
"""
MRI数据预处理分步脚本

1. DICOM转NIfTI
2. 所有序列重采样到统一模板 [512, 512, 32]
3. 基于T1序列配准其他5个序列
4. 基于T1序列获取身体掩码
5. 应用身体掩码到所有序列
6. 去除空白切片+归一化
7. 制作为h5格式

配置说明:
- 在 Config 类中设置 GLOBLE_TEMPLETE 可指定预设模板
- 目标尺寸: TARGET_SIZE = [512, 512, 32]
- 如 GLOBLE_TEMPLETE 为 None，使用第一个患者的T1创建模板

使用方式:
1. 完整流程: python run_preprocessing.py --all
2. 分步执行: 
   - 步骤1-2 (DICOM转NIfTI + 重采样): python CE-MRI-synthesis/preprocessing/preprocessing.py --steps 1 2
   - 步骤3-5 (配准 + 身体掩码 + 应用): python CE-MRI-synthesis/preprocessing/preprocessing.py --steps 3 4 5
   - 步骤6-7 (归一化 + h5): python CE-MRI-synthesis/preprocessing/preprocessing.py --steps 6 7

"""

import argparse
import glob
import os
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, '/mnt/data/KASR/Dengsiyi/MRI_GAN/CE-MRI-synthesis')
sys.path.insert(0, '/mnt/data/KASR/Dengsiyi/MRI_GAN/CE-MRI-synthesis/preprocessing/reg_series')

# ==================== 配置 ====================
class Config:
    # 路径配置
    PROJECT_ROOT = "/mnt/data/KASR/Dengsiyi/MRI_GAN"
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data/A")
    PRE_DATA_ROOT = os.path.join(PROJECT_ROOT, "pre-data", "ts")
    
    # 中间输出目录（与7步流程对应）
    NII_DIR = os.path.join(PRE_DATA_ROOT, "01_nii")              # 步骤1: DICOM转NIfTI
    RESAMPLE_DIR = os.path.join(PRE_DATA_ROOT, "02_resample")   # 步骤2: 重采样到统一模板
    REG_DIR = os.path.join(PRE_DATA_ROOT, "03_registered")      # 步骤3: 配准
    BODY_MASK_DIR = os.path.join(PRE_DATA_ROOT, "body_masks")   # 步骤4: 身体掩码
    MASKED_DIR = os.path.join(PRE_DATA_ROOT, "04_masked")       # 步骤5: 应用掩码后
    NORM_DIR = os.path.join(PRE_DATA_ROOT, "05_normalized")     # 步骤6: 归一化后
    H5_DIR = os.path.join(PROJECT_ROOT, "train-data/images_tr")  # 步骤7: 最终h5 (训练数据目录)
    
    # 配准参数
    REG_PARAM = os.path.join(PROJECT_ROOT, "CE-MRI-synthesis", "preprocessing", "reg_series", "reg_param", "series_param.json")
    
    # 全局T1参考模板（用于统一所有数据的尺寸和spacing）
    # 设置为None时，使用第一个患者的T1作为模板
    GLOBLE_TEMPLETE = None  # 例如: "/path/to/template/T1.nii.gz"
    
    # 目标尺寸和spacing配置
    TARGET_SIZE = [512, 512, 32]  
    TARGET_SPACING = None 
    
    SERIES_MAP = {
        "AXT1": "T1",
        "AXT11+": "T1CE",
        "AXT1+1": "T1CE",
        "AXT1+": "T1CE",
        "AXT2": "T2",
        "AXT2FS": "T2",
        "AXPreT1": "PreT1",
        "DWI": "DWI",
        "ADC": "ADC",
    }


# ==================== 全局 badcase 管理 ====================
_badcases = set()

REQUIRED_SERIES = ["T1", "T1CE", "T2", "DWI", "ADC", "PreT1"]


def is_badcase(patient_id):
    """检查患者是否已在 badcase 中（前面步骤已报错）"""
    return patient_id in _badcases


def add_badcase(patient_id, reason=""):
    """将患者加入 badcase 集合并记录到 CSV，同时加入 done_case.csv"""
    if patient_id not in _badcases:
        _badcases.add(patient_id)
        log_badcase(patient_id, reason)
        mark_as_done(patient_id)


def check_series_complete(patient_dir, suffix=""):
    """检查患者目录中是否包含完整的6个序列
    
    Args:
        patient_dir: 患者目录路径
        suffix: 文件名后缀，如 "_w"
    
    Returns:
        (is_complete, missing_list)
    """
    missing = []
    for ser in REQUIRED_SERIES:
        fname = f"{ser}{suffix}.nii.gz" if suffix else f"{ser}.nii.gz"
        if not os.path.exists(os.path.join(patient_dir, fname)):
            missing.append(fname)
    return len(missing) == 0, missing


def print_header(text):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_step(step_num, text):
    """打印步骤"""
    print(f"\n[步骤 {step_num}] {text}")
    print("-" * 40)


def get_patient_list():
    """获取患者列表（排除 done_case.csv 中的患者）"""
    patient_dirs = [d for d in glob.glob(os.path.join(Config.DATA_ROOT, "*")) if os.path.isdir(d)]
    all_patients = sorted([os.path.basename(d) for d in patient_dirs])
    
    # 读取 done_case.csv
    done_cases = set()
    done_csv_path = os.path.join(Config.PRE_DATA_ROOT, "done_case.csv")
    if os.path.exists(done_csv_path):
        import csv
        with open(done_csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # 跳过表头
            for row in reader:
                if row:
                    done_cases.add(row[0].strip())
    
    # 过滤掉已完成的病例
    filtered_patients = [p for p in all_patients if p not in done_cases]
    print(f"数据目录总患者数: {len(all_patients)}, 已完成: {len(done_cases)}, 待处理: {len(filtered_patients)}")
    
    return filtered_patients


class DualOutput:
    """同时输出到终端和日志文件"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")
        start_line = f"\n{'='*60}\n[日志启动] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n"
        self.log.write(start_line)
        self.log.flush()
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        if self.log and not self.log.closed:
            end_line = f"\n[日志结束] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            self.log.write(end_line)
            self.log.close()


# ==================== 步骤1: DICOM转NIfTI ====================
def step1_dicom_to_nifti(patient_ids=None):
    """DICOM转NIfTI"""
    print_step(1, "DICOM转NIfTI")
    
    import SimpleITK as sitk
    
    os.makedirs(Config.NII_DIR, exist_ok=True)
    
    patients = patient_ids if patient_ids else get_patient_list()
    
    for i, pid in enumerate(patients, 1):
        print(f"\n[{i}/{len(patients)}] 处理患者: {pid}")
        
        patient_dicom = os.path.join(Config.DATA_ROOT, pid)
        patient_nii = os.path.join(Config.NII_DIR, pid)
        os.makedirs(patient_nii, exist_ok=True)
        
        # 构建清洗后的映射表（去掉下划线和空格）
        clean_series_map = {}
        for k, v in Config.SERIES_MAP.items():
            clean_key = k.replace("_", "").replace(" ", "")
            clean_series_map[clean_key] = v
        
        # 遍历患者目录下的实际文件夹
        if os.path.exists(patient_dicom):
            for entry in os.listdir(patient_dicom):
                dicom_dir = os.path.join(patient_dicom, entry)
                
                if not os.path.isdir(dicom_dir):
                    continue
                
                # 清洗目录名：去掉下划线和空格后进行匹配
                clean_entry = entry.replace("_", "").replace(" ", "")
                
                if clean_entry not in clean_series_map:
                    continue
                
                std_name = clean_series_map[clean_entry]
                
                # 检查DICOM文件
                dicom_files = glob.glob(os.path.join(dicom_dir, "IM*"))
                if len(dicom_files) == 0:
                    print(f"  ! {entry} 中没有找到DICOM文件(IM*)，跳过")
                    continue
                
                try:
                    # 读取DICOM序列
                    reader = sitk.ImageSeriesReader()
                    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
                    reader.SetFileNames(dicom_names)
                    image = reader.Execute()
                    
                    # 保存为NIfTI
                    output_file = os.path.join(patient_nii, f"{std_name}.nii.gz")
                    
                    # 处理重复名称
                    if os.path.exists(output_file):
                        print(f"  警告: {std_name} 已存在，跳过 {entry}")
                        continue
                    
                    sitk.WriteImage(image, output_file)
                    print(f"  ✓ {entry} -> {std_name}")
                    
                except Exception as e:
                    print(f"  ✗ {entry} 转换失败: {str(e)}")
        
        # 步骤1完整性检查：转换后检查是否包含6个完整序列
        is_complete, missing = check_series_complete(patient_nii)
        if not is_complete:
            print(f"  ✗ 步骤1序列不完整，缺失: {', '.join(missing)}，记录到badcase.csv")
            add_badcase(pid, f"步骤1序列不完整: {', '.join(missing)}")
        else:
            print(f"  ✓ 步骤1序列完整 (6/6)")
    
    print(f"\n✓ 步骤1完成，输出目录: {Config.NII_DIR}")


# ==================== 步骤2: 图像配准 ====================
def step2_registration(patient_ids=None):
    """基于T1序列配准其他序列（使用原始NIfTI）"""
    print_step(2, "图像配准 (ANTsPy)")
    
    from antspy_registration import get_series, reg_series, move_resampled_t1_to_regfolder
    import json
    
    os.makedirs(Config.REG_DIR, exist_ok=True)
    
    patients = patient_ids if patient_ids else get_patient_list()
    
    for i, pid in enumerate(patients, 1):
        if is_badcase(pid):
            print(f"\n[{i}/{len(patients)}] 处理患者: {pid}  ! 前面步骤已报错，跳过")
            continue
        
        print(f"\n[{i}/{len(patients)}] 处理患者: {pid}")
        
        patient_dir = os.path.join(Config.NII_DIR, pid)
        patient_reg_dir = os.path.join(Config.REG_DIR, pid)
        
        # 检查是否已处理完成（输出目录存在且6个模态齐全）
        if os.path.exists(patient_reg_dir):
            is_complete, missing = check_series_complete(patient_reg_dir, suffix="_w")
            if is_complete:
                print(f"  ✓ 已处理完成，跳过")
                continue
            else:
                print(f"  ! 输出目录存在但模态不完整，重新处理 (缺失: {', '.join(missing)})")
        
        os.makedirs(patient_reg_dir, exist_ok=True)
        
        if not os.path.exists(patient_dir):
            print(f"  ✗ 目录不存在")
            continue
        
        # 检查输入模态是否完整
        required_input = ["T1.nii.gz", "T2.nii.gz", "T1CE.nii.gz",
                          "DWI.nii.gz", "ADC.nii.gz", "PreT1.nii.gz"]
        missing_input = []
        for rf in required_input:
            if not os.path.exists(os.path.join(patient_dir, rf)):
                missing_input.append(rf)
        if missing_input:
            print(f"  ✗ 输入模态不完整，缺失: {', '.join(missing_input)}，跳过并记录到badcase.csv")
            add_badcase(pid, f"步骤2输入缺失模态: {', '.join(missing_input)}")
            continue
        
        try:
            # 为每个患者设置正确的输出目录环境变量
            os.environ['MRI_REG_DIR'] = patient_reg_dir
            
            # 获取序列
            series_dict = get_series(patient_dir, Config.REG_PARAM, after_resample_t1=False)
            
            # 配准
            series_dict = reg_series(series_dict)
            
            # 移动T1
            series_dict = move_resampled_t1_to_regfolder(series_dict)
            
            # 检查配准后输出是否完整
            required_output = ["T1_w.nii.gz", "T2_w.nii.gz", "T1CE_w.nii.gz",
                               "DWI_w.nii.gz", "ADC_w.nii.gz", "PreT1_w.nii.gz"]
            missing_output = []
            for rf in required_output:
                if not os.path.exists(os.path.join(patient_reg_dir, rf)):
                    missing_output.append(rf)
            if missing_output:
                print(f"  ✗ 配准后模态不完整，缺失: {', '.join(missing_output)}，记录到badcase.csv")
                add_badcase(pid, f"步骤2配准后缺失模态: {', '.join(missing_output)}")
            else:
                print(f"  ✓ 配准完成，6个模态齐全")
            
        except Exception as e:
            print(f"  ✗ 配准失败: {str(e)}")
            add_badcase(pid, f"步骤2配准失败: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    print(f"\n✓ 步骤2完成，输出目录: {Config.REG_DIR}")


# ==================== 步骤3: 将所有序列重采样到统一模板 ====================
def step3_resample_all_to_template(patient_ids=None):
    """将配准后的所有序列重采样到以患者T1为严格模板的统一空间
    
    目标尺寸: [512, 512, 32]
    
    核心逻辑:
    - 所有序列已配准对齐到T1坐标系，此处用Identity Transform安全重采样到统一网格
    - T1 通过插值充满 32 层，其他模态对齐到同一空间，确保层间一一对应
    """
    print_step(3, "所有序列重采样到统一模板（以配准后T1为模板）")
    
    import SimpleITK as sitk
    import numpy as np
    
    os.makedirs(Config.RESAMPLE_DIR, exist_ok=True)
    
    patients = patient_ids if patient_ids else get_patient_list()
    series_list = ["T1", "T1CE", "T2", "DWI", "ADC", "PreT1"]
    target_size = Config.TARGET_SIZE  # [512, 512, 32]
    
    print(f"目标尺寸: {target_size}")
    
    for i, pid in enumerate(patients, 1):
        if is_badcase(pid):
            print(f"\n[{i}/{len(patients)}] 处理患者: {pid}  ! 前面步骤已报错，跳过")
            continue
        
        patient_input = os.path.join(Config.REG_DIR, pid)
        patient_output = os.path.join(Config.RESAMPLE_DIR, pid)
        
        if not os.path.exists(patient_input):
            print(f"\n[{i}/{len(patients)}] 处理患者: {pid}  ✗ 输入目录不存在")
            continue
        
        # 检查是否已处理完成（输出目录存在且6个模态齐全）
        if os.path.exists(patient_output):
            is_complete, missing = check_series_complete(patient_output, suffix="_w")
            if is_complete:
                print(f"\n[{i}/{len(patients)}] 处理患者: {pid}  ✓ 已处理完成，跳过")
                continue
            else:
                print(f"\n[{i}/{len(patients)}] 处理患者: {pid}  ! 输出目录存在但模态不完整，重新处理 (缺失: {', '.join(missing)})")
        
        os.makedirs(patient_output, exist_ok=True)
        
        # 读取该患者配准后的T1作为空间坐标参考
        t1_file = os.path.join(patient_input, "T1_w.nii.gz")
        if not os.path.exists(t1_file):
            print(f"\n[{i}/{len(patients)}] 处理患者: {pid}  ✗ 未找到配准后的T1，跳过")
            continue
        
        t1_img = sitk.ReadImage(t1_file)
        t1_origin = np.array(t1_img.GetOrigin(), dtype=np.float64)
        t1_spacing = np.array(t1_img.GetSpacing(), dtype=np.float64)
        t1_size = np.array(t1_img.GetSize(), dtype=np.float64)
        
        # 基于T1自身物理范围计算目标spacing，使T1充满32层
        t1_phys_size = t1_size * t1_spacing
        if Config.TARGET_SPACING:
            target_spacing = np.array(Config.TARGET_SPACING, dtype=np.float64)
        else:
            target_spacing = t1_phys_size / np.array(target_size, dtype=np.float64)
        
        # 严格使用T1的origin和direction作为模板
        template_origin = t1_origin
        template_direction = t1_img.GetDirection()
        
        print(f"\n[{i}/{len(patients)}] 处理患者: {pid}")
        print(f"  T1尺寸: {tuple(t1_size.astype(int))}, T1间距: {tuple(t1_spacing)}")
        print(f"  T1物理范围: [{t1_phys_size[0]:.1f}, {t1_phys_size[1]:.1f}, {t1_phys_size[2]:.1f}] mm")
        print(f"  目标间距: [{target_spacing[0]:.4f}, {target_spacing[1]:.4f}, {target_spacing[2]:.4f}]")
        print(f"  模板原点: [{template_origin[0]:.2f}, {template_origin[1]:.2f}, {template_origin[2]:.2f}]")
        
        # 重采样函数 - 以T1为空间参考模板
        def resample_to_template(moving, target_size, target_spacing, target_origin, target_direction):
            resampler = sitk.ResampleImageFilter()
            resampler.SetSize(target_size)
            resampler.SetOutputSpacing(target_spacing.tolist())
            resampler.SetOutputOrigin(target_origin.tolist())
            resampler.SetOutputDirection(target_direction)
            resampler.SetOutputPixelType(sitk.sitkFloat32)
            resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
            resampler.SetInterpolator(sitk.sitkLinear)
            return resampler.Execute(moving)
        
        # 重采样所有配准后的序列
        for ser in series_list:
            ser_file = os.path.join(patient_input, f"{ser}_w.nii.gz")
            if not os.path.exists(ser_file):
                print(f"  ! {ser}_w 不存在，跳过")
                continue
            
            try:
                moving = sitk.ReadImage(ser_file)
                moving_arr = sitk.GetArrayFromImage(moving)
                orig_nonzero = np.count_nonzero(moving_arr)
                orig_sum = moving_arr.sum()
                
                # 所有模态已配准对齐，重采样到T1统一模板空间是安全的
                resampled = resample_to_template(
                    moving, target_size, target_spacing, template_origin, template_direction
                )
                resampled_arr = sitk.GetArrayFromImage(resampled)
                new_nonzero = np.count_nonzero(resampled_arr)
                new_sum = resampled_arr.sum()
                
                sitk.WriteImage(resampled, os.path.join(patient_output, f"{ser}_w.nii.gz"))
                
                print(f"  ✓ {ser}: {moving.GetSize()} -> {resampled.GetSize()}, "
                      f"nonzero: {orig_nonzero} -> {new_nonzero}, sum: {orig_sum:.0f} -> {new_sum:.0f}")
                
            except Exception as e:
                print(f"  ✗ {ser} 重采样失败: {str(e)}")
        
        # 保存该患者的模板信息
        template_path = os.path.join(Config.PROJECT_ROOT, "pre-data", "template", f"template_{pid}.nii.gz")
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        template_img_out = sitk.Image(target_size, sitk.sitkFloat32)
        template_img_out.SetSpacing(target_spacing.tolist())
        template_img_out.SetOrigin(template_origin.tolist())
        template_img_out.SetDirection(template_direction)
        sitk.WriteImage(template_img_out, template_path)
    
    print(f"\n✓ 步骤3完成，输出目录: {Config.RESAMPLE_DIR}")
    print(f"所有数据已统一为尺寸 {target_size}（以各患者配准后T1为严格模板，层间一一对应）")


# ==================== 步骤4: 基于T1获取身体掩码 ====================
def step4_generate_body_mask(patient_ids=None):
    """基于配准后的T1序列生成身体掩码"""
    print_step(4, "基于T1生成身体掩码")
    
    # 导入get_body_t1的函数
    sys.path.insert(0, os.path.join(Config.PROJECT_ROOT, "CE-MRI-synthesis", "preprocessing", "resemble_n_get_body"))
    from get_body_t1 import (
        fill_inter_3D, fill_inter_3D_with_wall,
        morph_operation, getmaxcomponent
    )
    import cv2
    import numpy as np
    import SimpleITK as sitk
    
    os.makedirs(Config.BODY_MASK_DIR, exist_ok=True)
    
    patients = patient_ids if patient_ids else get_patient_list()
    
    for i, pid in enumerate(patients, 1):
        if is_badcase(pid):
            print(f"\n[{i}/{len(patients)}] 处理患者: {pid}  ! 前面步骤已报错，跳过")
            continue
        
        print(f"\n[{i}/{len(patients)}] 处理患者: {pid}")
        
        # 从重采样后的目录读取T1（已统一为512x512x32）
        t1_file = os.path.join(Config.RESAMPLE_DIR, pid, "T1_w.nii.gz")
        mask_file = os.path.join(Config.BODY_MASK_DIR, f"{pid}_body.nii.gz")
        
        # 检查是否已处理完成
        if os.path.exists(mask_file):
            print(f"  ✓ 身体掩码已存在，跳过")
            continue
        
        if not os.path.exists(t1_file):
            print(f"  ✗ 未找到配准后的T1序列: {t1_file}")
            continue
        
        try:
            # 读取T1图像
            img = sitk.ReadImage(t1_file)
            img_array = sitk.GetArrayFromImage(img)
            print(f"  图像尺寸: {img.GetSize()}")
            
            # 阈值分割
            ret = 150
            new_img = (img_array > ret).astype(np.uint8)
            
            # 处理流程
            new_img = fill_inter_3D(new_img)
            new_img = sitk.GetImageFromArray(new_img)
            new_img = morph_operation(new_img, kernel_size=(7, 7), operation_type=cv2.MORPH_OPEN)
            new_img = getmaxcomponent(new_img, min_size=1e4, check_num=50)
            new_img = fill_inter_3D_with_wall(new_img, other_axis=True, wall_dim=1)
            new_img = sitk.GetImageFromArray(new_img)
            new_img = morph_operation(new_img, kernel_size=(11, 11), operation_type="erode")
            new_img = sitk.GetArrayFromImage(new_img)
            new_img = fill_inter_3D_with_wall(new_img, other_axis=True, wall_dim=2)
            new_img = sitk.GetImageFromArray(new_img)
            new_img = morph_operation(new_img, kernel_size=(9, 9), operation_type=cv2.MORPH_OPEN)
            new_img = getmaxcomponent(new_img, min_size=1e4, check_num=50)
            new_img = morph_operation(new_img, kernel_size=(11, 11), operation_type="dilate")
            new_img = morph_operation(new_img, kernel_size=(11, 11), operation_type=cv2.MORPH_CLOSE)
            new_img = morph_operation(new_img, kernel_size=(5, 5), operation_type="dilate")
            new_img = morph_operation(new_img, kernel_size=(5, 5), operation_type=cv2.MORPH_CLOSE)
            new_img = sitk.BinaryMorphologicalOpening(new_img, (3, 3, 3))
            
            # 保存
            save_path = os.path.join(Config.BODY_MASK_DIR, f"{pid}_body.nii.gz")
            sitk.WriteImage(new_img, save_path)
            print(f"  ✓ 身体掩码已保存: {save_path}")
            
        except Exception as e:
            print(f"  ✗ 处理失败: {str(e)}")
    
    print(f"\n✓ 步骤4完成，输出目录: {Config.BODY_MASK_DIR}")


# ==================== 步骤5: 应用身体掩码 ====================
def step5_apply_body_mask(patient_ids=None):
    """在身体掩码应用到所有配准后的序列"""
    print_step(5, "应用身体掩码到所有序列")
    
    import SimpleITK as sitk
    import numpy as np
    
    os.makedirs(Config.MASKED_DIR, exist_ok=True)
    
    patients = patient_ids if patient_ids else get_patient_list()
    
    for i, pid in enumerate(patients, 1):
        if is_badcase(pid):
            print(f"\n[{i}/{len(patients)}] 处理患者: {pid}  ! 前面步骤已报错，跳过")
            continue
        
        print(f"\n[{i}/{len(patients)}] 处理患者: {pid}")
        
        patient_reg = os.path.join(Config.RESAMPLE_DIR, pid)
        mask_file = os.path.join(Config.BODY_MASK_DIR, f"{pid}_body.nii.gz")
        patient_output = os.path.join(Config.MASKED_DIR, pid)
        
        if not os.path.exists(patient_reg):
            print(f"  ✗ 配准目录不存在: {patient_reg}")
            continue
        
        if not os.path.exists(mask_file):
            print(f"  ✗ 身体掩码不存在: {mask_file}")
            continue
        
        # 检查是否已处理完成（输出目录存在且6个模态齐全）
        if os.path.exists(patient_output):
            is_complete, missing = check_series_complete(patient_output, suffix="_w")
            if is_complete:
                print(f"  ✓ 已处理完成，跳过")
                continue
            else:
                print(f"  ! 输出目录存在但模态不完整，重新处理 (缺失: {', '.join(missing)})")
        
        os.makedirs(patient_output, exist_ok=True)
        
        try:
            # 读取身体掩码
            body = sitk.ReadImage(mask_file)
            body_mask = sitk.GetArrayFromImage(body)
            
            # 应用掩码到所有配准后的序列
            series_files = [f for f in os.listdir(patient_reg) if f.endswith('_w.nii.gz')]
            
            if not series_files:
                print(f"  ✗ 配准目录中没有配准后的图像")
                continue
            
            for filename in series_files:
                img_path = os.path.join(patient_reg, filename)
                img = sitk.ReadImage(img_path)
                img_array = sitk.GetArrayFromImage(img)
                
                # 应用掩码
                img_array = img_array * body_mask
                
                # 保存
                img_new = sitk.GetImageFromArray(img_array)
                img_new.CopyInformation(img)
                
                output_file = os.path.join(patient_output, filename)
                sitk.WriteImage(img_new, output_file)
                print(f"  ✓ {filename} 掩码已应用")
            
            # 复制身体掩码到输出目录
            sitk.WriteImage(body, os.path.join(patient_output, "body_mask.nii.gz"))
            print(f"  ✓ 身体掩码已应用完成")
            
        except Exception as e:
            print(f"  ✗ 处理失败: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    print(f"\n✓ 步骤5完成，输出目录: {Config.MASKED_DIR}")


def mark_as_done(patient_id):
    """将患者标记为已完成（加入 done_case.csv）"""
    done_csv_path = os.path.join(Config.PRE_DATA_ROOT, "done_case.csv")
    os.makedirs(os.path.dirname(done_csv_path), exist_ok=True)
    
    # 检查是否已存在
    if os.path.exists(done_csv_path):
        with open(done_csv_path, "r") as f:
            if patient_id in f.read():
                return
    
    with open(done_csv_path, "a") as f:
        f.write(f"{patient_id}\n")


def log_badcase(patient_id, reason=""):
    """记录bad case到CSV"""
    os.makedirs(Config.PRE_DATA_ROOT, exist_ok=True)
    csv_path = os.path.join(Config.PRE_DATA_ROOT, "badcase.csv")
    import csv
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["patient_id", "reason", "timestamp"])
        writer.writerow([patient_id, reason, time.strftime("%Y-%m-%d %H:%M:%S")])


# ==================== 步骤6: 去除空白切片+归一化 ====================
def step6_normalization(patient_ids=None):
    """去除不包含有效信息的切片并进行归一化"""
    print_step(6, "去除空白切片+归一化")
    
    import SimpleITK as sitk
    import numpy as np
    
    os.makedirs(Config.NORM_DIR, exist_ok=True)
    
    def find_valid_slice(array, mask):
        """找到包含有效信息的切片"""
        non_zero_array = array > 0
        valid_list = []
        for i in range(array.shape[0]):
            ratio = non_zero_array[i].sum() / (mask[i].sum() + 1)
            if ratio > 0.5:
                valid_list.append(i)
        return np.array(valid_list)
    
    patients = patient_ids if patient_ids else get_patient_list()
    
    for i, pid in enumerate(patients, 1):
        if is_badcase(pid):
            print(f"\n[{i}/{len(patients)}] 处理患者: {pid}  ! 前面步骤已报错，跳过")
            continue
        
        print(f"\n[{i}/{len(patients)}] 处理患者: {pid}")
        
        patient_masked = os.path.join(Config.MASKED_DIR, pid)
        
        if not os.path.exists(patient_masked):
            print(f"  ✗ 应用掩码后的目录不存在")
            continue
        
        patient_output = os.path.join(Config.NORM_DIR, pid)
        
        # 检查是否已处理完成（输出目录存在且6个模态齐全）
        if os.path.exists(patient_output):
            is_complete, missing = check_series_complete(patient_output, suffix="")
            if is_complete:
                print(f"  ✓ 已处理完成，跳过")
                continue
            else:
                print(f"  ! 输出目录存在但模态不完整，重新处理 (缺失: {', '.join(missing)})")
        
        os.makedirs(patient_output, exist_ok=True)
        
        # 检查6个模态是否完整
        required_files = ["T1_w.nii.gz", "T2_w.nii.gz", "T1CE_w.nii.gz",
                          "DWI_w.nii.gz", "ADC_w.nii.gz", "PreT1_w.nii.gz"]
        missing = []
        for rf in required_files:
            if not os.path.exists(os.path.join(patient_masked, rf)):
                missing.append(rf)
        if missing:
            print(f"  ✗ 模态不完整，缺失: {', '.join(missing)}，跳过并记录到badcase.csv")
            add_badcase(pid, f"步骤6缺失模态: {', '.join(missing)}")
            continue
        
        try:
            # 读取图像 (配准后的文件带_w后缀)
            files = {
                "T1": os.path.join(patient_masked, "T1_w.nii.gz"),
                "T2": os.path.join(patient_masked, "T2_w.nii.gz"),
                "T1CE": os.path.join(patient_masked, "T1CE_w.nii.gz"),
                "DWI": os.path.join(patient_masked, "DWI_w.nii.gz"),
                "ADC": os.path.join(patient_masked, "ADC_w.nii.gz"),
                "PreT1": os.path.join(patient_masked, "PreT1_w.nii.gz"),
                "body_mask": os.path.join(patient_masked, "body_mask.nii.gz"),
            }
            
            # 读取数组
            arrays = {}
            images = {}
            for key, filepath in files.items():
                if os.path.exists(filepath):
                    images[key] = sitk.ReadImage(filepath)
                    arrays[key] = sitk.GetArrayFromImage(images[key])
                else:
                    print(f"  ! 跳过不存在的文件: {key}")
            
            # 找到有效切片
            valid_slices = {}
            for key in ["T1", "T2", "T1CE", "DWI", "ADC", "PreT1"]:
                if key in arrays:
                    valid_slices[key] = find_valid_slice(arrays[key], arrays["body_mask"])
            
            if not valid_slices:
                print(f"  ✗ 没有有效切片")
                add_badcase(pid, "步骤6没有有效切片")
                continue
            
            # 检查每个模态是否有有效切片
            empty_slices = [k for k, v in valid_slices.items() if len(v) == 0]
            if empty_slices:
                print(f"  ✗ 以下模态没有有效切片: {', '.join(empty_slices)}")
                add_badcase(pid, f"步骤6以下模态没有有效切片: {', '.join(empty_slices)}")
                continue
            
            start_point = max([v.min() for v in valid_slices.values()])
            end_point = min([v.max() for v in valid_slices.values()])
            
            if start_point > end_point:
                print(f"  ✗ 切片后没有保留任何数据 (start={start_point}, end={end_point})")
                add_badcase(pid, f"步骤6切片后没有保留任何数据 (start={start_point}, end={end_point})")
                continue
            
            print(f"  有效切片范围: {start_point} - {end_point}")
            
            # 处理每个文件
            for key, img_o in images.items():
                img = sitk.GetArrayFromImage(img_o)
                
                # 归一化 (非掩码文件)
                if key != "body_mask":
                    # 截断负值（配准插值产生的负值伪影）
                    img = np.clip(img, 0, None)
                    # 使用百分位数clip，避免极端离群值压缩动态范围
                    nz = img[img > 0]
                    if len(nz) > 0:
                        upper = np.percentile(nz, 99.5)
                    else:
                        upper = img.max()
                    img[img > upper] = upper
                    img = ((img - img.min()) / (img.max() - img.min())) * 2 - 1
                
                # 裁剪 - 去除空白切片
                img = img[start_point:end_point + 1]
                
                # 保存
                img_new = sitk.GetImageFromArray(img)
                img_new.SetOrigin(img_o.GetOrigin())
                img_new.SetSpacing(img_o.GetSpacing())
                img_new.SetDirection(img_o.GetDirection())
                
                output_file = os.path.join(patient_output, f"{key}.nii.gz")
                sitk.WriteImage(img_new, output_file)
            
            print(f"  ✓ 归一化完成，去除空白切片后共 {end_point - start_point + 1} 层")
            
        except Exception as e:
            print(f"  ✗ 处理失败: {str(e)}")
            add_badcase(pid, f"步骤6处理失败: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    print(f"\n✓ 步骤6完成，输出目录: {Config.NORM_DIR}")


# ==================== 步骤7: 转换为h5格式 ====================
def step7_to_h5(patient_ids=None):
    """转换为h5格式"""
    print_step(7, "转换为h5格式")
    
    import SimpleITK as sitk
    import h5py
    
    os.makedirs(Config.H5_DIR, exist_ok=True)
    
    patients = patient_ids if patient_ids else get_patient_list()
    
    for i, pid in enumerate(patients, 1):
        if is_badcase(pid):
            print(f"\n[{i}/{len(patients)}] 处理患者: {pid}  ! 前面步骤已报错，跳过")
            continue
        
        print(f"\n[{i}/{len(patients)}] 处理患者: {pid}")
        
        patient_dir = os.path.join(Config.NORM_DIR, pid)
        patient_output = os.path.join(Config.H5_DIR, pid)
        
        if not os.path.exists(patient_dir):
            print(f"  ✗ 目录不存在")
            continue
        
        # 检查是否已处理完成（输出目录存在且有h5文件）
        if os.path.exists(patient_output):
            h5_files = [f for f in os.listdir(patient_output) if f.endswith('.h5')]
            if h5_files:
                print(f"  ✓ 已处理完成 ({len(h5_files)} layers)，跳过")
                mark_as_done(pid)  # 步骤7是最后一步，标记为完成
                continue
            else:
                print(f"  ! 输出目录存在但无h5文件，重新处理")
        
        os.makedirs(patient_output, exist_ok=True)
        
        # 检查6个模态是否完整
        required_files = ["T1.nii.gz", "T2.nii.gz", "T1CE.nii.gz",
                          "DWI.nii.gz", "ADC.nii.gz", "PreT1.nii.gz"]
        missing = []
        for rf in required_files:
            if not os.path.exists(os.path.join(patient_dir, rf)):
                missing.append(rf)
        if missing:
            print(f"  ✗ 模态不完整，缺失: {', '.join(missing)}，跳过并记录到badcase.csv")
            add_badcase(pid, f"步骤7缺失模态: {', '.join(missing)}")
            continue
        
        try:
            # 读取图像
            files = {
                "t1": os.path.join(patient_dir, "T1.nii.gz"),
                "t2": os.path.join(patient_dir, "T2.nii.gz"),
                "t1ce": os.path.join(patient_dir, "T1CE.nii.gz"),
                "dwi": os.path.join(patient_dir, "DWI.nii.gz"),
                "adc": os.path.join(patient_dir, "ADC.nii.gz"),
                "pret1": os.path.join(patient_dir, "PreT1.nii.gz"),
                "mask": os.path.join(patient_dir, "body_mask.nii.gz"),
            }
            
            arrays = {}
            for key, filepath in files.items():
                if os.path.exists(filepath):
                    dtype = sitk.sitkUInt8 if key == "mask" else sitk.sitkFloat32
                    arrays[key] = sitk.GetArrayFromImage(sitk.ReadImage(filepath, outputPixelType=dtype))
            
            if not arrays:
                print(f"  ✗ 没有有效图像")
                continue
            
            slice_num = list(arrays.values())[0].shape[0]
            
            # 保存为h5
            for layer in range(slice_num):
                h5_file_path = os.path.join(patient_output, f"layer_{layer}.h5")
                
                with h5py.File(h5_file_path, "w") as h5_file:
                    for key, arr in arrays.items():
                        h5_file[key] = arr[layer, :, :]
            
            print(f"  ✓ h5转换完成，共{slice_num}层")
            mark_as_done(pid)  # 标记为已完成
            
        except Exception as e:
            print(f"  ✗ 处理失败: {str(e)}")
    
    print(f"\n✓ 步骤7完成，输出目录: {Config.H5_DIR}")





# ==================== 主函数 ====================
def main():
    # 先解析参数（不重定向stdout）
    parser = argparse.ArgumentParser(description='MRI数据预处理总控脚本')
    parser.add_argument('--all', action='store_true', help='执行所有步骤')
    parser.add_argument('--steps', nargs='+', type=int, choices=range(1, 8),
                        help='执行指定步骤 (1-7)')
    parser.add_argument('--patients', nargs='+', default=None,
                        help='指定处理的患者ID (如: P000030031)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='跳过已存在的输出')
    parser.add_argument('--num', type=int, default=None,
                        help='只处理数据目录中的前num例患者')
    parser.add_argument('--data_root', type=str, default=None,
                        help='覆盖DICOM数据根目录 (如: /path/to/data/B/中心B)')
    parser.add_argument('--predata_root', type=str, default=None,
                        help='覆盖中间数据输出根目录 (如: /path/to/pre-data/B)')
    parser.add_argument('--h5_dir', type=str, default=None,
                        help='覆盖最终h5输出目录 (如: /path/to/train-data/images_ts_B)')
    
    args = parser.parse_args()
    
    # 覆盖路径配置
    if args.data_root:
        Config.DATA_ROOT = args.data_root
    if args.predata_root:
        Config.PRE_DATA_ROOT = args.predata_root
        Config.NII_DIR = os.path.join(args.predata_root, "01_nii")
        Config.RESAMPLE_DIR = os.path.join(args.predata_root, "02_resample")
        Config.REG_DIR = os.path.join(args.predata_root, "03_registered")
        Config.BODY_MASK_DIR = os.path.join(args.predata_root, "body_masks")
        Config.MASKED_DIR = os.path.join(args.predata_root, "04_masked")
        Config.NORM_DIR = os.path.join(args.predata_root, "05_normalized")
    if args.h5_dir:
        Config.H5_DIR = args.h5_dir
    
    # 设置日志目录和文件（路径配置已更新）
    log_dir = os.path.join(Config.PRE_DATA_ROOT, "log")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"preprocessing_{timestamp}.log")
    
    # 重定向 stdout 到终端+文件
    original_stdout = sys.stdout
    sys.stdout = DualOutput(log_file)
    dual_output = sys.stdout
    
    try:
        if args.all:
            steps = list(range(1, 8))
        elif args.steps:
            steps = args.steps
        else:
            print("请指定要执行的步骤:")
            print("  --all          执行所有步骤")
            print("  --steps 1 2 3  执行指定步骤")
            print("\n步骤说明:")
            print("  1: DICOM转NIfTI")
            print("  2: 基于T1序列配准其他序列")
            print("  3: 所有序列重采样到统一模板 [512,512,32]")
            print("  4: 基于T1序列获取身体掩码")
            print("  5: 应用身体掩码到所有序列")
            print("  6: 去除空白切片+归一化")
            print("  7: 转换为h5格式")
            return
        
        # 获取患者列表
        patient_list = args.patients if args.patients else get_patient_list()

        # 如果指定了--num参数，只取前num例
        if args.num is not None and args.num > 0:
            patient_list = patient_list[:args.num]

        print_header("MRI数据预处理总控脚本")
        print(f"执行步骤: {steps}")
        print(f"患者数量: {len(patient_list)}")
        print(f"数据根目录: {Config.DATA_ROOT}")
        print(f"中间输出目录: {Config.PRE_DATA_ROOT}")
        print(f"H5输出目录: {Config.H5_DIR}")
        print(f"项目根目录: {Config.PROJECT_ROOT}")
        
        start_time = time.time()
        
        # 执行步骤（7步流程）
        step_functions = {
            1: step1_dicom_to_nifti,
            2: step2_registration,
            3: step3_resample_all_to_template,
            4: step4_generate_body_mask,
            5: step5_apply_body_mask,
            6: step6_normalization,
            7: step7_to_h5,
        }
        
        for step in steps:
            if step in step_functions:
                step_functions[step](patient_list)
            else:
                print(f"未知步骤: {step}")
        
        elapsed_time = time.time() - start_time
        
        print_header("预处理完成")
        print(f"总耗时: {elapsed_time/60:.2f} 分钟")
        print(f"输出目录: {os.path.join(Config.PROJECT_ROOT, 'pre-data')}")
    
    finally:
        sys.stdout = original_stdout
        dual_output.close()


if __name__ == "__main__":
    main()
