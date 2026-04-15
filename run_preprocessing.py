#!/usr/bin/env python3
"""
MRI数据预处理总控脚本 - 简化版
基于现有代码结构，整合预处理流程

新流程顺序（7步）：
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
   - 步骤1-2 (DICOM转NIfTI + 重采样): python run_preprocessing.py --steps 1 2
   - 步骤3-5 (配准 + 身体掩码 + 应用): python run_preprocessing.py --steps 3 4 5
   - 步骤6-7 (归一化 + h5): python run_preprocessing.py --steps 6 7

"""

import argparse
import glob
import os
import sys
import time
import shutil
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/mnt/data/KASR/Dengsiyi/MRI_GAN/CE-MRI-synthesis')
sys.path.insert(0, '/mnt/data/KASR/Dengsiyi/MRI_GAN/CE-MRI-synthesis/reg_series')

# ==================== 配置 ====================
class Config:
    # 路径配置
    PROJECT_ROOT = "/mnt/data/KASR/Dengsiyi/MRI_GAN"
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
    
    # 中间输出目录（与7步流程对应）
    NII_DIR = os.path.join(PROJECT_ROOT, "pre-data", "01_nii")              # 步骤1: DICOM转NIfTI
    RESAMPLE_DIR = os.path.join(PROJECT_ROOT, "pre-data", "02_resample")   # 步骤2: 重采样到统一模板
    REG_DIR = os.path.join(PROJECT_ROOT, "pre-data", "03_registered")      # 步骤3: 配准
    BODY_MASK_DIR = os.path.join(PROJECT_ROOT, "pre-data", "body_masks")   # 步骤4: 身体掩码
    MASKED_DIR = os.path.join(PROJECT_ROOT, "pre-data", "04_masked")       # 步骤5: 应用掩码后
    NORM_DIR = os.path.join(PROJECT_ROOT, "pre-data", "05_normalized")     # 步骤6: 归一化后
    H5_DIR = os.path.join(PROJECT_ROOT, "train-data")                      # 步骤7: 最终h5 (训练数据目录)
    
    # 配准参数
    REG_PARAM = os.path.join(PROJECT_ROOT, "CE-MRI-synthesis", "reg_series", "reg_param", "series_param.json")
    
    # 全局T1参考模板（用于统一所有数据的尺寸和spacing）
    # 设置为None时，使用第一个患者的T1作为模板
    GLOBLE_TEMPLETE = None  # 例如: "/path/to/template/T1.nii.gz"
    
    # 目标尺寸和spacing配置
    TARGET_SIZE = [512, 512, 32]  
    TARGET_SPACING = None 
    
    SERIES_MAP = {
        "AX_T1": "T1",
        "AX_T1 1+": "T1CE",
        "AX_T1+": "T1CE",
        "AX_T2": "T2",
        "AX_PreT1": "PreT1",  
        "DWI": "DWI", 
        "ADC": "ADC",
    }


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
    """获取患者列表"""
    patient_dirs = [d for d in glob.glob(os.path.join(Config.DATA_ROOT, "P*")) if os.path.isdir(d)]
    return sorted([os.path.basename(d) for d in patient_dirs])


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
        
        # 遍历序列文件夹
        for folder_name, std_name in Config.SERIES_MAP.items():
            dicom_dir = os.path.join(patient_dicom, folder_name)
            
            if not os.path.exists(dicom_dir):
                continue
            
            # 检查DICOM文件
            dicom_files = glob.glob(os.path.join(dicom_dir, "IM*"))
            if len(dicom_files) == 0:
                continue
            
            try:
                # 读取DICOM序列
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
                reader.SetFileNames(dicom_names)
                image = reader.Execute()
                
                # 保存为NIfTI
                output_file = os.path.join(patient_nii, f"{std_name}.nii.gz")
                
                # 处理重复名称 (如AX_T1和AX_PreT1都映射到T1)
                if os.path.exists(output_file):
                    print(f"  警告: {std_name} 已存在，跳过 {folder_name}")
                    continue
                
                sitk.WriteImage(image, output_file)
                print(f"  ✓ {folder_name} -> {std_name}.nii.gz")
                
            except Exception as e:
                print(f"  ✗ {folder_name} 转换失败: {str(e)}")
    
    print(f"\n✓ 步骤1完成，输出目录: {Config.NII_DIR}")


# ==================== 步骤2: 将所有序列重采样到统一模板 ====================
def step2_resample_all_to_template(patient_ids=None):
    """将所有序列重采样到全局参考模板，统一尺寸和间距
    
    目标尺寸: [512, 512, 32]
    所有患者的所有序列都将统一到这个尺寸
    """
    print_step(2, "所有序列重采样到统一模板")
    
    import SimpleITK as sitk
    
    os.makedirs(Config.RESAMPLE_DIR, exist_ok=True)
    
    patients = patient_ids if patient_ids else get_patient_list()
    
    # 获取或创建参考模板
    template_img = None
    
    if Config.GLOBLE_TEMPLETE and os.path.exists(Config.GLOBLE_TEMPLETE):
        # 使用预设的全局模板
        template_img = sitk.ReadImage(Config.GLOBLE_TEMPLETE)
        print(f"使用预设全局模板: {Config.GLOBLE_TEMPLETE}")
    else:
        # 使用第一个患者的T1创建模板
        for pid in patients:
            template_file = os.path.join(Config.NII_DIR, pid, "T1.nii.gz")
            if os.path.exists(template_file):
                template_img = sitk.ReadImage(template_file)
                print(f"使用 {pid} 的T1创建参考模板")
                break
    
    if template_img is None:
        print("✗ 未找到模板T1")
        return
    
    # 创建标准模板：设置目标尺寸为 [512, 512, 32]
    target_size = Config.TARGET_SIZE  # [512, 512, 32]
    
    # 计算目标spacing，保持物理尺寸一致
    original_size = template_img.GetSize()
    original_spacing = template_img.GetSpacing()
    
    target_spacing = [
        original_size[0] * original_spacing[0] / target_size[0],
        original_size[1] * original_spacing[1] / target_size[1],
        original_size[2] * original_spacing[2] / target_size[2]
    ]
    
    # 如果配置了目标spacing，则使用配置的
    if Config.TARGET_SPACING:
        target_spacing = Config.TARGET_SPACING
    
    print(f"目标尺寸: {target_size}")
    print(f"目标间距: {target_spacing}")
    
    # 创建标准模板图像
    standard_template = sitk.Image(target_size, sitk.sitkFloat32)
    standard_template.SetSpacing(target_spacing)
    standard_template.SetOrigin(template_img.GetOrigin())
    standard_template.SetDirection(template_img.GetDirection())
    
    # 将原始模板重采样到标准尺寸
    resampler_template = sitk.ResampleImageFilter()
    resampler_template.SetSize(target_size)
    resampler_template.SetOutputSpacing(target_spacing)
    resampler_template.SetOutputOrigin(template_img.GetOrigin())
    resampler_template.SetOutputDirection(template_img.GetDirection())
    resampler_template.SetOutputPixelType(sitk.sitkFloat32)
    resampler_template.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler_template.SetInterpolator(sitk.sitkLinear)
    standard_template = resampler_template.Execute(template_img)
    
    # 保存标准模板
    template_path = os.path.join(Config.PROJECT_ROOT, "pre-data", "t1_template.nii.gz")
    sitk.WriteImage(standard_template, template_path)
    print(f"标准模板已保存: {template_path}")
    print(f"  模板尺寸: {standard_template.GetSize()}")
    print(f"  模板间距: {standard_template.GetSpacing()}")
    
    # 重采样函数 - 重采样到标准模板
    def resample_to_standard(moving, target_size, target_spacing, target_origin, target_direction):
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(target_size)
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetOutputOrigin(target_origin)
        resampler.SetOutputDirection(target_direction)
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(sitk.sitkLinear)
        return resampler.Execute(moving)
    
    # 需要处理的序列
    series_list = ["T1", "T1CE", "T2", "DWI", "ADC", "PreT1"]
    
    for i, pid in enumerate(patients, 1):
        print(f"\n[{i}/{len(patients)}] 处理患者: {pid}")
        
        patient_input = os.path.join(Config.NII_DIR, pid)
        patient_output = os.path.join(Config.RESAMPLE_DIR, pid)
        
        if not os.path.exists(patient_input):
            print(f"  ✗ 输入目录不存在")
            continue
        
        os.makedirs(patient_output, exist_ok=True)
        
        try:
            # 重采样所有序列到统一模板
            for ser in series_list:
                ser_file = os.path.join(patient_input, f"{ser}.nii.gz")
                if os.path.exists(ser_file):
                    moving = sitk.ReadImage(ser_file)
                    resampled = resample_to_standard(
                        moving, 
                        target_size, 
                        target_spacing,
                        standard_template.GetOrigin(),
                        standard_template.GetDirection()
                    )
                    sitk.WriteImage(resampled, os.path.join(patient_output, f"{ser}.nii.gz"))
                    print(f"  ✓ {ser} 重采样完成 -> {resampled.GetSize()}")
                else:
                    print(f"  ! {ser} 不存在，跳过")
            
        except Exception as e:
            print(f"  ✗ 处理失败: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    print(f"\n✓ 步骤2完成，输出目录: {Config.RESAMPLE_DIR}")
    print(f"所有数据已统一为尺寸 {target_size}")


# ==================== 步骤3: 图像配准 ====================
def step3_registration(patient_ids=None):
    """基于T1序列配准其他序列（所有数据已统一尺寸）"""
    print_step(3, "图像配准 (ANTsPy)")
    
    from antspy_registration import get_series, reg_series, move_resampled_t1_to_regfolder
    import json
    
    os.makedirs(Config.REG_DIR, exist_ok=True)
    
    patients = patient_ids if patient_ids else get_patient_list()
    
    for i, pid in enumerate(patients, 1):
        print(f"\n[{i}/{len(patients)}] 处理患者: {pid}")
        
        patient_dir = os.path.join(Config.RESAMPLE_DIR, pid)
        patient_reg_dir = os.path.join(Config.REG_DIR, pid)
        os.makedirs(patient_reg_dir, exist_ok=True)
        
        if not os.path.exists(patient_dir):
            print(f"  ✗ 目录不存在")
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
            
            print(f"  ✓ 配准完成")
            
        except Exception as e:
            print(f"  ✗ 配准失败: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    print(f"\n✓ 步骤3完成，输出目录: {Config.REG_DIR}")


# ==================== 步骤4: 基于T1获取身体掩码 ====================
def step4_generate_body_mask(patient_ids=None):
    """基于配准后的T1序列生成身体掩码"""
    print_step(4, "基于T1生成身体掩码")
    
    # 导入get_body_t1的函数
    sys.path.insert(0, os.path.join(Config.PROJECT_ROOT, "CE-MRI-synthesis", "resemble_n_get_body"))
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
        print(f"\n[{i}/{len(patients)}] 处理患者: {pid}")
        
        # 从配准后的目录读取T1
        t1_file = os.path.join(Config.REG_DIR, pid, "T1_w.nii.gz")
        
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
        print(f"\n[{i}/{len(patients)}] 处理患者: {pid}")
        
        patient_reg = os.path.join(Config.REG_DIR, pid)
        mask_file = os.path.join(Config.BODY_MASK_DIR, f"{pid}_body.nii.gz")
        patient_output = os.path.join(Config.MASKED_DIR, pid)
        
        if not os.path.exists(patient_reg):
            print(f"  ✗ 配准目录不存在: {patient_reg}")
            continue
        
        if not os.path.exists(mask_file):
            print(f"  ✗ 身体掩码不存在: {mask_file}")
            continue
        
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
        print(f"\n[{i}/{len(patients)}] 处理患者: {pid}")
        
        patient_masked = os.path.join(Config.MASKED_DIR, pid)
        
        if not os.path.exists(patient_masked):
            print(f"  ✗ 应用掩码后的目录不存在")
            continue
        
        patient_output = os.path.join(Config.NORM_DIR, pid)
        os.makedirs(patient_output, exist_ok=True)
        
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
                continue
            
            start_point = max([v.min() for v in valid_slices.values()])
            end_point = min([v.max() for v in valid_slices.values()])
            
            print(f"  有效切片范围: {start_point} - {end_point}")
            
            # 处理每个文件
            for key, img_o in images.items():
                img = sitk.GetArrayFromImage(img_o)
                
                # 归一化 (非掩码文件)
                if key != "body_mask":
                    upper = img.max() * 0.75
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
        print(f"\n[{i}/{len(patients)}] 处理患者: {pid}")
        
        patient_dir = os.path.join(Config.NORM_DIR, pid)
        patient_output = os.path.join(Config.H5_DIR, pid)
        
        if not os.path.exists(patient_dir):
            print(f"  ✗ 目录不存在")
            continue
        
        os.makedirs(patient_output, exist_ok=True)
        
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
            
        except Exception as e:
            print(f"  ✗ 处理失败: {str(e)}")
    
    print(f"\n✓ 步骤7完成，输出目录: {Config.H5_DIR}")





# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description='MRI数据预处理总控脚本')
    parser.add_argument('--all', action='store_true', help='执行所有步骤')
    parser.add_argument('--steps', nargs='+', type=int, choices=range(1, 8),
                        help='执行指定步骤 (1-7)')
    parser.add_argument('--patients', nargs='+', default=None,
                        help='指定处理的患者ID (如: P000030031)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='跳过已存在的输出')
    
    args = parser.parse_args()
    
    # 确定执行步骤
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
        print("  2: 所有序列重采样到统一模板 [512,512,32]")
        print("  3: 基于T1序列配准其他序列")
        print("  4: 基于T1序列获取身体掩码")
        print("  5: 应用身体掩码到所有序列")
        print("  6: 去除空白切片+归一化")
        print("  7: 转换为h5格式")
        return
    
    print_header("MRI数据预处理总控脚本")
    print(f"执行步骤: {steps}")
    print(f"患者列表: {args.patients if args.patients else '全部'}")
    print(f"项目根目录: {Config.PROJECT_ROOT}")
    
    start_time = time.time()
    
    # 执行步骤（7步流程）
    step_functions = {
        1: step1_dicom_to_nifti,
        2: step2_resample_all_to_template,
        3: step3_registration,
        4: step4_generate_body_mask,
        5: step5_apply_body_mask,
        6: step6_normalization,
        7: step7_to_h5,
    }
    
    for step in steps:
        if step in step_functions:
            step_functions[step](args.patients)
        else:
            print(f"未知步骤: {step}")
    
    elapsed_time = time.time() - start_time
    
    print_header("预处理完成")
    print(f"总耗时: {elapsed_time/60:.2f} 分钟")
    print(f"输出目录: {os.path.join(Config.PROJECT_ROOT, 'pre-data')}")


if __name__ == "__main__":
    main()
