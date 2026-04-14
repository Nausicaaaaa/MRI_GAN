#!/usr/bin/env python3
"""
MRI数据预处理总控脚本 - 简化版
基于现有代码结构，整合预处理流程

使用方式:
1. 完整流程: python run_preprocessing.py --all
2. 分步执行: 
   - 步骤1-2 (DICOM转NIfTI + 身体掩码): python run_preprocessing.py --steps 1 2
   - 步骤3-5 (重采样 + 配准): python run_preprocessing.py --steps 3 4 5
   - 步骤6-8 (应用掩码 + 归一化 + h5): python run_preprocessing.py --steps 6 7 8

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
    
    # 中间输出目录
    NII_DIR = os.path.join(PROJECT_ROOT, "pre-data", "01_nii")              # DICOM转NIfTI
    BODY_MASK_DIR = os.path.join(PROJECT_ROOT, "pre-data", "body_masks")   # 身体掩码
    MASKED_DIR = os.path.join(PROJECT_ROOT, "pre-data", "02_masked")       # 应用掩码后
    RESAMPLE_DIR = os.path.join(PROJECT_ROOT, "pre-data", "03_resample")   # 重采样后
    REG_DIR = os.path.join(PROJECT_ROOT, "pre-data", "04_registered")      # 配准后
    NORM_DIR = os.path.join(PROJECT_ROOT, "pre-data", "05_normalized")     # 归一化后
    H5_DIR = os.path.join(PROJECT_ROOT, "pre-data", "06_h5")               # 最终h5
    
    # 配准参数
    REG_PARAM = os.path.join(PROJECT_ROOT, "CE-MRI-synthesis", "reg_series", "reg_param", "series_param.json")
    
    # 序列映射 (文件夹名 -> 标准名)
    SERIES_MAP = {
        "AX_T1": "T1",
        "AX_T1 1+": "T1CE",
        "AX_T1+": "T1CE",
        "AX_T2": "T2",
        "AX_PreT1": "PreT1",  # 独立映射，不与AX_T1冲突
        "DWI": "DWI",  # 保持原名称
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


# ==================== 步骤2: 生成身体掩码 ====================
def step2_generate_body_mask(patient_ids=None):
    """生成身体掩码"""
    print_step(2, "生成身体掩码")
    
    # 导入get_body_t1的函数
    sys.path.insert(0, os.path.join(Config.PROJECT_ROOT, "CE-MRI-synthesis", "resemble_n_get_body"))
    from get_body_t1 import (
        read_dicom_series, fill_inter_3D, fill_inter_3D_with_wall,
        morph_operation, getmaxcomponent
    )
    import cv2
    import numpy as np
    import SimpleITK as sitk
    
    os.makedirs(Config.BODY_MASK_DIR, exist_ok=True)
    
    patients = patient_ids if patient_ids else get_patient_list()
    
    for i, pid in enumerate(patients, 1):
        print(f"\n[{i}/{len(patients)}] 处理患者: {pid}")
        
        # 从DICOM读取AX_T1
        dicom_dir = os.path.join(Config.DATA_ROOT, pid, "AX_T1")
        
        if not os.path.exists(dicom_dir):
            # 尝试AX_PreT1
            dicom_dir = os.path.join(Config.DATA_ROOT, pid, "AX_PreT1")
            if not os.path.exists(dicom_dir):
                print(f"  ✗ 未找到T1序列")
                continue
        
        try:
            # 读取DICOM
            img = read_dicom_series(dicom_dir)
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
    
    print(f"\n✓ 步骤2完成，输出目录: {Config.BODY_MASK_DIR}")


# ==================== 步骤3: T1重采样 ====================
def step3_resample_t1(patient_ids=None):
    """T1重采样"""
    print_step(3, "T1重采样")
    
    import SimpleITK as sitk
    
    os.makedirs(Config.RESAMPLE_DIR, exist_ok=True)
    
    patients = patient_ids if patient_ids else get_patient_list()
    
    # 使用第一个患者的T1作为模板
    template_img = None
    for pid in patients:
        template_file = os.path.join(Config.NII_DIR, pid, "T1.nii.gz")
        if os.path.exists(template_file):
            template_img = sitk.ReadImage(template_file)
            print(f"使用 {pid} 的T1作为模板")
            break
    
    if template_img is None:
        print("✗ 未找到模板T1")
        return
    
    # 重采样函数
    def resample(moving, target):
        target_Size = [0, 0, 0]
        ori_size = moving.GetSize()
        ori_spacing = moving.GetSpacing()
        target_Spacing = target.GetSpacing()
        
        target_Size[0] = round(ori_size[0] * ori_spacing[0] / target_Spacing[0])
        target_Size[1] = round(ori_size[1] * ori_spacing[1] / target_Spacing[1])
        target_Size[2] = round(ori_size[2] * ori_spacing[2] / target_Spacing[2])
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(target_Size)
        resampler.SetOutputDirection(moving.GetDirection())
        resampler.SetOutputOrigin(moving.GetOrigin())
        resampler.SetOutputSpacing(target_Spacing)
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(sitk.sitkLinear)
        
        return resampler.Execute(moving)
    
    for i, pid in enumerate(patients, 1):
        print(f"\n[{i}/{len(patients)}] 处理患者: {pid}")
        
        patient_input = os.path.join(Config.NII_DIR, pid)
        patient_output = os.path.join(Config.RESAMPLE_DIR, pid)
        
        if not os.path.exists(patient_input):
            print(f"  ✗ 输入目录不存在")
            continue
        
        os.makedirs(patient_output, exist_ok=True)
        
        try:
            # 重采样T1
            t1_file = os.path.join(patient_input, "T1.nii.gz")
            if os.path.exists(t1_file):
                moving = sitk.ReadImage(t1_file)
                resampled = resample(moving, template_img)
                sitk.WriteImage(resampled, os.path.join(patient_output, "T1.nii.gz"))
                print(f"  ✓ T1重采样完成")
            
            # 复制其他文件
            for f in os.listdir(patient_input):
                if f != "T1.nii.gz":
                    src = os.path.join(patient_input, f)
                    dst = os.path.join(patient_output, f)
                    shutil.copy(src, dst)
            
        except Exception as e:
            print(f"  ✗ 处理失败: {str(e)}")
    
    print(f"\n✓ 步骤3完成，输出目录: {Config.RESAMPLE_DIR}")


# ==================== 步骤4: 所有序列重采样到T1 ====================
def step4_resample_all(patient_ids=None):
    """所有序列重采样到T1空间"""
    print_step(4, "所有序列重采样到T1")
    
    import SimpleITK as sitk
    
    patients = patient_ids if patient_ids else get_patient_list()
    
    def resample_to_target(moving, target):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(target)
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(sitk.sitkLinear)
        return resampler.Execute(moving)
    
    for i, pid in enumerate(patients, 1):
        print(f"\n[{i}/{len(patients)}] 处理患者: {pid}")
        
        patient_dir = os.path.join(Config.RESAMPLE_DIR, pid)
        
        if not os.path.exists(patient_dir):
            print(f"  ✗ 目录不存在")
            continue
        
        try:
            # 读取T1作为目标
            t1_file = os.path.join(patient_dir, "T1.nii.gz")
            if not os.path.exists(t1_file):
                print(f"  ✗ T1文件不存在")
                continue
            
            target = sitk.ReadImage(t1_file)
            
            # 重采样其他序列
            series = ["T1CE", "T2", "DWI", "ADC", "PreT1"]
            for ser in series:
                ser_file = os.path.join(patient_dir, f"{ser}.nii.gz")
                if os.path.exists(ser_file):
                    moving = sitk.ReadImage(ser_file)
                    resampled = resample_to_target(moving, target)
                    sitk.WriteImage(resampled, ser_file)
                    print(f"  ✓ {ser} 重采样完成")
            
        except Exception as e:
            print(f"  ✗ 处理失败: {str(e)}")
    
    print(f"\n✓ 步骤4完成")


# ==================== 步骤5: 图像配准 ====================
def step5_registration(patient_ids=None):
    """图像配准"""
    print_step(5, "图像配准 (ANTsPy)")
    
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
    
    print(f"\n✓ 步骤5完成，输出目录: {Config.REG_DIR}")


# ==================== 步骤6: 应用身体掩码 ====================
def step6_apply_body_mask(patient_ids=None):
    """应用身体掩码到所有序列"""
    print_step(6, "应用身体掩码")
    
    import SimpleITK as sitk
    import numpy as np
    
    os.makedirs(Config.MASKED_DIR, exist_ok=True)
    
    patients = patient_ids if patient_ids else get_patient_list()
    
    for i, pid in enumerate(patients, 1):
        print(f"\n[{i}/{len(patients)}] 处理患者: {pid}")
        
        # 查找配准后的目录
        patient_reg = None
        for root, dirs, files in os.walk(Config.REG_DIR):
            if pid in os.path.basename(root):
                patient_reg = root
                break
        
        mask_file = os.path.join(Config.BODY_MASK_DIR, f"{pid}_body.nii.gz")
        patient_output = os.path.join(Config.MASKED_DIR, pid)
        
        if not patient_reg or not os.path.exists(patient_reg):
            print(f"  ✗ 配准目录不存在")
            continue
        
        if not os.path.exists(mask_file):
            print(f"  ✗ 身体掩码不存在")
            continue
        
        os.makedirs(patient_output, exist_ok=True)
        
        try:
            # 读取原始掩码
            body = sitk.ReadImage(mask_file)
            
            # 获取第一个配准后的图像作为参考（用于重采样掩码）
            ref_img = None
            for filename in os.listdir(patient_reg):
                if filename.endswith('.nii.gz'):
                    ref_img = sitk.ReadImage(os.path.join(patient_reg, filename))
                    break
            
            if ref_img is None:
                print(f"  ✗ 配准目录中没有图像")
                continue
            
            # 检查尺寸是否匹配，如果不匹配则重采样掩码
            body_size = body.GetSize()
            ref_size = ref_img.GetSize()
            
            if body_size != ref_size:
                print(f"  掩码尺寸 {body_size} 与目标尺寸 {ref_size} 不匹配，进行重采样")
                # 重采样掩码到目标尺寸，使用最近邻插值保持二值特性
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(ref_img)
                resampler.SetOutputPixelType(sitk.sitkUInt8)
                resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                body = resampler.Execute(body)
            
            body_mask = sitk.GetArrayFromImage(body)
            
            # 应用掩码到所有序列
            for filename in os.listdir(patient_reg):
                if not filename.endswith('.nii.gz'):
                    continue
                
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
            
            # 保存重采样后的掩码
            sitk.WriteImage(body, os.path.join(patient_output, "body_mask.nii.gz"))
            print(f"  ✓ 身体掩码已应用")
            
        except Exception as e:
            print(f"  ✗ 处理失败: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    print(f"\n✓ 步骤6完成，输出目录: {Config.MASKED_DIR}")


# ==================== 步骤7: 归一化 ====================
def step7_normalization(patient_ids=None):
    """归一化和去除空白切片"""
    print_step(7, "归一化和去除空白切片")
    
    import SimpleITK as sitk
    import numpy as np
    
    os.makedirs(Config.NORM_DIR, exist_ok=True)
    
    def find_valid_slice(array, mask):
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
            
            # 检查文件
            for key, filepath in files.items():
                if not os.path.exists(filepath):
                    print(f"  ✗ 文件不存在: {key}")
                    continue
            
            # 读取数组
            arrays = {}
            images = {}
            for key, filepath in files.items():
                if os.path.exists(filepath):
                    images[key] = sitk.ReadImage(filepath)
                    arrays[key] = sitk.GetArrayFromImage(images[key])
            
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
                
                # 裁剪
                img = img[start_point:end_point + 1]
                
                # 保存
                img_new = sitk.GetImageFromArray(img)
                img_new.SetOrigin(img_o.GetOrigin())
                img_new.SetSpacing(img_o.GetSpacing())
                img_new.SetDirection(img_o.GetDirection())
                
                output_file = os.path.join(patient_output, f"{key}.nii.gz")
                sitk.WriteImage(img_new, output_file)
            
            print(f"  ✓ 归一化完成")
            
        except Exception as e:
            print(f"  ✗ 处理失败: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    print(f"\n✓ 步骤7完成，输出目录: {Config.NORM_DIR}")


# ==================== 步骤8: 转h5 ====================
def step8_to_h5(patient_ids=None):
    """转换为h5格式"""
    print_step(8, "转换为h5格式")
    
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
    
    print(f"\n✓ 步骤8完成，输出目录: {Config.H5_DIR}")


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description='MRI数据预处理总控脚本')
    parser.add_argument('--all', action='store_true', help='执行所有步骤')
    parser.add_argument('--steps', nargs='+', type=int, choices=range(1, 9),
                        help='执行指定步骤 (1-8)')
    parser.add_argument('--patients', nargs='+', default=None,
                        help='指定处理的患者ID (如: P000030031)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='跳过已存在的输出')
    
    args = parser.parse_args()
    
    # 确定执行步骤
    if args.all:
        steps = list(range(1, 9))
    elif args.steps:
        steps = args.steps
    else:
        print("请指定要执行的步骤:")
        print("  --all          执行所有步骤")
        print("  --steps 1 2 3  执行指定步骤")
        print("\n步骤说明:")
        print("  1: DICOM转NIfTI")
        print("  2: 生成身体掩码")
        print("  3: T1重采样")
        print("  4: 所有序列重采样到T1")
        print("  5: 图像配准")
        print("  6: 应用身体掩码")
        print("  7: 归一化和去除空白切片")
        print("  8: 转换为h5格式")
        return
    
    print_header("MRI数据预处理总控脚本")
    print(f"执行步骤: {steps}")
    print(f"患者列表: {args.patients if args.patients else '全部'}")
    print(f"项目根目录: {Config.PROJECT_ROOT}")
    
    start_time = time.time()
    
    # 执行步骤
    step_functions = {
        1: step1_dicom_to_nifti,
        2: step2_generate_body_mask,
        3: step3_resample_t1,
        4: step4_resample_all,
        5: step5_registration,
        6: step6_apply_body_mask,
        7: step7_normalization,
        8: step8_to_h5,
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
