#!/usr/bin/env python3
"""
MRI数据预处理总控脚本
整合从DICOM输入到最终h5格式的完整预处理流程

处理流程:
1. DICOM转NIfTI (dcm2niix)
2. 生成身体掩码 (从AX_T1)
3. T1重采样到统一模板
4. 所有序列重采样到T1空间
5. 图像配准 (ANTsPy)
6. 应用身体掩码到所有序列
7. 归一化和去除空白切片
8. 转换为h5格式

作者: AI Assistant
"""

import argparse
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk

# 添加项目路径
sys.path.insert(0, '/mnt/data/KASR/Dengsiyi/MRI_GAN/CE-MRI-synthesis/preprocessing/reg_series')
from antspy_registration import get_series, reg_series, make_log, move_resampled_t1_to_regfolder


# ==================== 配置参数 ====================
class Config:
    """预处理配置类"""
    # 根目录
    PROJECT_ROOT = "/mnt/data/KASR/Dengsiyi/MRI_GAN"
    
    # 输入数据目录 (DICOM格式)
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
    
    # 中间输出目录
    NII_OUTPUT = os.path.join(PROJECT_ROOT, "pre-data", "nii_raw")          # DICOM转NIfTI输出
    BODY_MASK_FOLDER = os.path.join(PROJECT_ROOT, "pre-data", "body_masks")  # 身体掩码保存
    MASKED_OUTPUT = os.path.join(PROJECT_ROOT, "pre-data", "nii_masked")     # 应用掩码后
    RESAMPLE_T1_FOLDER = os.path.join(PROJECT_ROOT, "pre-data", "resample_t1")  # T1重采样
    RESAMPLE_ALL_FOLDER = os.path.join(PROJECT_ROOT, "pre-data", "resample_all") # 所有序列重采样
    REG_OUTPUT = os.path.join(PROJECT_ROOT, "pre-data", "registered")        # 配准后
    NORM_OUTPUT = os.path.join(PROJECT_ROOT, "pre-data", "normalized")       # 归一化后
    
    # 最终输出目录
    H5_OUTPUT = os.path.join(PROJECT_ROOT, "pre-data", "h5_2d")              # h5格式输出
    
    # 配准参数文件
    REG_PARAM_FILE = os.path.join(PROJECT_ROOT, "CE-MRI-synthesis", "preprocessing", "reg_series", "reg_param", "series_param.json")
    
    # 模板图像 (用于T1重采样)
    TEMPLATE_T1 = None  # 将使用第一个患者的T1作为模板
    
    # 序列名称映射 (从DICOM文件夹名到标准名)
    SERIES_MAP = {
        "AX_T1": "T1",
        "AX_T1 1+": "T1CE",
        "AX_T1+": "T1CE",
        "AX_T2": "T2",
        "AX_PreT1": "T1",  # 预处理T1，如果没有AX_T1则使用
        "DWI": "B800",     # DWI -> B800
        "ADC": "ADC",
    }
    
    # 归一化模式: "01norm" (0-1归一化) 或 "stdnorm" (Z-score标准化)
    NORM_MODE = "01norm"


# ==================== 日志设置 ====================
def setup_logger(log_file):
    """设置日志记录器"""
    logger = logging.getLogger("preprocessing")
    logger.setLevel(logging.DEBUG)
    
    # 清除现有处理器
    logger.handlers = []
    
    # 文件处理器
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式化
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s]: %(message)s', '%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    
    return logger


# ==================== 步骤1: DICOM转NIfTI ====================
def dicom_to_nifti(patient_dir, output_dir, logger):
    """
    使用dcm2niix将DICOM转换为NIfTI格式
    
    Args:
        patient_dir: 患者DICOM数据目录
        output_dir: NIfTI输出目录
        logger: 日志记录器
    
    Returns:
        dict: 序列名到文件路径的映射
    """
    patient_id = os.path.basename(patient_dir)
    patient_output = os.path.join(output_dir, patient_id)
    os.makedirs(patient_output, exist_ok=True)
    
    series_files = {}
    
    # 遍历所有序列文件夹
    for series_folder in os.listdir(patient_dir):
        series_path = os.path.join(patient_dir, series_folder)
        if not os.path.isdir(series_path):
            continue
        
        # 检查是否有DICOM文件
        dicom_files = glob.glob(os.path.join(series_path, "IM*"))
        if len(dicom_files) == 0:
            continue
        
        # 获取标准序列名
        standard_name = None
        for key, value in Config.SERIES_MAP.items():
            if key in series_folder:
                standard_name = value
                break
        
        if not standard_name:
            logger.warning(f"  未知序列: {series_folder}, 跳过")
            continue
        
        # 读取DICOM序列
        try:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(series_path)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            
            # 保存为NIfTI
            output_file = os.path.join(patient_output, f"{standard_name}.nii.gz")
            sitk.WriteImage(image, output_file)
            series_files[standard_name] = output_file
            logger.info(f"  转换 {series_folder} -> {standard_name}.nii.gz")
            
        except Exception as e:
            logger.error(f"  转换失败 {series_folder}: {str(e)}")
    
    return series_files


# ==================== 步骤2: 生成身体掩码 ====================
def fill_inter_bone(mask):
    """对单层图像做孔洞填充"""
    mask = mask.astype(np.uint8)
    if np.sum(mask[:]) != 0:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_list = []
        for i in range(len(contours)):
            drawing = np.zeros_like(mask, np.uint8)
            img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
            contour_list.append(img_contour)
        mask = sum(contour_list)
        mask[mask >= 1] = 1
    return mask.astype(np.uint8)


def fill_inter_3d(mask, other_axis=True):
    """对3D图像做孔洞填充"""
    if not isinstance(mask, np.ndarray):
        mask = sitk.GetArrayFromImage(mask)
    mask_final = mask.copy()
    for i in range(mask.shape[0]):
        if np.max(mask[i, :, :]) > 0:
            mask_final[i, :, :] = fill_inter_bone(mask_final[i, :, :])
    if other_axis:
        for i in range(mask.shape[1]):
            if np.max(mask[:, i, :]) > 0:
                mask_final[:, i, :] = fill_inter_bone(mask_final[:, i, :])
        for i in range(mask.shape[2]):
            if np.max(mask[:, :, i]) > 0:
                mask_final[:, :, i] = fill_inter_bone(mask_final[:, :, i])
    return mask_final.astype(np.uint8)


def fill_inter_3d_with_wall(mask, other_axis=True, wall_dim=1):
    """带边界的孔洞填充"""
    if not isinstance(mask, np.ndarray):
        mask = sitk.GetArrayFromImage(mask)
    for ind in range(mask.shape[0]):
        if wall_dim == 1:
            mask[ind, :, 0] = 1
            mask[ind, :, -1] = 1
        else:
            mask[ind, 0, :] = 1
    return fill_inter_3d(mask, other_axis)


def morph_operation(img, kernel_size=(5, 5), operation_type=cv2.MORPH_OPEN):
    """形态学操作"""
    body_mask = sitk.GetArrayFromImage(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    mask_final = body_mask.copy()
    
    for i in range(body_mask.shape[0]):
        if np.max(body_mask[i, :, :]) > 0:
            if operation_type == "erode":
                mask_final[i, :, :] = cv2.erode(body_mask[i, :, :], kernel, iterations=1)
            elif operation_type == "dilate":
                mask_final[i, :, :] = cv2.dilate(body_mask[i, :, :], kernel, iterations=1)
            else:
                mask_final[i, :, :] = cv2.morphologyEx(body_mask[i, :, :], operation_type, kernel, iterations=1)
    
    return sitk.GetImageFromArray(mask_final.astype(np.uint8))


def get_max_component(mask_array, min_size=1e4, check_num=10000):
    """获取最大连通域"""
    if isinstance(mask_array, np.ndarray):
        mask_array = sitk.GetImageFromArray(mask_array)
    
    cca = sitk.ConnectedComponentImageFilter()
    cca.FullyConnectedOff()
    output_ex = cca.Execute(mask_array)
    labeled_img = sitk.GetArrayFromImage(output_ex)
    num = cca.GetObjectCount()
    
    if num <= check_num:
        check_num = num
    
    max_label = 1
    max_num = 0
    for i in range(1, check_num + 1):
        if np.sum(labeled_img == i) < min_size:
            continue
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    
    max_component = (labeled_img == max_label).astype(np.uint8)
    return sitk.GetImageFromArray(max_component)


def generate_body_mask(t1_file, output_file, logger):
    """
    从T1图像生成身体掩码
    
    Args:
        t1_file: T1 NIfTI文件路径
        output_file: 掩码输出路径
        logger: 日志记录器
    """
    try:
        # 读取图像
        img = sitk.ReadImage(t1_file)
        img_array = sitk.GetArrayFromImage(img)
        
        # 阈值分割
        ret = 150
        new_img = (img_array > ret).astype(np.uint8)
        
        # 填补孔洞
        new_img = fill_inter_3d(new_img)
        
        # 开运算去伪影
        new_img = sitk.GetImageFromArray(new_img)
        new_img = morph_operation(new_img, kernel_size=(7, 7), operation_type=cv2.MORPH_OPEN)
        
        # 取最大连通域
        new_img = get_max_component(new_img, min_size=1e4, check_num=50)
        
        # 填补孔洞
        new_img = fill_inter_3d_with_wall(new_img, other_axis=True, wall_dim=1)
        
        # 转sitk操作
        new_img = sitk.GetImageFromArray(new_img)
        
        # 腐蚀去细伪影
        new_img = morph_operation(new_img, kernel_size=(11, 11), operation_type="erode")
        new_img = sitk.GetArrayFromImage(new_img)
        new_img = fill_inter_3d_with_wall(new_img, other_axis=True, wall_dim=2)
        
        # 开运算去细伪影
        new_img = sitk.GetImageFromArray(new_img)
        new_img = morph_operation(new_img, kernel_size=(9, 9), operation_type=cv2.MORPH_OPEN)
        
        # 去除小连通域
        new_img = get_max_component(new_img, min_size=1e4, check_num=50)
        
        # 膨胀再闭运算
        new_img = morph_operation(new_img, kernel_size=(11, 11), operation_type="dilate")
        new_img = morph_operation(new_img, kernel_size=(11, 11), operation_type=cv2.MORPH_CLOSE)
        
        # 膨胀获得大一点的mask
        new_img = morph_operation(new_img, kernel_size=(5, 5), operation_type="dilate")
        
        # 闭开运算
        new_img = morph_operation(new_img, kernel_size=(5, 5), operation_type=cv2.MORPH_CLOSE)
        new_img = sitk.BinaryMorphologicalOpening(new_img, (3, 3, 3))
        
        # 保存
        sitk.WriteImage(new_img, output_file)
        logger.info(f"  身体掩码已保存: {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"  生成身体掩码失败: {str(e)}")
        return False


# ==================== 步骤3: 应用身体掩码 ====================
def apply_body_mask(patient_dir, mask_file, output_dir, logger):
    """
    将身体掩码应用到所有序列
    
    Args:
        patient_dir: 患者NIfTI目录
        mask_file: 身体掩码文件
        output_dir: 输出目录
        logger: 日志记录器
    """
    patient_id = os.path.basename(patient_dir)
    patient_output = os.path.join(output_dir, patient_id)
    os.makedirs(patient_output, exist_ok=True)
    
    try:
        # 读取掩码
        body = sitk.ReadImage(mask_file)
        body_mask = sitk.GetArrayFromImage(body)
        
        # 遍历所有nii文件
        for filename in os.listdir(patient_dir):
            if not filename.endswith('.nii.gz'):
                continue
            
            img_path = os.path.join(patient_dir, filename)
            img = sitk.ReadImage(img_path)
            img_array = sitk.GetArrayFromImage(img)
            
            # 应用掩码
            img_array = img_array * body_mask
            
            # 保存
            img_new = sitk.GetImageFromArray(img_array)
            img_new.CopyInformation(img)
            
            output_file = os.path.join(patient_output, filename)
            sitk.WriteImage(img_new, output_file)
        
        # 复制掩码文件
        shutil.copy(mask_file, os.path.join(patient_output, "body_mask.nii.gz"))
        
        logger.info(f"  身体掩码已应用到所有序列")
        return True
        
    except Exception as e:
        logger.error(f"  应用身体掩码失败: {str(e)}")
        return False


# ==================== 步骤4: T1重采样 ====================
def itk_resample(moving, target, resamplemethod=sitk.sitkLinear):
    """ITK重采样"""
    target_Size = [0, 0, 0]
    ori_size = moving.GetSize()
    ori_spacing = moving.GetSpacing()
    target_Spacing = target.GetSpacing()
    target_direction = moving.GetDirection()
    target_origin = moving.GetOrigin()
    
    target_Size[0] = round(ori_size[0] * ori_spacing[0] / target_Spacing[0])
    target_Size[1] = round(ori_size[1] * ori_spacing[1] / target_Spacing[1])
    target_Size[2] = round(ori_size[2] * ori_spacing[2] / target_Spacing[2])
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_Size)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputSpacing(target_Spacing)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    
    return resampler.Execute(moving)


def resample_t1(patient_dir, template_img, output_dir, logger):
    """
    将T1重采样到模板
    
    Args:
        patient_dir: 患者目录
        template_img: 模板图像
        output_dir: 输出目录
        logger: 日志记录器
    """
    patient_id = os.path.basename(patient_dir)
    patient_output = os.path.join(output_dir, patient_id)
    os.makedirs(patient_output, exist_ok=True)
    
    try:
        t1_file = os.path.join(patient_dir, "T1.nii.gz")
        if not os.path.exists(t1_file):
            logger.warning(f"  T1文件不存在")
            return False
        
        moving_image = sitk.ReadImage(t1_file)
        resampled_img = itk_resample(moving_image, template_img)
        
        output_file = os.path.join(patient_output, "T1.nii.gz")
        sitk.WriteImage(resampled_img, output_file)
        
        logger.info(f"  T1重采样完成")
        return True
        
    except Exception as e:
        logger.error(f"  T1重采样失败: {str(e)}")
        return False


# ==================== 步骤5: 所有序列重采样到T1 ====================
def resample_all_to_t1(patient_dir, output_dir, logger):
    """
    将所有序列重采样到T1空间
    
    Args:
        patient_dir: 患者目录
        output_dir: 输出目录
        logger: 日志记录器
    """
    patient_id = os.path.basename(patient_dir)
    patient_output = os.path.join(output_dir, patient_id)
    os.makedirs(patient_output, exist_ok=True)
    
    try:
        # 读取T1作为模板
        t1_file = os.path.join(patient_dir, "T1.nii.gz")
        if not os.path.exists(t1_file):
            logger.warning(f"  T1文件不存在")
            return False
        
        target_image = sitk.ReadImage(t1_file)
        
        # 需要重采样的序列
        series_list = ["T1", "T1CE", "T2", "B50", "B800", "B1400", "ADC"]
        
        for ser in series_list:
            ser_file = os.path.join(patient_dir, f"{ser}.nii.gz")
            if not os.path.exists(ser_file):
                continue
            
            moving_image = sitk.ReadImage(ser_file)
            resampled_img = itk_resample(moving_image, target_image)
            
            output_file = os.path.join(patient_output, f"{ser}.nii.gz")
            sitk.WriteImage(resampled_img, output_file)
        
        # 复制掩码
        mask_file = os.path.join(patient_dir, "body_mask.nii.gz")
        if os.path.exists(mask_file):
            shutil.copy(mask_file, os.path.join(patient_output, "body_mask.nii.gz"))
        
        logger.info(f"  所有序列重采样完成")
        return True
        
    except Exception as e:
        logger.error(f"  重采样失败: {str(e)}")
        return False


# ==================== 步骤6: 图像配准 ====================
def register_patient(patient_dir, param_file, output_dir, logger):
    """
    对患者进行图像配准
    
    Args:
        patient_dir: 患者目录
        param_file: 配准参数文件
        output_dir: 输出目录
        logger: 日志记录器
    """
    patient_id = os.path.basename(patient_dir)
    
    try:
        # 创建日志
        make_log(output_dir, repeat=False)
        
        # 获取序列
        series_dict = get_series(patient_dir, param_file, after_resample_t1=False)
        
        # 配准
        series_dict = reg_series(series_dict)
        
        # 移动重采样T1到配准文件夹
        series_dict = move_resampled_t1_to_regfolder(series_dict)
        
        # 保存JSON
        save_dir = os.path.dirname(series_dict["T1CE"]["filename_reg"])
        save_json_name = os.path.join(save_dir, "series_param.json")
        with open(save_json_name, "w") as f:
            json.dump(series_dict, f, indent=4)
        
        logger.info(f"  配准完成")
        return True
        
    except Exception as e:
        logger.error(f"  配准失败: {str(e)}")
        return False


# ==================== 步骤7: 归一化和去除空白切片 ====================
def find_valid_slice(array, mask):
    """找到有效切片"""
    non_zero_array = array > 0
    valid_list = []
    for i in range(array.shape[0]):
        ratio = non_zero_array[i].sum() / (mask[i].sum() + 1)
        if ratio > 0.5:
            valid_list.append(i)
    return np.array(valid_list)


def normalize_and_crop(patient_dir, output_dir, mode="01norm", logger=None):
    """
    归一化并去除空白切片
    
    Args:
        patient_dir: 患者目录
        output_dir: 输出目录
        mode: 归一化模式
        logger: 日志记录器
    """
    patient_id = os.path.basename(patient_dir)
    patient_output = os.path.join(output_dir, patient_id)
    os.makedirs(patient_output, exist_ok=True)
    
    try:
        # 文件列表
        file_list = {
            "T1": os.path.join(patient_dir, "T1_w.nii.gz"),
            "T2": os.path.join(patient_dir, "T2_w.nii.gz"),
            "T1CE": os.path.join(patient_dir, "T1CE_w.nii.gz"),
            "B50": os.path.join(patient_dir, "B50_w.nii.gz"),
            "B800": os.path.join(patient_dir, "B800_w.nii.gz"),
            "B1400": os.path.join(patient_dir, "B1400_w.nii.gz"),
            "ADC": os.path.join(patient_dir, "ADC_w.nii.gz"),
            "body_mask": os.path.join(patient_dir, "body_mask.nii.gz"),
        }
        
        # 检查文件是否存在
        for key, filepath in file_list.items():
            if not os.path.exists(filepath):
                if key != "B50" and key != "B1400":  # B50和B1400可能不存在
                    logger.warning(f"  {key} 文件不存在: {filepath}")
                    return False
        
        # 读取图像
        arrays = {}
        for key, filepath in file_list.items():
            if os.path.exists(filepath):
                arrays[key] = sitk.GetArrayFromImage(sitk.ReadImage(filepath))
            else:
                arrays[key] = None
        
        # 找到有效切片范围
        valid_slices = {}
        for key in ["T1", "T2", "T1CE", "B800", "ADC"]:
            if arrays[key] is not None:
                valid_slices[key] = find_valid_slice(arrays[key], arrays["body_mask"])
        
        # 计算共同的有效范围
        start_point = max([v.min() for v in valid_slices.values()])
        end_point = min([v.max() for v in valid_slices.values()])
        
        if (end_point - start_point) < 8:
            logger.warning(f"  有效切片少于8层")
        
        # 处理每个文件
        for key, filepath in file_list.items():
            if not os.path.exists(filepath):
                continue
            
            img_o = sitk.ReadImage(filepath)
            img = sitk.GetArrayFromImage(img_o)
            
            # 归一化 (掩码文件不归一化)
            if key != "body_mask":
                if mode == "stdnorm":
                    mean = img[img != 0].mean()
                    std = img[img != 0].std()
                    img = (img - mean) / std
                else:  # 01norm
                    upper_1onk = img.max() * 0.75
                    img[img > upper_1onk] = upper_1onk
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
        
        logger.info(f"  归一化和裁剪完成")
        return True
        
    except Exception as e:
        logger.error(f"  归一化失败: {str(e)}")
        return False


# ==================== 步骤8: 转换为h5格式 ====================
def convert_to_h5(patient_dir, output_dir, logger):
    """
    将NIfTI转换为h5格式 (2D切片)
    
    Args:
        patient_dir: 患者目录
        output_dir: 输出目录
        logger: 日志记录器
    """
    import h5py
    
    patient_id = os.path.basename(patient_dir)
    patient_output = os.path.join(output_dir, patient_id)
    os.makedirs(patient_output, exist_ok=True)
    
    try:
        # 读取图像
        t1 = os.path.join(patient_dir, "T1.nii.gz")
        t2 = os.path.join(patient_dir, "T2.nii.gz")
        b800 = os.path.join(patient_dir, "B800.nii.gz")
        t1ce = os.path.join(patient_dir, "T1CE.nii.gz")
        adc = os.path.join(patient_dir, "ADC.nii.gz")
        mask = os.path.join(patient_dir, "body_mask.nii.gz")
        
        # 检查文件
        for f in [t1, t2, b800, t1ce, adc, mask]:
            if not os.path.exists(f):
                logger.warning(f"  文件不存在: {f}")
                return False
        
        # 读取3D图像
        mask_array = sitk.GetArrayFromImage(sitk.ReadImage(mask, outputPixelType=sitk.sitkUInt8))
        slice_num = mask_array.shape[0]
        
        t1_array = sitk.GetArrayFromImage(sitk.ReadImage(t1, outputPixelType=sitk.sitkFloat32))
        t2_array = sitk.GetArrayFromImage(sitk.ReadImage(t2, outputPixelType=sitk.sitkFloat32))
        b800_array = sitk.GetArrayFromImage(sitk.ReadImage(b800, outputPixelType=sitk.sitkFloat32))
        t1ce_array = sitk.GetArrayFromImage(sitk.ReadImage(t1ce, outputPixelType=sitk.sitkFloat32))
        adc_array = sitk.GetArrayFromImage(sitk.ReadImage(adc, outputPixelType=sitk.sitkFloat32))
        
        # B50和B1400可能不存在
        b50_array = None
        b1500_array = None
        b50 = os.path.join(patient_dir, "B50.nii.gz")
        b1500 = os.path.join(patient_dir, "B1400.nii.gz")
        if os.path.exists(b50):
            b50_array = sitk.GetArrayFromImage(sitk.ReadImage(b50, outputPixelType=sitk.sitkFloat32))
        if os.path.exists(b1500):
            b1500_array = sitk.GetArrayFromImage(sitk.ReadImage(b1500, outputPixelType=sitk.sitkFloat32))
        
        # 保存为h5
        for layer in range(slice_num):
            h5_file_path = os.path.join(patient_output, f"layer_{layer}.h5")
            
            with h5py.File(h5_file_path, "w") as h5_file:
                h5_file["t1"] = t1_array[layer, :, :]
                h5_file["t2"] = t2_array[layer, :, :]
                h5_file["b800"] = b800_array[layer, :, :]
                h5_file["t1ce"] = t1ce_array[layer, :, :]
                h5_file["adc"] = adc_array[layer, :, :]
                h5_file["mask"] = mask_array[layer, :, :]
                if b50_array is not None:
                    h5_file["b50"] = b50_array[layer, :, :]
                if b1500_array is not None:
                    h5_file["b1500"] = b1500_array[layer, :, :]
        
        logger.info(f"  h5转换完成，共{slice_num}层")
        return True
        
    except Exception as e:
        logger.error(f"  h5转换失败: {str(e)}")
        return False


# ==================== 主流程 ====================
def process_patient(patient_id, steps, logger):
    """
    处理单个患者
    
    Args:
        patient_id: 患者ID
        steps: 要执行的步骤列表
        logger: 日志记录器
    """
    logger.info(f"========== 开始处理患者: {patient_id} ==========")
    
    patient_dicom = os.path.join(Config.DATA_ROOT, patient_id)
    
    # 步骤1: DICOM转NIfTI
    if 1 in steps:
        logger.info("步骤1: DICOM转NIfTI")
        dicom_to_nifti(patient_dicom, Config.NII_OUTPUT, logger)
    
    patient_nii = os.path.join(Config.NII_OUTPUT, patient_id)
    
    # 步骤2: 生成身体掩码
    if 2 in steps:
        logger.info("步骤2: 生成身体掩码")
        t1_file = os.path.join(patient_nii, "T1.nii.gz")
        if os.path.exists(t1_file):
            mask_file = os.path.join(Config.BODY_MASK_FOLDER, f"{patient_id}_body.nii.gz")
            generate_body_mask(t1_file, mask_file, logger)
        else:
            logger.warning("  T1文件不存在，跳过身体掩码生成")
    
    # 步骤3: T1重采样
    if 3 in steps:
        logger.info("步骤3: T1重采样")
        # 使用第一个患者的T1作为模板
        if Config.TEMPLATE_T1 is None:
            first_t1 = os.path.join(patient_nii, "T1.nii.gz")
            if os.path.exists(first_t1):
                Config.TEMPLATE_T1 = sitk.ReadImage(first_t1)
                logger.info(f"  使用 {patient_id} 的T1作为模板")
        
        if Config.TEMPLATE_T1 is not None:
            resample_t1(patient_nii, Config.TEMPLATE_T1, Config.RESAMPLE_T1_FOLDER, logger)
        else:
            logger.warning("  模板T1不存在，跳过")
    
    # 步骤4: 所有序列重采样到T1
    if 4 in steps:
        logger.info("步骤4: 所有序列重采样到T1")
        # 使用重采样后的T1目录
        patient_resample_t1 = os.path.join(Config.RESAMPLE_T1_FOLDER, patient_id)
        if os.path.exists(patient_resample_t1):
            # 复制其他序列到重采样T1目录
            for f in os.listdir(patient_nii):
                if f != "T1.nii.gz" and f.endswith('.nii.gz'):
                    src = os.path.join(patient_nii, f)
                    dst = os.path.join(patient_resample_t1, f)
                    if not os.path.exists(dst):
                        shutil.copy(src, dst)
            
            resample_all_to_t1(patient_resample_t1, Config.RESAMPLE_ALL_FOLDER, logger)
        else:
            logger.warning("  T1重采样目录不存在，跳过")
    
    # 步骤5: 图像配准
    if 5 in steps:
        logger.info("步骤5: 图像配准")
        patient_resample = os.path.join(Config.RESAMPLE_ALL_FOLDER, patient_id)
        if os.path.exists(patient_resample):
            register_patient(patient_resample, Config.REG_PARAM_FILE, Config.REG_OUTPUT, logger)
        else:
            logger.warning("  重采样目录不存在，跳过")
    
    # 步骤6: 应用身体掩码
    if 6 in steps:
        logger.info("步骤6: 应用身体掩码")
        mask_file = os.path.join(Config.BODY_MASK_FOLDER, f"{patient_id}_body.nii.gz")
        # 配准后的文件在registered目录的子目录中
        patient_reg = None
        for root, dirs, files in os.walk(Config.REG_OUTPUT):
            if patient_id in root:
                patient_reg = root
                break
        
        if os.path.exists(mask_file) and patient_reg and os.path.exists(patient_reg):
            apply_body_mask(patient_reg, mask_file, Config.MASKED_OUTPUT, logger)
        else:
            if not os.path.exists(mask_file):
                logger.warning("  身体掩码不存在，跳过")
            if not patient_reg or not os.path.exists(patient_reg):
                logger.warning("  配准目录不存在，跳过")
    
    # 步骤7: 归一化和去除空白切片
    if 7 in steps:
        logger.info("步骤7: 归一化和去除空白切片")
        patient_masked = os.path.join(Config.MASKED_OUTPUT, patient_id)
        
        if os.path.exists(patient_masked):
            normalize_and_crop(patient_masked, Config.NORM_OUTPUT, Config.NORM_MODE, logger)
        else:
            logger.warning("  应用掩码后的目录不存在，跳过")
    
    # 步骤8: 转换为h5格式
    if 8 in steps:
        logger.info("步骤8: 转换为h5格式")
        patient_norm = os.path.join(Config.NORM_OUTPUT, patient_id)
        if os.path.exists(patient_norm):
            convert_to_h5(patient_norm, Config.H5_OUTPUT, logger)
        else:
            logger.warning("  归一化目录不存在，跳过")
    
    logger.info(f"========== 患者 {patient_id} 处理完成 ==========\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MRI数据预处理总控脚本')
    parser.add_argument('--data_root', default=Config.DATA_ROOT, help='DICOM数据根目录')
    parser.add_argument('--output_root', default=Config.PROJECT_ROOT, help='输出根目录')
    parser.add_argument('--steps', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8],
                        help='要执行的步骤 (1-8)，默认执行所有步骤')
    parser.add_argument('--patients', nargs='+', default=None, help='指定处理的患者ID列表')
    parser.add_argument('--start', type=int, default=0, help='开始索引')
    parser.add_argument('--end', type=int, default=None, help='结束索引')
    parser.add_argument('--skip_existing', action='store_true', help='跳过已存在的输出')
    
    args = parser.parse_args()
    
    # 更新配置
    Config.DATA_ROOT = args.data_root
    Config.PROJECT_ROOT = args.output_root
    
    # 创建输出目录
    for folder in [Config.NII_OUTPUT, Config.BODY_MASK_FOLDER, Config.MASKED_OUTPUT,
                   Config.RESAMPLE_T1_FOLDER, Config.RESAMPLE_ALL_FOLDER, Config.REG_OUTPUT,
                   Config.NORM_OUTPUT, Config.H5_OUTPUT]:
        os.makedirs(folder, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(Config.PROJECT_ROOT, "preprocessing.log")
    logger = setup_logger(log_file)
    
    logger.info("=" * 60)
    logger.info("MRI数据预处理总控脚本启动")
    logger.info(f"数据根目录: {Config.DATA_ROOT}")
    logger.info(f"执行步骤: {args.steps}")
    logger.info("=" * 60)
    
    # 获取患者列表
    if args.patients:
        patient_list = args.patients
    else:
        patient_dirs = [d for d in glob.glob(os.path.join(Config.DATA_ROOT, "P*")) if os.path.isdir(d)]
        patient_list = [os.path.basename(d) for d in sorted(patient_dirs)]
    
    # 切片
    patient_list = patient_list[args.start:args.end]
    
    logger.info(f"共找到 {len(patient_list)} 个患者")
    logger.info("-" * 60)
    
    # 处理每个患者
    start_time = time.time()
    success_count = 0
    
    for i, patient_id in enumerate(patient_list, 1):
        try:
            process_patient(patient_id, args.steps, logger)
            success_count += 1
        except Exception as e:
            logger.error(f"处理患者 {patient_id} 时发生错误: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    elapsed_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"预处理完成!")
    logger.info(f"成功: {success_count}/{len(patient_list)}")
    logger.info(f"总耗时: {elapsed_time/60:.2f} 分钟")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
