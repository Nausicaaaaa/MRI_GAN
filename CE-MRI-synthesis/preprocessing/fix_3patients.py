#!/usr/bin/env python3
"""
修复3例body mask过小的患者: P003619128, P003639406, P003702881

完整流程:
  Part 1 - 预处理修复 (Steps 1.1-1.4):
    1.1 重新生成 body mask (降低阈值 150->100)
    1.2 重新应用 body mask
    1.3 重新归一化
    1.4 重新生成 h5
  Part 2 - 推理 + 指标更新 + DICOM转换 (Steps 3-5)

使用方式:
  # 预处理修复 + 生成mask可视化(等待确认)
  python fix_3patients.py --part preprocess
  # 用户确认mask后，运行推理+指标+DICOM
  python fix_3patients.py --part inference
  # 运行全部(不推荐,应先确认mask)
  python fix_3patients.py --part all
"""

import argparse
import glob
import os
import shutil
import sys
import time

import cv2
import numpy as np
import SimpleITK as sitk
import h5py
import pandas as pd

# ==================== 配置 ====================
PROJECT_ROOT = "/mnt/data/KASR/Dengsiyi/MRI_GAN"
PATIENTS = ["P003619128", "P003639406", "P003702881"]

# 预处理路径
RESAMPLE_DIR = os.path.join(PROJECT_ROOT, "pre-data", "ts", "02_resample")
TS_BODY_MASK_DIR = os.path.join(PROJECT_ROOT, "pre-data", "ts", "body_masks")
TOP_BODY_MASK_DIR = os.path.join(PROJECT_ROOT, "pre-data", "body_masks")
MASKED_DIR = os.path.join(PROJECT_ROOT, "pre-data", "ts", "04_masked")
NORM_DIR = os.path.join(PROJECT_ROOT, "pre-data", "05_normalized")
H5_DIR = os.path.join(PROJECT_ROOT, "train-data", "images_ts")

# 推理路径
PRED_DIR = os.path.join(PROJECT_ROOT, "output_trans", "pred_nii")
METRICS_CSV = os.path.join(PROJECT_ROOT, "output_trans", "metrics", "all_patients_summary_metrics.csv")

# DICOM路径
DICOM_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_dicom_A")

# 可视化路径
VIS_MASK_DIR = os.path.join(PROJECT_ROOT, "vis", "mask")

# body mask 阈值策略: 自适应 (基于T1强度分布)
# 不再使用固定阈值, 因为不同患者T1信号差异很大
BODY_MASK_THRESHOLD_MODE = "adaptive"  # "adaptive" or "fixed"
BODY_MASK_THRESHOLD_FIXED = 100  # 仅fixed模式使用


# ==================== 形态学操作函数 (复制自 get_body_t1.py) ====================

def morph_operation(img, kernel_size=(5, 5), anchor=(-1, -1), operation_type=cv2.MORPH_OPEN):
    body_mask = sitk.GetArrayFromImage(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    mask_final = body_mask.copy()
    for i in range(body_mask.shape[0]):
        if np.max(body_mask[i, :, :]) > 0:
            if operation_type == "erode":
                mask_final[i, :, :] = cv2.erode(body_mask[i, :, :], kernel, anchor=anchor, iterations=1)
            elif operation_type == "dilate":
                mask_final[i, :, :] = cv2.dilate(body_mask[i, :, :], kernel, anchor=anchor, iterations=1)
            else:
                mask_final[i, :, :] = cv2.morphologyEx(body_mask[i, :, :], operation_type, kernel, iterations=1)
    mask_final = sitk.GetImageFromArray(mask_final.astype(np.uint8))
    return mask_final


def fill_inter_bone(mask):
    mask = mask.astype(np.uint8)
    if np.sum(mask[:]) != 0:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        len_contour = len(contours)
        contour_list = []
        for i in range(len_contour):
            drawing = np.zeros_like(mask, np.uint8)
            img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
            contour_list.append(img_contour)
        mask_fill = sum(contour_list)
        mask_fill[mask_fill >= 1] = 1
        return mask_fill.astype(np.uint8)
    return mask.astype(np.uint8)


def fill_inter_3D(mask, other_axis=True):
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


def fill_inter_3D_with_wall(mask, other_axis=True, wall_dim=1):
    if not isinstance(mask, np.ndarray):
        mask = sitk.GetArrayFromImage(mask)
    for ind in range(mask.shape[0]):
        if wall_dim == 1:
            mask[ind, :, 0] = 1
            mask[ind, :, -1] = 1
        else:
            mask[ind, 0, :] = 1
    return fill_inter_3D(mask, other_axis)


def getmaxcomponent(mask_array, min_size=1e4, check_num=10000, print_num=False, id_num=None):
    if isinstance(mask_array, np.ndarray):
        mask_array = sitk.GetImageFromArray(mask_array)
    cca = sitk.ConnectedComponentImageFilter()
    cca.FullyConnectedOff()
    _input = mask_array
    output_ex = cca.Execute(_input)
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
    if print_num:
        print(str(num) + '/' + str(max_label) + ':' + str(np.sum(labeled_img == max_label)))
    if np.sum(labeled_img == max_label) < min_size:
        print("Don't get the right component!! size:" + str(np.sum(labeled_img == max_label)))
    maxcomponent = (labeled_img == max_label).astype(np.uint8)
    maxcomponent = sitk.GetImageFromArray(maxcomponent)
    return maxcomponent


# ==================== Part 1: 预处理修复 ====================

def step1_1_generate_body_mask(pid):
    """1.1 重新生成 body mask (降低阈值)"""
    print(f"\n  [1.1] 生成 body mask: {pid}")
    t1_file = os.path.join(RESAMPLE_DIR, pid, "T1_w.nii.gz")
    if not os.path.exists(t1_file):
        print(f"    ✗ T1文件不存在: {t1_file}")
        return False

    try:
        img = sitk.ReadImage(t1_file)
        img_array = sitk.GetArrayFromImage(img)
        print(f"    图像尺寸: {img.GetSize()}")
        print(f"    T1信号: max={img_array.max():.1f}, mean={img_array.mean():.1f}, "
              f"nonzero_mean={img_array[img_array>0].mean():.1f}")

        # 自适应阈值: 基于T1强度分布的百分位阈值
        # 使用全部体素(含背景0)的百分位, 而非仅非零体素, 以正确分离背景与身体
        if BODY_MASK_THRESHOLD_MODE == "adaptive":
            p99 = np.percentile(img_array, 99)
            ret = max(int(p99 * 0.08), 5)  # P99的8%, 最低不低于5
            print(f"    自适应阈值: P99={p99:.1f}, threshold={ret}")
        else:
            ret = BODY_MASK_THRESHOLD_FIXED
            print(f"    固定阈值: {ret}")
        new_img = (img_array > ret).astype(np.uint8)

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

        # 保存到 ts/body_masks/
        os.makedirs(TS_BODY_MASK_DIR, exist_ok=True)
        save_path = os.path.join(TS_BODY_MASK_DIR, f"{pid}_body.nii.gz")
        sitk.WriteImage(new_img, save_path)
        print(f"    ✓ 已保存: {save_path}")

        # 同时覆盖 top-level body_masks/
        if os.path.exists(TOP_BODY_MASK_DIR):
            top_save = os.path.join(TOP_BODY_MASK_DIR, f"{pid}_body.nii.gz")
            sitk.WriteImage(new_img, top_save)
            print(f"    ✓ 已覆盖: {top_save}")

        # 统计mask覆盖率
        mask_arr = sitk.GetArrayFromImage(new_img)
        total_voxels = mask_arr.size
        fg_voxels = mask_arr.sum()
        print(f"    mask覆盖率: {fg_voxels/total_voxels*100:.1f}%")

        # 保存可视化到 vis/mask/ (所有层)
        try:
            vis_dir = VIS_MASK_DIR
            os.makedirs(vis_dir, exist_ok=True)
            # 清理旧的可视化文件
            import glob as glob_mod
            old_files = glob_mod.glob(os.path.join(vis_dir, f"{pid}_mask_*.png"))
            for old_f in old_files:
                os.remove(old_f)
            t1_arr = img_array
            mid = mask_arr.shape[0] // 2
            best_slice = np.argmax(mask_arr.sum(axis=(1,2)))
            num_slices = mask_arr.shape[0]
            saved_count = 0
            for slice_idx in range(num_slices):
                t1_slice = t1_arr[slice_idx].astype(np.float32)
                mask_slice = mask_arr[slice_idx]
                # 跳过全黑切片（无T1信号）
                if t1_slice.max() == 0:
                    continue
                # 归一化T1到0-255
                t1_norm = (t1_slice / t1_slice.max() * 255).astype(np.uint8)
                # 生成RGB叠加图: T1灰度 + mask红色轮廓
                rgb = np.stack([t1_norm, t1_norm, t1_norm], axis=-1)
                contour_mask = mask_slice.astype(np.uint8)
                contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # 填充mask区域(半透明红色)
                overlay = rgb.copy()
                overlay[mask_slice > 0] = [255, 0, 0]
                rgb = cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0)
                # 画轮廓线
                cv2.drawContours(rgb, contours, -1, (0, 255, 0), 2)
                # 标记特殊层
                suffix = ""
                if slice_idx == mid:
                    suffix = "_mid"
                elif slice_idx == best_slice:
                    suffix = "_best"
                vis_path = os.path.join(vis_dir, f"{pid}_mask_s{slice_idx:02d}{suffix}.png")
                cv2.imwrite(vis_path, rgb)
                saved_count += 1
            print(f"    ✓ 可视化已保存: {saved_count} 层到 {vis_dir}")
        except Exception as e:
            print(f"    ⚠ 可视化生成失败: {e}")

        return True

    except Exception as e:
        print(f"    ✗ 生成失败: {e}")
        import traceback; traceback.print_exc()
        return False


def step1_2_apply_body_mask(pid):
    """1.2 重新应用 body mask"""
    print(f"\n  [1.2] 应用 body mask: {pid}")
    patient_reg = os.path.join(RESAMPLE_DIR, pid)
    mask_file = os.path.join(TS_BODY_MASK_DIR, f"{pid}_body.nii.gz")
    patient_output = os.path.join(MASKED_DIR, pid)

    if not os.path.exists(patient_reg):
        print(f"    ✗ 重采样目录不存在: {patient_reg}")
        return False
    if not os.path.exists(mask_file):
        print(f"    ✗ 身体掩码不存在: {mask_file}")
        return False

    try:
        # 清空旧输出
        if os.path.exists(patient_output):
            shutil.rmtree(patient_output)
        os.makedirs(patient_output, exist_ok=True)

        body = sitk.ReadImage(mask_file)
        body_mask = sitk.GetArrayFromImage(body)

        series_files = [f for f in os.listdir(patient_reg) if f.endswith('_w.nii.gz')]
        if not series_files:
            print(f"    ✗ 配准目录中没有 _w.nii.gz 文件")
            return False

        for filename in series_files:
            img_path = os.path.join(patient_reg, filename)
            img = sitk.ReadImage(img_path)
            img_array = sitk.GetArrayFromImage(img)
            img_array = img_array * body_mask
            img_new = sitk.GetImageFromArray(img_array)
            img_new.CopyInformation(img)
            output_file = os.path.join(patient_output, filename)
            sitk.WriteImage(img_new, output_file)
            print(f"    ✓ {filename}")

        sitk.WriteImage(body, os.path.join(patient_output, "body_mask.nii.gz"))
        print(f"    ✓ body_mask.nii.gz")
        return True

    except Exception as e:
        print(f"    ✗ 应用失败: {e}")
        import traceback; traceback.print_exc()
        return False


def step1_3_normalize(pid):
    """1.3 重新归一化 (与 preprocessing.py step6 完全一致)"""
    print(f"\n  [1.3] 归一化: {pid}")
    patient_masked = os.path.join(MASKED_DIR, pid)
    patient_output = os.path.join(NORM_DIR, pid)

    if not os.path.exists(patient_masked):
        print(f"    ✗ masked目录不存在: {patient_masked}")
        return False

    def find_valid_slice(array, mask):
        non_zero_array = array > 0
        valid_list = []
        for i in range(array.shape[0]):
            ratio = non_zero_array[i].sum() / (mask[i].sum() + 1)
            if ratio > 0.5:
                valid_list.append(i)
        return np.array(valid_list)

    try:
        if os.path.exists(patient_output):
            shutil.rmtree(patient_output)
        os.makedirs(patient_output, exist_ok=True)

        # 检查文件完整性
        required_files = ["T1_w.nii.gz", "T2_w.nii.gz", "T1CE_w.nii.gz",
                          "DWI_w.nii.gz", "ADC_w.nii.gz", "PreT1_w.nii.gz"]
        for rf in required_files:
            fp = os.path.join(patient_masked, rf)
            if not os.path.exists(fp):
                print(f"    ✗ 缺失文件: {rf}")
                return False

        files = {
            "T1": os.path.join(patient_masked, "T1_w.nii.gz"),
            "T2": os.path.join(patient_masked, "T2_w.nii.gz"),
            "T1CE": os.path.join(patient_masked, "T1CE_w.nii.gz"),
            "DWI": os.path.join(patient_masked, "DWI_w.nii.gz"),
            "ADC": os.path.join(patient_masked, "ADC_w.nii.gz"),
            "PreT1": os.path.join(patient_masked, "PreT1_w.nii.gz"),
            "body_mask": os.path.join(patient_masked, "body_mask.nii.gz"),
        }

        arrays = {}
        images = {}
        for key, filepath in files.items():
            if os.path.exists(filepath):
                images[key] = sitk.ReadImage(filepath)
                arrays[key] = sitk.GetArrayFromImage(images[key])

        # 找有效切片
        valid_slices = {}
        for key in ["T1", "T2", "T1CE", "DWI", "ADC", "PreT1"]:
            if key in arrays:
                valid_slices[key] = find_valid_slice(arrays[key], arrays["body_mask"])

        start_point = max([v.min() for v in valid_slices.values()])
        end_point = min([v.max() for v in valid_slices.values()])
        print(f"    有效切片范围: {start_point} - {end_point}")

        for key, img_o in images.items():
            img = sitk.GetArrayFromImage(img_o)

            if key != "body_mask":
                img = np.clip(img, 0, None)
                nz = img[img > 0]
                if len(nz) > 0:
                    upper = np.percentile(nz, 99.5)
                else:
                    upper = img.max()
                img[img > upper] = upper
                img = ((img - img.min()) / (img.max() - img.min())) * 2 - 1

            img = img[start_point:end_point + 1]

            img_new = sitk.GetImageFromArray(img)
            img_new.SetOrigin(img_o.GetOrigin())
            img_new.SetSpacing(img_o.GetSpacing())
            img_new.SetDirection(img_o.GetDirection())

            output_file = os.path.join(patient_output, f"{key}.nii.gz")
            sitk.WriteImage(img_new, output_file)

        print(f"    ✓ 归一化完成, 共 {end_point - start_point + 1} 层")
        return True

    except Exception as e:
        print(f"    ✗ 归一化失败: {e}")
        import traceback; traceback.print_exc()
        return False


def step1_4_to_h5(pid):
    """1.4 重新生成 h5"""
    print(f"\n  [1.4] 生成 h5: {pid}")
    patient_dir = os.path.join(NORM_DIR, pid)
    patient_output = os.path.join(H5_DIR, pid)

    if not os.path.exists(patient_dir):
        print(f"    ✗ 归一化目录不存在: {patient_dir}")
        return False

    try:
        if os.path.exists(patient_output):
            shutil.rmtree(patient_output)
        os.makedirs(patient_output, exist_ok=True)

        files = {
            "t1": os.path.join(patient_dir, "T1.nii.gz"),
            "t2": os.path.join(patient_dir, "T2.nii.gz"),
            "t1ce": os.path.join(patient_dir, "T1CE.nii.gz"),
            "dwi": os.path.join(patient_dir, "DWI.nii.gz"),
            "adc": os.path.join(patient_dir, "ADC.nii.gz"),
            "pret1": os.path.join(patient_dir, "PreT1.nii.gz"),
            "mask": os.path.join(patient_dir, "body_mask.nii.gz"),
        }

        for key, fp in files.items():
            if not os.path.exists(fp):
                print(f"    ✗ 缺失: {fp}")
                return False

        arrays = {}
        for key, fp in files.items():
            if key == "mask":
                arrays[key] = sitk.GetArrayFromImage(sitk.ReadImage(fp, outputPixelType=sitk.sitkUInt8))
            else:
                arrays[key] = sitk.GetArrayFromImage(sitk.ReadImage(fp, outputPixelType=sitk.sitkFloat32))

        slice_num = arrays["mask"].shape[0]
        for layer in range(slice_num):
            h5_file_path = os.path.join(patient_output, f"layer_{layer}.h5")
            with h5py.File(h5_file_path, "w") as h5_file:
                for key in arrays:
                    h5_file[key] = arrays[key][layer, :, :]

        print(f"    ✓ h5完成, 共 {slice_num} 层")
        return True

    except Exception as e:
        print(f"    ✗ h5失败: {e}")
        import traceback; traceback.print_exc()
        return False


def run_preprocessing():
    """运行 Part 1: 预处理修复"""
    print("=" * 60)
    print("  Part 1: 预处理修复 (mask + 应用 + 归一化 + h5)")
    print("=" * 60)

    for pid in PATIENTS:
        print(f"\n{'='*50}")
        print(f"  处理患者: {pid}")
        print(f"{'='*50}")

        ok = step1_1_generate_body_mask(pid)
        if not ok:
            print(f"  ✗ {pid} body mask 生成失败, 跳过后续步骤")
            continue

        ok = step1_2_apply_body_mask(pid)
        if not ok:
            print(f"  ✗ {pid} 应用 mask 失败, 跳过后续步骤")
            continue

        ok = step1_3_normalize(pid)
        if not ok:
            print(f"  ✗ {pid} 归一化失败, 跳过后续步骤")
            continue

        ok = step1_4_to_h5(pid)
        if not ok:
            print(f"  ✗ {pid} h5生成失败")
            continue

        print(f"\n  ✓ {pid} 预处理修复完成")

    print("\n✓ Part 1 预处理修复全部完成")
    print(f"\n{'='*60}")
    print(f"  ★ 请检查 mask 可视化: {VIS_MASK_DIR}")
    print(f"  ★ 确认无误后再运行: python fix_3patients.py --part inference")
    print(f"{'='*60}")


# ==================== Part 2: 推理 + 指标 + DICOM ====================

def run_inference():
    """运行 Part 2: 推理 + 指标更新 + DICOM转换"""
    import torch
    import torchmetrics
    import lightning.pytorch as pl
    from lightning.pytorch import seed_everything
    from monai.utils import set_determinism

    sys.path.insert(0, os.path.join(PROJECT_ROOT, "CE-MRI-synthesis"))
    # 确保能正确导入
    os.chdir(os.path.join(PROJECT_ROOT, "CE-MRI-synthesis", "inference"))
    from inference.test_param import config
    from training_project.trainer_pix2pix_mulD import Pix2Pix_2d_MulD

    torch.multiprocessing.set_sharing_strategy('file_system')
    set_determinism(2023)
    seed_everything(2023, workers=True)

    print("=" * 60)
    print("  Part 2: 推理 + 指标更新 + DICOM转换")
    print("=" * 60)

    # --- Step 3: 推理 (仅3例) ---
    print("\n[Step 3] 推理 (仅3例患者)...")
    ckpt_path = os.path.join(PROJECT_ROOT, "results",
                             "CE_MRI_simulate_PCa_1_trans_fold5-1",
                             "checkpoint", "best-epoch=39.ckpt")
    if not os.path.exists(ckpt_path):
        print(f"✗ 未找到checkpoint: {ckpt_path}")
        return
    output_dir = os.path.join(PROJECT_ROOT, "output_trans")

    print(f"  Checkpoint: {ckpt_path}")

    # 加载模型
    target_device = f"cuda:{config.cuda_idx_list[0]}"
    model = Pix2Pix_2d_MulD.load_from_checkpoint(
        ckpt_path,
        map_location={"cuda:0": target_device},
        weights_only=False
    )

    config.filepath_img = os.path.join(PROJECT_ROOT, config.gt_data_dir)
    config.h5_2d_img_dir = os.path.join(PROJECT_ROOT, os.path.dirname(config.test_data_dir))

    model.template_dir = config.filepath_img
    model.data_dir = config.h5_2d_img_dir

    # 创建临时目录，只包含3例患者的h5数据，避免跑全部155个患者
    import tempfile
    tmp_test_root = os.path.join(PROJECT_ROOT, "_tmp_test_3patients")
    if os.path.exists(tmp_test_root):
        shutil.rmtree(tmp_test_root)
    os.makedirs(tmp_test_root, exist_ok=True)
    real_test_dir = os.path.join(PROJECT_ROOT, config.test_data_dir)
    for pid in PATIENTS:
        src = os.path.join(real_test_dir, pid)
        dst = os.path.join(tmp_test_root, pid)
        if os.path.exists(src):
            os.symlink(src, dst)
        else:
            print(f"  ✗ h5数据不存在: {src}")
    model.test_dir = tmp_test_root
    print(f"  临时测试目录: {tmp_test_root} (仅{len(PATIENTS)}例)")

    model.pred_result_dir = os.path.join(output_dir, "pred_nii")
    os.makedirs(model.pred_result_dir, exist_ok=True)

    print(f"  template_dir: {model.template_dir}")
    print(f"  test_dir: {model.test_dir}")
    print(f"  pred_result_dir: {model.pred_result_dir}")

    torch.set_float32_matmul_precision('high')
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[config.cuda_idx_list[0]],
        enable_progress_bar=True,
    )

    predictions = trainer.predict(model)
    print("  ✓ 推理完成")

    # 清理临时目录
    if os.path.exists(tmp_test_root):
        shutil.rmtree(tmp_test_root)
        print(f"  ✓ 临时目录已清理: {tmp_test_root}")

    # --- Step 4: 重新计算指标并更新 CSV ---
    print("\n[Step 4] 更新指标...")
    update_metrics(model.template_dir, model.pred_result_dir)

    # --- Step 5: 重新生成 DICOM ---
    print("\n[Step 5] 生成 DICOM...")
    generate_dicom_for_3patients()


def update_metrics(template_dir, pred_result_dir):
    """重新计算3例患者的指标并更新CSV"""
    from inference.inference_2d_main import compute_patient_metrics

    if not os.path.exists(METRICS_CSV):
        print(f"  ✗ CSV不存在: {METRICS_CSV}")
        return

    df = pd.read_csv(METRICS_CSV)

    updated_count = 0
    for pid in PATIENTS:
        gt_path = os.path.join(template_dir, pid, "T1CE.nii.gz")
        pred_path = os.path.join(pred_result_dir, f"{pid}_pred.nii.gz")
        mask_path = os.path.join(template_dir, pid, "body_mask.nii.gz")

        if not os.path.exists(gt_path):
            print(f"  ✗ GT不存在: {gt_path}")
            continue
        if not os.path.exists(pred_path):
            print(f"  ✗ Pred不存在: {pred_path}")
            continue

        print(f"  计算指标: {pid}...")
        metrics = compute_patient_metrics(
            gt_path, pred_path,
            mask_path if os.path.exists(mask_path) else None
        )

        # 更新DataFrame中的行
        idx = df.index[df['patient_id'] == pid]
        if len(idx) > 0:
            for col in ['ms_ssim', 'mi', 'nrmse', 'smape', 'logac', 'medsymac']:
                df.at[idx[0], col] = metrics[col]
            updated_count += 1
            print(f"    ✓ ms_ssim={metrics['ms_ssim']:.4f}, mi={metrics['mi']:.4f}, nrmse={metrics['nrmse']:.4f}")
        else:
            print(f"  ⚠ {pid} 不在CSV中, 添加新行")
            new_row = {'patient_id': pid}
            new_row.update(metrics)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            updated_count += 1

    if updated_count > 0:
        # 重新计算 MEAN 行
        mean_idx = df.index[df['patient_id'] == 'MEAN']
        numeric_cols = ['ms_ssim', 'mi', 'nrmse', 'smape', 'logac', 'medsymac']
        data_rows = df[df['patient_id'] != 'MEAN']

        if len(mean_idx) > 0:
            for col in numeric_cols:
                df.at[mean_idx[0], col] = data_rows[col].mean()
        else:
            mean_row = {'patient_id': 'MEAN'}
            for col in numeric_cols:
                mean_row[col] = data_rows[col].mean()
            df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

        df.to_csv(METRICS_CSV, index=False)
        print(f"  ✓ CSV已更新: {METRICS_CSV}")
        print(f"  更新了 {updated_count} 例患者")
    else:
        print("  ⚠ 没有更新任何指标")


# ==================== DICOM 转换 ====================

def nifti_to_dicom_series(nii_path, output_dicom_dir, patient_id="P001",
                          study_uid=None, series_uid=None, series_number=1,
                          frame_ref_uid=None, series_desc="MRI",
                          scale_min=None, scale_max=None,
                          window_center=None, window_width=None,
                          mask_nii_path=None):
    """将NIfTI转换为DICOM序列"""
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import generate_uid as dicom_generate_uid
    from datetime import datetime

    os.makedirs(output_dicom_dir, exist_ok=True)

    nii_image = sitk.ReadImage(nii_path)
    nii_array = sitk.GetArrayFromImage(nii_image)
    num_slices = nii_array.shape[0]

    foreground = None
    if mask_nii_path is not None and os.path.exists(mask_nii_path):
        mask_array = sitk.GetArrayFromImage(sitk.ReadImage(mask_nii_path))
        foreground = nii_array[mask_array > 0]
        if len(foreground) == 0:
            foreground = None

    ref = foreground if foreground is not None else nii_array
    if scale_min is None:
        scale_min = float(np.percentile(ref, 0.5))
    if scale_max is None:
        scale_max = float(np.percentile(ref, 99.5))
    if scale_max - scale_min < 1e-8:
        scale_max = scale_min + 1.0

    scaled = (nii_array - scale_min) / (scale_max - scale_min) * 65535
    scaled = np.clip(scaled, 0, 65535).astype(np.uint16)

    if window_center is None or window_width is None:
        s_min = float(scaled.min())
        s_max = float(scaled.max())
        window_center = (s_min + s_max) / 2.0
        window_width = s_max - s_min

    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S.%f")

    spacing = nii_image.GetSpacing()
    origin = nii_image.GetOrigin()
    direction = nii_image.GetDirection()
    rows, cols = scaled.shape[1], scaled.shape[2]

    for z in range(num_slices):
        slice_data = scaled[z, :, :]
        sop_instance_uid = dicom_generate_uid()
        pos_z = origin[2] + z * spacing[2]

        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
        file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        ds = FileDataset(
            os.path.join(output_dicom_dir, f"slice_{z:04d}.dcm"),
            {}, file_meta=file_meta, preamble=b"\x00" * 128
        )
        ds.is_little_endian = True
        ds.is_implicit_VR = True
        ds.SpecificCharacterSet = "ISO_IR 100"
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        ds.InstanceCreationDate = date_str
        ds.InstanceCreationTime = time_str
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        ds.SOPInstanceUID = sop_instance_uid
        ds.StudyDate = date_str
        ds.SeriesDate = date_str
        ds.StudyTime = time_str
        ds.SeriesTime = time_str
        ds.AccessionNumber = "ACC-001"
        ds.Modality = "MR"
        ds.Manufacturer = "Anonymous"
        ds.InstitutionName = "Anonymous"
        ds.StudyDescription = "MRI Study"
        ds.SeriesDescription = series_desc
        ds.PatientName = patient_id
        ds.PatientID = patient_id
        ds.PatientBirthDate = "19000101"
        ds.PatientSex = "O"
        ds.SliceThickness = round(float(spacing[2]), 6)
        ds.ProtocolName = "MRI"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_ref_uid
        ds.StudyID = "1"
        ds.SeriesNumber = series_number
        ds.AcquisitionNumber = z + 1
        ds.InstanceNumber = z + 1
        ds.ImagePositionPatient = [origin[0], origin[1], pos_z]
        ds.ImageOrientationPatient = [
            direction[0], direction[1], direction[2],
            direction[3], direction[4], direction[5]
        ]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = rows
        ds.Columns = cols
        ds.PixelSpacing = [spacing[1], spacing[0]]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.WindowCenter = round(window_center, 1)
        ds.WindowWidth = round(window_width, 1)
        ds.PixelData = slice_data.tobytes()
        ds.save_as(os.path.join(output_dicom_dir, f"slice_{z:04d}.dcm"))

    print(f"    已保存 {num_slices} 张DICOM到 {output_dicom_dir}")
    return scale_min, scale_max, window_center, window_width


def generate_dicom_for_3patients():
    """为3例患者重新生成DICOM"""
    from pydicom.uid import generate_uid as dicom_generate_uid

    gt_dir = os.path.join(PROJECT_ROOT, "pre-data", "05_normalized")
    pred_dir = os.path.join(PROJECT_ROOT, "output_trans", "pred_nii")
    output_dir = os.path.join(PROJECT_ROOT, "output_dicom_A")

    for pid in PATIENTS:
        pred_path = os.path.join(pred_dir, f"{pid}_pred.nii.gz")
        gt_path = os.path.join(gt_dir, pid, "T1CE.nii.gz")
        mask_path = os.path.join(gt_dir, pid, "body_mask.nii.gz")
        if not os.path.exists(mask_path):
            mask_path = None

        print(f"\n  [{pid}]")
        if not os.path.exists(pred_path):
            print(f"    ✗ pred不存在: {pred_path}")
            continue
        if not os.path.exists(gt_path):
            print(f"    ✗ GT不存在: {gt_path}")
            continue

        patient_out = os.path.join(output_dir, pid)
        dir_1 = os.path.join(patient_out, "1")
        dir_2 = os.path.join(patient_out, "2")

        # 清空旧DICOM
        if os.path.exists(dir_1):
            shutil.rmtree(dir_1)
        if os.path.exists(dir_2):
            shutil.rmtree(dir_2)

        study_uid = dicom_generate_uid()
        series_uid_1 = dicom_generate_uid()
        series_uid_2 = dicom_generate_uid()
        frame_ref_uid = dicom_generate_uid()

        print(f"    转换 GT -> {dir_1}")
        nifti_to_dicom_series(
            nii_path=gt_path, output_dicom_dir=dir_1,
            patient_id=pid, study_uid=study_uid,
            series_uid=series_uid_1, series_number=1,
            frame_ref_uid=frame_ref_uid, series_desc="Series 1",
            mask_nii_path=mask_path
        )

        print(f"    转换 Pred -> {dir_2}")
        nifti_to_dicom_series(
            nii_path=pred_path, output_dicom_dir=dir_2,
            patient_id=pid, study_uid=study_uid,
            series_uid=series_uid_2, series_number=2,
            frame_ref_uid=frame_ref_uid, series_desc="Series 2",
            mask_nii_path=mask_path
        )
        print(f"    ✓ {pid} DICOM完成")

    print("\n✓ Part 2 全部完成")


# ==================== 主函数 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="修复3例body mask过小的患者")
    parser.add_argument('--part', type=str, default='all',
                        choices=['preprocess', 'inference', 'all'],
                        help='运行哪部分: preprocess=预处理, inference=推理+指标+DICOM, all=全部')
    args = parser.parse_args()

    print("=" * 60)
    print(f"  修复3例患者: {PATIENTS}")
    print(f"  body mask 阈值模式: {BODY_MASK_THRESHOLD_MODE}")
    print(f"  运行模式: {args.part}")
    print("=" * 60)

    start_time = time.time()

    if args.part in ('preprocess', 'all'):
        run_preprocessing()

    if args.part in ('inference', 'all'):
        run_inference()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"全部完成! 耗时: {elapsed/60:.1f} 分钟")
    print(f"{'='*60}")
