#!/usr/bin/env python3
"""
修复 C 中心 5 例 body mask 异常患者

原因: preprocessing.py step4 使用固定阈值 150 分割 body mask,
      C 中心部分患者 T1 信号偏低 (如秦秀英 p99=138 < 150),
      导致阈值高于几乎所有体素, mask 几乎为空.

修复方案: 使用自适应阈值 (P99 * 0.08, 最低 5), 与 fix_3patients.py 一致.

流程:
  1. 删除旧 body mask
  2. 自适应阈值重新生成 body mask
  3. 重新应用 body mask 到所有序列
  4. 重新归一化
  5. 重新生成 H5
"""

import glob
import os
import shutil
import sys
import time
import traceback

import cv2
import h5py
import numpy as np
import SimpleITK as sitk

# ==================== 配置 ====================
PROJECT_ROOT = "/mnt/data/KASR/Dengsiyi/MRI_GAN"
PATIENTS = ["秦秀英", "李方杰", "祝林", "谢会红", "王述华"]

# C 中心预处理路径
RESAMPLE_DIR = os.path.join(PROJECT_ROOT, "pre-data", "C", "02_resample")
BODY_MASK_DIR = os.path.join(PROJECT_ROOT, "pre-data", "C", "body_masks")
MASKED_DIR = os.path.join(PROJECT_ROOT, "pre-data", "C", "04_masked")
NORM_DIR = os.path.join(PROJECT_ROOT, "pre-data", "C", "05_normalized")
H5_DIR = os.path.join(PROJECT_ROOT, "train-data", "images_ts_C")
VIS_DIR = os.path.join(PROJECT_ROOT, "vis", "mask_C")


# ==================== 形态学操作函数 ====================

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
        print("WARNING: max component size=" + str(np.sum(labeled_img == max_label)) +
              " < min_size=" + str(min_size))
    maxcomponent = (labeled_img == max_label).astype(np.uint8)
    maxcomponent = sitk.GetImageFromArray(maxcomponent)
    return maxcomponent


# ==================== 步骤函数 ====================

def step1_generate_body_mask(pid):
    """重新生成 body mask (自适应阈值)"""
    print("\n  [Step 1] 生成 body mask: {}".format(pid))
    t1_file = os.path.join(RESAMPLE_DIR, pid, "T1_w.nii.gz")
    if not os.path.exists(t1_file):
        print("    x T1 文件不存在: {}".format(t1_file))
        return False

    try:
        img = sitk.ReadImage(t1_file)
        img_array = sitk.GetArrayFromImage(img)
        print("    图像尺寸: {}".format(img.GetSize()))
        nz = img_array[img_array > 0]
        print("    T1 信号: max={:.1f}, mean={:.1f}, nonzero_mean={:.1f}".format(
            img_array.max(), img_array.mean(), nz.mean() if len(nz) > 0 else 0))

        # 自适应阈值
        p99 = np.percentile(img_array, 99)
        ret = max(int(p99 * 0.08), 5)
        print("    自适应阈值: P99={:.1f}, threshold={}".format(p99, ret))
        print("    (原固定阈值 150 对此患者{})".format(
            "过高" if ret < 150 else "合理"))

        new_img = (img_array > ret).astype(np.uint8)

        # 形态学处理流程 (与 preprocessing.py step4 完全一致)
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
        os.makedirs(BODY_MASK_DIR, exist_ok=True)
        save_path = os.path.join(BODY_MASK_DIR, "{}_body.nii.gz".format(pid))
        sitk.WriteImage(new_img, save_path)

        # 统计覆盖率
        mask_arr = sitk.GetArrayFromImage(new_img)
        total_voxels = mask_arr.size
        fg_voxels = mask_arr.sum()
        coverage = fg_voxels / total_voxels * 100
        print("    v body mask 已保存: {}".format(save_path))
        print("    mask 覆盖率: {:.1f}% (之前 <15%)".format(coverage))

        # 保存可视化 (中间层 + 最佳层)
        try:
            os.makedirs(VIS_DIR, exist_ok=True)
            # 清理旧可视化
            old_files = glob.glob(os.path.join(VIS_DIR, "{}_mask_*.png".format(pid)))
            for old_f in old_files:
                os.remove(old_f)

            t1_arr = img_array
            mid = mask_arr.shape[0] // 2
            best_slice = np.argmax(mask_arr.sum(axis=(1, 2)))
            num_slices = mask_arr.shape[0]
            saved_count = 0
            for slice_idx in range(num_slices):
                t1_slice = t1_arr[slice_idx].astype(np.float32)
                mask_slice = mask_arr[slice_idx]
                if t1_slice.max() == 0:
                    continue
                t1_norm = (t1_slice / t1_slice.max() * 255).astype(np.uint8)
                rgb = np.stack([t1_norm, t1_norm, t1_norm], axis=-1)
                contour_mask = mask_slice.astype(np.uint8)
                contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                overlay = rgb.copy()
                overlay[mask_slice > 0] = [255, 0, 0]
                rgb = cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0)
                cv2.drawContours(rgb, contours, -1, (0, 255, 0), 2)
                suffix = ""
                if slice_idx == mid:
                    suffix = "_mid"
                elif slice_idx == best_slice:
                    suffix = "_best"
                vis_path = os.path.join(VIS_DIR, "{}_mask_s{:02d}{}.png".format(pid, slice_idx, suffix))
                cv2.imwrite(vis_path, rgb)
                saved_count += 1
            print("    v 可视化: {} 层 -> {}".format(saved_count, VIS_DIR))
        except Exception as e:
            print("    ! 可视化失败(不影响主流程): {}".format(e))

        return True

    except Exception as e:
        print("    x 生成失败: {}".format(e))
        traceback.print_exc()
        return False


def step2_apply_body_mask(pid):
    """重新应用 body mask 到所有序列"""
    print("\n  [Step 2] 应用 body mask: {}".format(pid))
    patient_reg = os.path.join(RESAMPLE_DIR, pid)
    mask_file = os.path.join(BODY_MASK_DIR, "{}_body.nii.gz".format(pid))
    patient_output = os.path.join(MASKED_DIR, pid)

    if not os.path.exists(patient_reg):
        print("    x 重采样目录不存在: {}".format(patient_reg))
        return False
    if not os.path.exists(mask_file):
        print("    x 身体掩码不存在: {}".format(mask_file))
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
            print("    x 没有 _w.nii.gz 文件")
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
            print("    v {}".format(filename))

        # 复制 mask 到 masked 目录
        sitk.WriteImage(body, os.path.join(patient_output, "body_mask.nii.gz"))
        print("    v body_mask.nii.gz")
        return True

    except Exception as e:
        print("    x 应用失败: {}".format(e))
        traceback.print_exc()
        return False


def step3_normalize(pid):
    """重新归一化 (与 preprocessing.py step6 完全一致)"""
    print("\n  [Step 3] 归一化: {}".format(pid))
    patient_masked = os.path.join(MASKED_DIR, pid)
    patient_output = os.path.join(NORM_DIR, pid)

    if not os.path.exists(patient_masked):
        print("    x masked 目录不存在: {}".format(patient_masked))
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

        # 检查必需文件
        required_files = ["T1_w.nii.gz", "T2_w.nii.gz", "T1CE_w.nii.gz",
                          "DWI_w.nii.gz", "ADC_w.nii.gz", "PreT1_w.nii.gz"]
        for rf in required_files:
            fp = os.path.join(patient_masked, rf)
            if not os.path.exists(fp):
                print("    x 缺失文件: {}".format(rf))
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
        print("    有效切片范围: {} - {}".format(start_point, end_point))

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

            output_file = os.path.join(patient_output, "{}.nii.gz".format(key))
            sitk.WriteImage(img_new, output_file)

        print("    v 归一化完成, 共 {} 层".format(end_point - start_point + 1))
        return True

    except Exception as e:
        print("    x 归一化失败: {}".format(e))
        traceback.print_exc()
        return False


def step4_to_h5(pid):
    """重新生成 H5"""
    print("\n  [Step 4] 生成 H5: {}".format(pid))
    patient_dir = os.path.join(NORM_DIR, pid)
    patient_output = os.path.join(H5_DIR, pid)

    if not os.path.exists(patient_dir):
        print("    x 归一化目录不存在: {}".format(patient_dir))
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
                print("    x 缺失: {}".format(fp))
                return False

        arrays = {}
        for key, fp in files.items():
            if key == "mask":
                arrays[key] = sitk.GetArrayFromImage(sitk.ReadImage(fp, outputPixelType=sitk.sitkUInt8))
            else:
                arrays[key] = sitk.GetArrayFromImage(sitk.ReadImage(fp, outputPixelType=sitk.sitkFloat32))

        slice_num = arrays["mask"].shape[0]
        for layer in range(slice_num):
            h5_file_path = os.path.join(patient_output, "layer_{}.h5".format(layer))
            with h5py.File(h5_file_path, "w") as h5_file:
                for key in arrays:
                    h5_file[key] = arrays[key][layer, :, :]

        print("    v H5 完成, 共 {} 层".format(slice_num))
        return True

    except Exception as e:
        print("    x H5 失败: {}".format(e))
        traceback.print_exc()
        return False


# ==================== 主流程 ====================

def main():
    print("=" * 60)
    print("  修复 C 中心 5 例 body mask 异常患者")
    print("  患者: {}".format(PATIENTS))
    print("  阈值策略: 自适应 (P99 * 0.08, min 5)")
    print("=" * 60)

    results = {}

    for pid in PATIENTS:
        print("\n" + "=" * 50)
        print("  处理患者: {}".format(pid))
        print("=" * 50)

        ok = step1_generate_body_mask(pid)
        if not ok:
            print("\n  x {} body mask 生成失败, 跳过".format(pid))
            results[pid] = "FAIL: mask generation"
            continue

        ok = step2_apply_body_mask(pid)
        if not ok:
            print("\n  x {} 应用 mask 失败, 跳过".format(pid))
            results[pid] = "FAIL: apply mask"
            continue

        ok = step3_normalize(pid)
        if not ok:
            print("\n  x {} 归一化失败, 跳过".format(pid))
            results[pid] = "FAIL: normalize"
            continue

        ok = step4_to_h5(pid)
        if not ok:
            print("\n  x {} H5 生成失败".format(pid))
            results[pid] = "FAIL: H5"
            continue

        print("\n  v {} 修复完成".format(pid))
        results[pid] = "OK"

    # 汇总
    print("\n" + "=" * 60)
    print("  修复结果汇总")
    print("=" * 60)
    for pid, status in results.items():
        mark = "v" if status == "OK" else "x"
        print("  {} {} : {}".format(mark, pid, status))

    ok_count = sum(1 for v in results.values() if v == "OK")
    print("\n  成功: {}/{}".format(ok_count, len(PATIENTS)))

    if ok_count > 0:
        print("\n  * 可视化结果请查看: {}".format(VIS_DIR))
        print("  * 修复后需重新推理以获得更新后的预测结果")
    print("=" * 60)


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print("\n总耗时: {:.1f} 秒".format(elapsed))
