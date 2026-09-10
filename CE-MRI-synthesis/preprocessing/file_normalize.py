#!/usr/bin/env python3
"""
DICOM 数据标准化脚本

功能：
1. 递归查找患者模态文件夹下的所有文件
2. 读取 DICOM 元数据（InstanceNumber 优先，其次 SliceLocation）
3. 检查 DICOM 文件是否损坏，损坏的文件记录到错误日志
4. 按元数据排序后将文件重命名为 IM0、IM1、IM2...（扁平化）
5. 删除空的子文件夹

处理策略（安全优先）：
- 所有有效文件先统一移动到根目录的临时文件名，避免重命名冲突
- 再从临时文件名重命名为最终的 IM0、IM1...
- 若模态文件夹已规范（所有文件在根目录且命名连续），则跳过

使用方法：
    python CE-MRI-synthesis/preprocessing/file_normalize.py --data_root /mnt/data/KASR/Dengsiyi/MRI_GAN/data/C
    python CE-MRI-synthesis/preprocessing/file_normalize.py --patients P000047913 P000421260
    python CE-MRI-synthesis/preprocessing/file_normalize.py --dry_run  # 仅预览，不实际修改
"""

import argparse
import csv
import os
import re
import shutil
import sys

import pydicom


def extract_im_number(filename):
    """从文件名中提取 IM 后的数字编号。"""
    match = re.match(r'^IM(\d+)$', filename)
    if match:
        return int(match.group(1))
    return None


def is_folder_normalized(folder_path):
    """
    判断模态文件夹是否已经规范化：
    - 所有文件都在根目录
    - 所有文件都符合 IM{数字} 命名
    - 编号从 0 开始连续递增
    """
    if not os.path.isdir(folder_path):
        return False

    try:
        entries = os.listdir(folder_path)
    except OSError:
        return False

    files = []
    dirs = []
    for entry in entries:
        if entry.startswith('.'):
            continue
        full = os.path.join(folder_path, entry)
        if os.path.isfile(full):
            files.append(entry)
        elif os.path.isdir(full):
            dirs.append(entry)

    # 存在子目录 -> 未扁平化
    if dirs:
        return False

    if not files:
        return True  # 空文件夹视为已规范（无需处理）

    # 检查是否全部符合 IM{数字}
    numbers = []
    for f in files:
        num = extract_im_number(f)
        if num is None:
            return False
        numbers.append(num)

    numbers.sort()
    if numbers[0] != 0:
        return False
    if numbers != list(range(len(numbers))):
        return False

    return True


def read_dicom_meta(file_path):
    """
    读取 DICOM 文件的排序元数据。

    返回: (sort_key, error_type, error_msg)
    - sort_key: 用于排序的元组
    - error_type: None 表示成功，否则为 'corrupted' 或 'not_dicom'
    - error_msg: 错误描述
    """
    try:
        ds = pydicom.dcmread(file_path, stop_before_pixels=True)
    except pydicom.errors.InvalidDicomError:
        return None, 'not_dicom', '非 DICOM 格式文件'
    except Exception as e:
        return None, 'corrupted', f'DICOM 读取失败: {e}'

    inst = getattr(ds, 'InstanceNumber', None)
    slice_loc = getattr(ds, 'SliceLocation', None)

    if inst is not None:
        sort_key = (0, int(inst))
    elif slice_loc is not None:
        sort_key = (1, float(slice_loc))
    else:
        sort_key = (2, 0)

    return sort_key, None, None


def normalize_modality_folder(folder_path, error_records, dry_run=False):
    """
    标准化单个模态文件夹。

    返回: (success, message)
    """
    if not os.path.isdir(folder_path):
        return False, "目录不存在"

    folder_name = os.path.basename(folder_path)

    # 递归查找所有文件（排除隐藏文件）
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.startswith('.'):
                continue
            all_files.append(os.path.join(root, f))

    if not all_files:
        return True, "空文件夹，无需处理"

    # 若已规范则跳过
    if is_folder_normalized(folder_path):
        return True, "已规范，跳过"

    # 读取所有文件的 DICOM 元数据
    valid_files = []   # [(old_path, sort_key), ...]
    errors = []        # [(old_path, error_type, error_msg), ...]

    for f in all_files:
        sort_key, err_type, err_msg = read_dicom_meta(f)
        if err_type is None:
            valid_files.append((f, sort_key))
        else:
            errors.append((f, err_type, err_msg))
            error_records.append({
                'folder': folder_path,
                'file': f,
                'error_type': err_type,
                'error_msg': err_msg
            })

    if not valid_files:
        return False, f"所有文件均无法读取（共 {len(errors)} 个错误）"

    # 按元数据排序
    valid_files.sort(key=lambda x: x[1])

    if dry_run:
        msg = f"[Dry Run] 将处理 {len(valid_files)} 个文件"
        if errors:
            msg += f"，发现 {len(errors)} 个错误文件"
        msg += "；重命名示例:"
        for i in range(min(3, len(valid_files))):
            old = valid_files[i][0]
            msg += f"\n    {old} -> IM{i}"
        if len(valid_files) > 3:
            msg += f"\n    ... 共 {len(valid_files)} 个文件"
        return len(errors) == 0, msg

    # === 实际处理 ===
    # 第一步：将文件移动到根目录的临时名（避免重命名冲突）
    temp_paths = []
    try:
        for i, (old_path, _) in enumerate(valid_files):
            temp_name = f".tmp_normalize_{i:06d}"
            temp_path = os.path.join(folder_path, temp_name)
            shutil.move(old_path, temp_path)
            temp_paths.append(temp_path)
    except Exception as e:
        # 尝试回滚已移动的临时文件
        for tp in temp_paths:
            # 这里不尝试恢复原始位置，因为比较复杂
            pass
        return False, f"临时重命名失败: {e}"

    # 第二步：从临时名重命名为 IM0, IM1...
    try:
        for i, temp_path in enumerate(temp_paths):
            new_name = f"IM{i}"
            new_path = os.path.join(folder_path, new_name)
            shutil.move(temp_path, new_path)
    except Exception as e:
        return False, f"最终重命名失败: {e}"

    # 清理空子目录（自下而上）
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for d in dirs:
            dpath = os.path.join(root, d)
            if os.path.isdir(dpath) and not os.listdir(dpath):
                try:
                    os.rmdir(dpath)
                except OSError:
                    pass

    msg = f"已标准化 {len(valid_files)} 个文件"
    if errors:
        msg += f"，发现 {len(errors)} 个错误文件（已记录）"
    return len(errors) == 0, msg


def process_patient(patient_dir, error_records, dry_run=False):
    """
    处理单个患者目录下的所有模态子文件夹。

    返回: list of (subfolder_name, success, message)
    """
    results = []
    patient_id = os.path.basename(patient_dir)

    try:
        subfolders = sorted([
            d for d in os.listdir(patient_dir)
            if os.path.isdir(os.path.join(patient_dir, d))
        ])
    except OSError as e:
        return [(patient_id, False, f"无法读取患者目录: {e}")]

    for subfolder in subfolders:
        folder_path = os.path.join(patient_dir, subfolder)
        success, msg = normalize_modality_folder(folder_path, error_records, dry_run)
        results.append((subfolder, success, msg))

    return results


def main():
    parser = argparse.ArgumentParser(description='DICOM 数据标准化脚本')
    parser.add_argument(
        '--data_root',
        type=str,
        default='/mnt/data/KASR/Dengsiyi/MRI_GAN/data/A',
        help='原始数据根目录，默认: /mnt/data/KASR/Dengsiyi/MRI_GAN/data/A'
    )
    parser.add_argument(
        '--patients',
        nargs='+',
        default=None,
        help='指定处理的患者ID列表，如 P000030031 P000047913'
    )
    parser.add_argument(
        '--error_log',
        type=str,
        default='normalize_errors.csv',
        help='错误日志文件，默认: normalize_errors.csv'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='仅预览，不实际修改文件'
    )

    args = parser.parse_args()

    data_root = args.data_root
    error_log = args.error_log

    if not os.path.isdir(data_root):
        print(f"错误: 数据目录不存在: {data_root}")
        sys.exit(1)

    # 获取患者列表
    if args.patients:
        patient_ids = args.patients
    else:
        patient_ids = sorted([
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
        ])

    mode_str = "[预览模式]" if args.dry_run else "[实际处理]"
    print(f"{'=' * 60}")
    print(f"  DICOM 数据标准化 {mode_str}")
    print(f"{'=' * 60}")
    print(f"数据目录: {data_root}")
    print(f"患者数量: {len(patient_ids)}")
    print(f"错误日志: {error_log}")
    print()

    all_error_records = []
    processed_count = 0
    skipped_count = 0
    error_count = 0

    for i, pid in enumerate(patient_ids, 1):
        patient_dir = os.path.join(data_root, pid)
        if not os.path.isdir(patient_dir):
            print(f"[{i}/{len(patient_ids)}] {pid} - 目录不存在，跳过")
            continue

        results = process_patient(patient_dir, all_error_records, args.dry_run)

        has_action = False
        for subfolder, success, msg in results:
            if "已规范，跳过" in msg:
                skipped_count += 1
            else:
                has_action = True
                if not success:
                    error_count += 1
                else:
                    processed_count += 1

        if has_action:
            print(f"[{i}/{len(patient_ids)}] {pid}")
            for subfolder, success, msg in results:
                if "已规范，跳过" not in msg:
                    status = "✓" if success else "✗"
                    print(f"    {status} {subfolder}: {msg}")
        else:
            print(f"[{i}/{len(patient_ids)}] {pid} - 全部已规范")

    # 写入错误日志
    if all_error_records:
        with open(error_log, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['folder', 'file', 'error_type', 'error_msg'])
            writer.writeheader()
            writer.writerows(all_error_records)
    elif os.path.exists(error_log):
        os.remove(error_log)

    print()
    print(f"{'=' * 60}")
    print(f"处理完成")
    print(f"{'=' * 60}")
    print(f"总患者数: {len(patient_ids)}")
    print(f"已处理子文件夹: {processed_count}")
    print(f"已跳过（已规范）: {skipped_count}")
    print(f"含错误的子文件夹: {error_count}")
    if all_error_records:
        print(f"错误记录数: {len(all_error_records)}")
        print(f"错误日志: {os.path.abspath(error_log)}")
    else:
        print("未发现错误文件")


if __name__ == '__main__':
    main()
