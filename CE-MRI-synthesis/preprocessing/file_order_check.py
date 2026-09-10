#!/usr/bin/env python3
"""
检查原始DICOM数据各序列文件夹中的文件是否按顺序连续命名。

检查规则：
1. 遍历指定数据目录下的所有患者ID文件夹
2. 对每个患者ID下的直接子文件夹（序列文件夹），递归查找其下所有文件
3. 检查所有文件是否都符合 IM{数字} 命名格式
4. 对符合 IM{数字} 的文件，检查数字编号是否从0开始连续递增（如IM0, IM1, IM2, ...）
5. 若存在命名不规范的文件、编号缺失、不连续等情况，将该患者ID记录到badcase.csv

输出格式 (badcase.csv):
    patient_id, subfolder, total_files, im_files, non_im_files, expected_count,
    actual_im_count, missing_indices, non_im_sample, issue_description

使用方法:

    python CE-MRI-synthesis/preprocessing/file_order_check.py --data_root /mnt/data/KASR/Dengsiyi/MRI_GAN/data/C
    python check_dicom_order.py  # 使用默认路径
"""

import argparse
import csv
import os
import re
import sys


def extract_im_number(filename):
    """从文件名中提取IM后的数字编号。"""
    match = re.match(r'^IM(\d+)$', filename)
    if match:
        return int(match.group(1))
    return None


def check_folder_order(folder_path):
    """
    递归检查单个序列文件夹中的所有文件命名是否规范且连续。

    返回: (is_valid, info_dict)
    info_dict 包含:
        total_files, im_files, non_im_files, expected_count,
        actual_im_count, missing_indices, non_im_sample, description
    """
    if not os.path.isdir(folder_path):
        return True, {
            'total_files': 0, 'im_files': 0, 'non_im_files': 0,
            'expected_count': 0, 'actual_im_count': 0,
            'missing_indices': '', 'non_im_sample': '',
            'description': '非文件夹'
        }

    # 递归获取所有文件（不包括目录本身）
    all_files = []
    try:
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                all_files.append(f)
    except OSError as e:
        return False, {
            'total_files': 0, 'im_files': 0, 'non_im_files': 0,
            'expected_count': 0, 'actual_im_count': 0,
            'missing_indices': '', 'non_im_sample': '',
            'description': f"无法读取文件夹: {e}"
        }

    total_files = len(all_files)

    if total_files == 0:
        return True, {
            'total_files': 0, 'im_files': 0, 'non_im_files': 0,
            'expected_count': 0, 'actual_im_count': 0,
            'missing_indices': '', 'non_im_sample': '',
            'description': '无文件'
        }

    im_files = []
    non_im_files = []

    for filename in all_files:
        num = extract_im_number(filename)
        if num is not None:
            im_files.append(num)
        else:
            non_im_files.append(filename)

    im_files.sort()

    # 情况1: 存在非 IM{数字} 命名的文件
    if non_im_files:
        description = f"存在 {len(non_im_files)} 个非IM规范命名文件"
        return False, {
            'total_files': total_files,
            'im_files': len(im_files),
            'non_im_files': len(non_im_files),
            'expected_count': '',
            'actual_im_count': len(im_files),
            'missing_indices': '',
            'non_im_sample': ';'.join(non_im_files[:5]),
            'description': description
        }

    # 情况2: 没有 IM 文件
    if not im_files:
        return True, {
            'total_files': total_files, 'im_files': 0, 'non_im_files': total_files,
            'expected_count': 0, 'actual_im_count': 0,
            'missing_indices': '', 'non_im_sample': ';'.join(non_im_files[:5]),
            'description': '无IM命名文件'
        }

    # 情况3: 检查 IM 编号是否从0开始且连续
    if im_files[0] != 0:
        description = f"IM编号未从0开始，首个编号为IM{im_files[0]}"
        return False, {
            'total_files': total_files,
            'im_files': len(im_files),
            'non_im_files': 0,
            'expected_count': im_files[-1] + 1,
            'actual_im_count': len(im_files),
            'missing_indices': '',
            'non_im_sample': '',
            'description': description
        }

    expected = list(range(len(im_files)))
    if im_files == expected:
        return True, {
            'total_files': total_files,
            'im_files': len(im_files),
            'non_im_files': 0,
            'expected_count': len(im_files),
            'actual_im_count': len(im_files),
            'missing_indices': '',
            'non_im_sample': '',
            'description': '顺序正确'
        }

    # 编号不连续
    full_set = set(range(im_files[-1] + 1))
    actual_set = set(im_files)
    missing = sorted(full_set - actual_set)

    # 检查重复
    duplicates = [num for num in set(im_files) if im_files.count(num) > 1]

    if duplicates:
        description = f"存在重复编号: {', '.join(f'IM{d}' for d in duplicates)}"
    elif missing:
        description = f"缺失编号: {', '.join(f'IM{m}' for m in missing)}"
    else:
        description = "编号不连续（原因未知）"

    return False, {
        'total_files': total_files,
        'im_files': len(im_files),
        'non_im_files': 0,
        'expected_count': im_files[-1] + 1,
        'actual_im_count': len(im_files),
        'missing_indices': ';'.join(str(m) for m in missing),
        'non_im_sample': '',
        'description': description
    }


def check_patient(patient_dir):
    """
    检查单个患者目录下的所有直接子文件夹（序列文件夹）。

    返回: list of dicts, 每个dict包含一个badcase的信息
    """
    badcases = []
    patient_id = os.path.basename(patient_dir)

    try:
        subfolders = [d for d in os.listdir(patient_dir)
                      if os.path.isdir(os.path.join(patient_dir, d))]
    except OSError as e:
        return [{
            'patient_id': patient_id,
            'subfolder': '',
            'total_files': 0, 'im_files': 0, 'non_im_files': 0,
            'expected_count': '', 'actual_im_count': 0,
            'missing_indices': '', 'non_im_sample': '',
            'issue_description': f"无法读取患者目录: {e}"
        }]

    if not subfolders:
        return [{
            'patient_id': patient_id,
            'subfolder': '',
            'total_files': 0, 'im_files': 0, 'non_im_files': 0,
            'expected_count': '', 'actual_im_count': 0,
            'missing_indices': '', 'non_im_sample': '',
            'issue_description': "无子文件夹"
        }]

    for subfolder in sorted(subfolders):
        folder_path = os.path.join(patient_dir, subfolder)
        is_valid, info = check_folder_order(folder_path)

        if not is_valid:
            badcases.append({
                'patient_id': patient_id,
                'subfolder': subfolder,
                'total_files': info['total_files'],
                'im_files': info['im_files'],
                'non_im_files': info['non_im_files'],
                'expected_count': info['expected_count'],
                'actual_im_count': info['actual_im_count'],
                'missing_indices': info['missing_indices'],
                'non_im_sample': info['non_im_sample'],
                'issue_description': info['description']
            })

    return badcases


def main():
    parser = argparse.ArgumentParser(
        description='检查原始DICOM数据各序列文件夹中的文件是否按顺序连续命名'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='/mnt/data/KASR/Dengsiyi/MRI_GAN/data/B',
        help='原始数据根目录，默认: /mnt/data/KASR/Dengsiyi/MRI_GAN/data/B'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='badcase.csv',
        help='输出CSV文件名，默认: badcase.csv'
    )
    parser.add_argument(
        '--patients',
        nargs='+',
        default=None,
        help='指定检查的患者ID列表，如 P000030031 P000047913'
    )

    args = parser.parse_args()

    data_root = args.data_root
    output_file = args.output

    if not os.path.isdir(data_root):
        print(f"错误: 数据目录不存在: {data_root}")
        sys.exit(1)

    # 获取患者列表
    if args.patients:
        patient_ids = args.patients
    else:
        patient_ids = sorted([
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d)) and d.startswith('P')
        ])

    print(f"=" * 60)
    print(f"  DICOM文件顺序检查")
    print(f"=" * 60)
    print(f"数据目录: {data_root}")
    print(f"患者数量: {len(patient_ids)}")
    print(f"输出文件: {output_file}")
    print()

    all_badcases = []
    issue_patients = set()

    for i, pid in enumerate(patient_ids, 1):
        patient_dir = os.path.join(data_root, pid)
        if not os.path.isdir(patient_dir):
            print(f"[{i}/{len(patient_ids)}] {pid} - 目录不存在，跳过")
            continue

        badcases = check_patient(patient_dir)

        if badcases:
            issue_patients.add(pid)
            all_badcases.extend(badcases)
            print(f"[{i}/{len(patient_ids)}] {pid} - 发现问题 ({len(badcases)}个子文件夹)")
            for bc in badcases:
                print(f"    -> {bc['subfolder']}: {bc['issue_description']}")
        else:
            print(f"[{i}/{len(patient_ids)}] {pid} - 顺序正确")

    # 写入CSV
    csv_headers = [
        'patient_id', 'subfolder', 'total_files', 'im_files', 'non_im_files',
        'expected_count', 'actual_im_count', 'missing_indices', 'non_im_sample',
        'issue_description'
    ]

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(all_badcases)

    print()
    print(f"=" * 60)
    print(f"检查完成")
    print(f"=" * 60)
    print(f"总患者数: {len(patient_ids)}")
    print(f"问题患者数: {len(issue_patients)}")
    print(f"问题子文件夹数: {len(all_badcases)}")
    print(f"结果已保存至: {os.path.abspath(output_file)}")


if __name__ == '__main__':
    main()
