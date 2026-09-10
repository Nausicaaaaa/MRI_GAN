#!/usr/bin/env python3
"""
将 output_trans/pred_nii 中的预测图与原图（T1CE）进行逐层对比拼接

功能:
1. 读取原始T1CE nii.gz和推理生成的pred nii.gz
2. 逐层左右拼接显示（左: 原图 | 右: 生成图）
3. 支持指定患者或处理全部患者

使用方式:
    # 处理所有患者
    python visualize_pred_vs_gt.py --all_patients

    # 只处理指定患者
    python visualize_pred_vs_gt.py --patient P003600746

    # 自定义路径
    python visualize_pred_vs_gt.py --pred_dir output_trans/pred_nii --gt_dir pre-data/05_normalized --output vis_pred_gt --all_patients
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_nii(path):
    """读取nii.gz文件，返回3D numpy数组"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    nii = nib.load(path)
    return nii.get_fdata().astype(np.float32)


def normalize_for_display(arr, percentile=(1, 99)):
    """归一化到[0,1]用于显示"""
    p_low, p_high = np.percentile(arr, percentile)
    arr = np.clip(arr, p_low, p_high)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min + 1e-8)
    return arr


def visualize_patient(gt_data, pred_data, patient_id, output_dir, figsize=None):
    """逐层左右拼接原图和生成图，保存为图像"""
    os.makedirs(output_dir, exist_ok=True)
    num_slices = gt_data.shape[2]

    # 动态计算figsize：根据数据宽高比确定合适的figure尺寸
    # 保持和check_all_seq一致的每张图高度
    h, w = gt_data.shape[0], gt_data.shape[1]  # 数据维度
    fig_h = 4  # 每张图的显示高度(英寸)
    n = 2  # 2列
    # 计算宽度：每张图按数据宽高比等比缩放
    per_img_w = fig_h * (w / h)  # 单张图的宽度(英寸)
    fig_w = per_img_w * n + 0.5 * n  # 总宽 = 图宽*列数 + 间距

    for z in range(num_slices):
        gt_slice = gt_data[:, :, z].T
        pred_slice = pred_data[:, :, z].T

        gt_norm = normalize_for_display(gt_slice)
        pred_norm = normalize_for_display(pred_slice)

        fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))

        axes[0].imshow(gt_norm, cmap='gray', aspect='equal')
        axes[0].set_title('Original T1CE', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(pred_norm, cmap='gray', aspect='equal')
        axes[1].set_title('Predicted T1CE', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        fig.suptitle(f'{patient_id} | Layer {z:03d}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        out_path = os.path.join(output_dir, f'{patient_id}_layer_{z:03d}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    print(f"  [{patient_id}] 已保存 {num_slices} 张对比图到 {output_dir}")


def process_patient(pred_dir, gt_dir, patient_id, output_dir, figsize):
    """处理单个患者"""
    pred_path = os.path.join(pred_dir, f'{patient_id}_pred.nii.gz')
    gt_path = os.path.join(gt_dir, patient_id, 'T1CE.nii.gz')

    print(f"\n患者: {patient_id}")
    print(f"  原始: {gt_path}")
    print(f"  预测: {pred_path}")

    if not os.path.exists(pred_path):
        print(f"  警告: 预测文件不存在，跳过")
        return False
    if not os.path.exists(gt_path):
        print(f"  警告: 原图文件不存在，跳过")
        return False

    gt_data = read_nii(gt_path)
    pred_data = read_nii(pred_path)

    print(f"  原始shape: {gt_data.shape}, 预测shape: {pred_data.shape}")

    if gt_data.shape != pred_data.shape:
        print(f"  警告: shape不一致！原始{gt_data.shape} vs 预测{pred_data.shape}")
        return False

    patient_out = os.path.join(output_dir, patient_id)
    visualize_patient(gt_data, pred_data, patient_id, patient_out, figsize=figsize)
    return True


def main():
    parser = argparse.ArgumentParser(description='预测图与原图逐层对比拼接')
    parser.add_argument('--pred_dir', '-p', default='output_trans/pred_nii',
                        help='预测nii目录 (默认: output_trans/pred_nii)')
    parser.add_argument('--gt_dir', '-g', default='pre-data/05_normalized',
                        help='原始nii目录 (默认: pre-data/05_normalized)')
    parser.add_argument('--output', '-o', default='vis_pred_gt',
                        help='可视化输出目录 (默认: vis_pred_gt)')
    parser.add_argument('--patient', default=None,
                        help='指定患者ID (如: P003600746)，不指定则处理全部')
    parser.add_argument('--all_patients', action='store_true',
                        help='处理预测目录下的所有患者')
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 6],
                        help='图像尺寸 (宽 高)，默认 12 6')

    args = parser.parse_args()

    figsize = tuple(args.figsize)

    if args.patient:
        process_patient(args.pred_dir, args.gt_dir, args.patient, args.output, figsize)
    elif args.all_patients:
        pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith('_pred.nii.gz')])
        print(f"发现 {len(pred_files)} 个预测文件")
        success_count = 0
        for pred_file in pred_files:
            patient_id = pred_file.replace('_pred.nii.gz', '')
            if process_patient(args.pred_dir, args.gt_dir, patient_id, args.output, figsize):
                success_count += 1
        print(f"\n全部完成! 成功处理 {success_count}/{len(pred_files)} 个患者")
        print(f"输出目录: {args.output}")
    else:
        print("错误: 请指定 --patient 或 --all_patients")
        sys.exit(1)


if __name__ == '__main__':
    main()
