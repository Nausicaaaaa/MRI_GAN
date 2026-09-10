#!/usr/bin/env python3
"""
三列对比可视化：真实图像(GT) vs output_trans vs output_trans_2

功能:
1. 读取原始T1CE nii.gz、output_trans预测、output_trans_2预测
2. 逐层一排3列拼接: GT | output_trans | output_trans_2
3. 自动取三者的交集患者进行处理
4. 结果保存到 vis_pred_gt_2/

使用方式:
    # 处理所有共有患者
    python visualize_3col_compare.py

    # 只处理指定患者
    python visualize_3col_compare.py --patient P003600746

    # 自定义路径
    python visualize_3col_compare.py --gt_dir pre-data/05_normalized \
        --pred1_dir output_trans/pred_nii \
        --pred2_dir output_trans_2/pred_nii \
        --output vis_pred_gt_2
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


def get_common_patients(gt_dir, pred1_dir, pred2_dir):
    """获取三个目录中共有的患者ID列表"""
    # GT: gt_dir/{patient_id}/T1CE.nii.gz
    gt_patients = set()
    for d in os.listdir(gt_dir):
        t1ce_path = os.path.join(gt_dir, d, 'T1CE.nii.gz')
        if os.path.isfile(t1ce_path):
            gt_patients.add(d)

    # pred1: pred1_dir/{patient_id}_pred.nii.gz
    pred1_patients = set()
    for f in os.listdir(pred1_dir):
        if f.endswith('_pred.nii.gz'):
            pred1_patients.add(f.replace('_pred.nii.gz', ''))

    # pred2: pred2_dir/{patient_id}_pred.nii.gz
    pred2_patients = set()
    for f in os.listdir(pred2_dir):
        if f.endswith('_pred.nii.gz'):
            pred2_patients.add(f.replace('_pred.nii.gz', ''))

    common = sorted(gt_patients & pred1_patients & pred2_patients)
    return common


def visualize_patient_3col(gt_data, pred1_data, pred2_data, patient_id, output_dir, figsize=None):
    """逐层3列拼接: GT | output_trans | output_trans_2"""
    os.makedirs(output_dir, exist_ok=True)
    num_slices = gt_data.shape[2]

    # 动态计算figsize
    h, w = gt_data.shape[0], gt_data.shape[1]
    fig_h = 4
    n = 3
    per_img_w = fig_h * (w / h)
    fig_w = per_img_w * n + 0.5 * n

    for z in range(num_slices):
        gt_slice = gt_data[:, :, z].T
        pred1_slice = pred1_data[:, :, z].T
        pred2_slice = pred2_data[:, :, z].T

        gt_norm = normalize_for_display(gt_slice)
        pred1_norm = normalize_for_display(pred1_slice)
        pred2_norm = normalize_for_display(pred2_slice)

        fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))

        axes[0].imshow(gt_norm, cmap='gray', aspect='equal')
        axes[0].set_title('Ground Truth (T1CE)', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(pred1_norm, cmap='gray', aspect='equal')
        axes[1].set_title('output_trans', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(pred2_norm, cmap='gray', aspect='equal')
        axes[2].set_title('output_trans_2', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        fig.suptitle(f'{patient_id} | Layer {z:03d}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        out_path = os.path.join(output_dir, f'{patient_id}_layer_{z:03d}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    print(f"  [{patient_id}] 已保存 {num_slices} 张三列对比图到 {output_dir}")


def process_patient(gt_dir, pred1_dir, pred2_dir, patient_id, output_dir, figsize):
    """处理单个患者的三列对比"""
    gt_path = os.path.join(gt_dir, patient_id, 'T1CE.nii.gz')
    pred1_path = os.path.join(pred1_dir, f'{patient_id}_pred.nii.gz')
    pred2_path = os.path.join(pred2_dir, f'{patient_id}_pred.nii.gz')

    print(f"\n患者: {patient_id}")
    print(f"  GT:       {gt_path}")
    print(f"  pred_v1:  {pred1_path}")
    print(f"  pred_v2:  {pred2_path}")

    for label, path in [('GT', gt_path), ('output_trans', pred1_path), ('output_trans_2', pred2_path)]:
        if not os.path.exists(path):
            print(f"  警告: {label} 文件不存在 ({path})，跳过")
            return False

    gt_data = read_nii(gt_path)
    pred1_data = read_nii(pred1_path)
    pred2_data = read_nii(pred2_path)

    print(f"  GT shape: {gt_data.shape}, pred_v1 shape: {pred1_data.shape}, pred_v2 shape: {pred2_data.shape}")

    if gt_data.shape != pred1_data.shape:
        print(f"  警告: GT与pred_v1 shape不一致！{gt_data.shape} vs {pred1_data.shape}，跳过")
        return False
    if gt_data.shape != pred2_data.shape:
        print(f"  警告: GT与pred_v2 shape不一致！{gt_data.shape} vs {pred2_data.shape}，跳过")
        return False

    patient_out = os.path.join(output_dir, patient_id)
    visualize_patient_3col(gt_data, pred1_data, pred2_data, patient_id, patient_out, figsize=figsize)
    return True


def main():
    parser = argparse.ArgumentParser(description='三列对比: GT vs output_trans vs output_trans_2')
    parser.add_argument('--gt_dir', default='pre-data/05_normalized',
                        help='原始T1CE nii目录 (默认: pre-data/05_normalized)')
    parser.add_argument('--pred1_dir', default='output_trans/pred_nii',
                        help='output_trans预测nii目录 (默认: output_trans/pred_nii)')
    parser.add_argument('--pred2_dir', default='output_trans_2/pred_nii',
                        help='output_trans_2预测nii目录 (默认: output_trans_2/pred_nii)')
    parser.add_argument('--output', '-o', default='vis_pred_gt_2',
                        help='输出目录 (默认: vis_pred_gt_2)')
    parser.add_argument('--patient', default=None,
                        help='指定患者ID (如: P003600746)，不指定则处理所有共有患者')
    parser.add_argument('--figsize', nargs=2, type=int, default=[18, 6],
                        help='图像尺寸 (宽 高)，默认 18 6')

    args = parser.parse_args()
    figsize = tuple(args.figsize)

    if args.patient:
        patients = [args.patient]
    else:
        patients = get_common_patients(args.gt_dir, args.pred1_dir, args.pred2_dir)
        print(f"发现 {len(patients)} 个共有患者")

    success_count = 0
    for patient_id in patients:
        if process_patient(args.gt_dir, args.pred1_dir, args.pred2_dir,
                           patient_id, args.output, figsize):
            success_count += 1

    print(f"\n全部完成! 成功处理 {success_count}/{len(patients)} 个患者")
    print(f"输出目录: {args.output}")


if __name__ == '__main__':
    main()
