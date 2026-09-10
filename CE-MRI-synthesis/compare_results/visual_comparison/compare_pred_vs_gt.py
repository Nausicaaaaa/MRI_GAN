#!/usr/bin/env python3
"""
推理结果与原始T1CE对比可视化脚本

功能:
1. 读取原始T1CE nii.gz和推理生成的pred nii.gz
2. 逐层左右对比显示（原始 | 预测 | 差值）
3. 支持指定患者ID、指定层号、或均匀采样

使用方式:
    # 对比单个患者所有层
    python CE-MRI-synthesis/compare_results/visual_comparison/compare_pred_vs_gt.py \
        --pred_dir results/CE_MRI_simulate_PCa_1_pix2pix_mulD_fold5-1/pred_nii_concat_checkpoint \
        --gt_dir pre-data/05_normalized \
        --patient P003248663 \
        --output vis_nii_output/compare

    # 只对比指定层
    python compare_pred_vs_gt.py ... --layers 30 40 50 60 70

    # 均匀采样16层
    python compare_pred_vs_gt.py ... --sample 16

    # 对比目录下所有患者
    python compare_pred_vs_gt.py ... --all_patients
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchmetrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import normalized_mutual_info_score as sk_mi


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


def get_slice_indices(num_slices, sample_n=None, layers=None):
    """确定要可视化的层号"""
    if layers is not None:
        return [l for l in layers if 0 <= l < num_slices]
    if sample_n is not None and sample_n < num_slices:
        # 均匀采样，包含首尾
        return np.linspace(0, num_slices - 1, sample_n, dtype=int).tolist()
    return list(range(num_slices))


def scale12bit(img):
    """scale to 12 bit range"""
    new_mean = 2048.
    new_std = 400.
    std = np.std(img)
    if std == 0:
        std = 1e-8
    return np.clip(((img - np.mean(img)) / (std / new_std)) + new_mean, 1e-10, 4095)


def compute_nrmse(gt_slice, pred_slice):
    """归一化均方根误差"""
    mse_val = np.mean((gt_slice - pred_slice) ** 2)
    rmse = np.sqrt(mse_val)
    data_range = gt_slice.max() - gt_slice.min()
    if data_range == 0:
        return 0.0
    return rmse / data_range


def compute_smape(gt_slice, pred_slice):
    """对称平均绝对百分比误差"""
    gt_img = scale12bit(gt_slice)
    pred_img = scale12bit(pred_slice)
    return np.mean(np.fabs(pred_img - gt_img) / (np.fabs(gt_img) + np.fabs(pred_img) + 1e-10))


def compute_logac(gt_slice, pred_slice):
    """对数准确率比"""
    gt_img = scale12bit(gt_slice)
    pred_img = scale12bit(pred_slice)
    return np.mean(np.fabs(np.log(pred_img / gt_img)))


def compute_medsymac(gt_slice, pred_slice):
    """中位数对称准确率"""
    gt_img = scale12bit(gt_slice)
    pred_img = scale12bit(pred_slice)
    return np.exp(np.median(np.fabs(np.log(pred_img / gt_img)))) - 1


def compute_mi(gt_slice, pred_slice, bins=256):
    """直方图互信息 (Histogram Mutual Information)"""
    gt_flat = gt_slice.flatten()
    pred_flat = pred_slice.flatten()
    joint_hist, _, _ = np.histogram2d(gt_flat, pred_flat, bins=bins)
    pxy = joint_hist / float(np.sum(joint_hist))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    return mi


def compute_ms_ssim(gt_slice, pred_slice):
    """多尺度结构相似性指数 (MS-SSIM)"""
    ms_ssim_fn = torchmetrics.functional.image.multiscale_structural_similarity_index_measure
    gt_tensor = torch.from_numpy(gt_slice).unsqueeze(0).unsqueeze(0).float()
    pred_tensor = torch.from_numpy(pred_slice).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        val = ms_ssim_fn(preds=pred_tensor, target=gt_tensor)
    return val.item()


def compute_slice_metrics(gt_slice, pred_slice):
    """计算单层的所有指标"""
    metrics = {
        'ms_ssim': compute_ms_ssim(gt_slice, pred_slice),
        'mi': compute_mi(gt_slice, pred_slice),
        'nrmse': compute_nrmse(gt_slice, pred_slice),
        'smape': compute_smape(gt_slice, pred_slice),
        'logac': compute_logac(gt_slice, pred_slice),
        'medsymac': compute_medsymac(gt_slice, pred_slice),
    }
    return metrics


def visualize_comparison(gt_data, pred_data, slice_indices, patient_id, output_dir,
                         figsize=(16, 5), cmap='gray'):
    """生成对比图：原始 | 预测 | 绝对差值，并返回逐层指标"""
    os.makedirs(output_dir, exist_ok=True)
    slice_metrics_list = []

    for z in slice_indices:
        gt_slice = gt_data[:, :, z].T
        pred_slice = pred_data[:, :, z].T

        # 使用原始数据的窗宽窗位归一化，保证两者对比一致性
        gt_norm = normalize_for_display(gt_slice)
        pred_norm = normalize_for_display(pred_slice)

        # 计算差值图（基于归一化后的数据）
        diff = np.abs(gt_norm - pred_norm)

        # 计算指标（基于原始切片数据，不归一化）
        metrics = compute_slice_metrics(gt_slice, pred_slice)
        metrics['layer'] = z
        slice_metrics_list.append(metrics)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # 原始T1CE
        axes[0].imshow(gt_norm, cmap=cmap, aspect='equal')
        axes[0].set_title('Original T1CE', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # 预测T1CE
        axes[1].imshow(pred_norm, cmap=cmap, aspect='equal')
        axes[1].set_title('Predicted T1CE', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # 差值
        im = axes[2].imshow(diff, cmap='hot', aspect='equal', vmin=0, vmax=1)
        axes[2].set_title('Absolute Difference', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        # 指标文本
        metric_text = (
            f"MS-SSIM: {metrics['ms_ssim']:.4f}\n"
            f"MI:      {metrics['mi']:.4f}\n"
            f"NRMSE:   {metrics['nrmse']:.4f}\n"
            f"SMAPE:   {metrics['smape']:.4f}\n"
            f"LOGAC:   {metrics['logac']:.4f}\n"
            f"MEDSYMAC:{metrics['medsymac']:.4f}"
        )
        axes[3].text(0.1, 0.5, metric_text, fontsize=11, family='monospace',
                     verticalalignment='center')
        axes[3].set_title('Metrics', fontsize=12, fontweight='bold')
        axes[3].axis('off')

        # 全局标题
        mae = np.mean(np.abs(gt_norm - pred_norm))
        fig.suptitle(f'{patient_id} | Layer {z} | Mean Diff: {mae:.4f}',
                     fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = os.path.join(output_dir, f'{patient_id}_layer_{z:03d}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    print(f"  [{patient_id}] 已保存 {len(slice_indices)} 张对比图到 {output_dir}")
    return slice_metrics_list


def save_patient_csv(slice_metrics_list, patient_id, output_dir):
    """保存单个患者的逐层指标CSV"""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'{patient_id}_metrics.csv')
    fieldnames = ['patient_id', 'layer', 'ms_ssim', 'mi', 'nrmse', 'smape', 'logac', 'medsymac']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in slice_metrics_list:
            row = {'patient_id': patient_id, 'layer': m['layer']}
            row.update({k: m[k] for k in fieldnames[2:]})
            writer.writerow(row)
    print(f"  [{patient_id}] 已保存逐层指标CSV: {csv_path}")


def compute_patient_mean(slice_metrics_list):
    """计算患者级别的平均指标"""
    if not slice_metrics_list:
        return {}
    keys = ['ms_ssim', 'mi', 'nrmse', 'smape', 'logac', 'medsymac']
    mean_metrics = {k: float(np.mean([m[k] for m in slice_metrics_list])) for k in keys}
    return mean_metrics


def visualize_patient(pred_dir, gt_dir, patient_id, output_dir,
                      layers=None, sample=None, figsize=(16, 5)):
    """对比单个患者，返回患者平均指标字典"""
    pred_path = os.path.join(pred_dir, f'{patient_id}_pred.nii.gz')
    gt_path = os.path.join(gt_dir, patient_id, 'T1CE.nii.gz')

    print(f"\n患者: {patient_id}")
    print(f"  原始: {gt_path}")
    print(f"  预测: {pred_path}")

    gt_data = read_nii(gt_path)
    pred_data = read_nii(pred_path)

    print(f"  原始shape: {gt_data.shape}, 预测shape: {pred_data.shape}")

    if gt_data.shape != pred_data.shape:
        print(f"  警告: shape不一致！原始{gt_data.shape} vs 预测{pred_data.shape}")
        return None

    num_slices = gt_data.shape[2]
    slice_indices = get_slice_indices(num_slices, sample_n=sample, layers=layers)
    print(f"  总层数: {num_slices}, 待可视化层数: {len(slice_indices)}")

    patient_out = os.path.join(output_dir, patient_id)
    slice_metrics_list = visualize_comparison(gt_data, pred_data, slice_indices, patient_id, patient_out,
                                               figsize=figsize)

    # 保存逐层CSV
    save_patient_csv(slice_metrics_list, patient_id, patient_out)

    # 计算并打印患者平均指标
    mean_metrics = compute_patient_mean(slice_metrics_list)
    print(f"  [{patient_id}] 平均指标 -> MS-SSIM:{mean_metrics['ms_ssim']:.4f} | "
          f"MI:{mean_metrics['mi']:.4f} | NRMSE:{mean_metrics['nrmse']:.4f} | "
          f"SMAPE:{mean_metrics['smape']:.4f} | LOGAC:{mean_metrics['logac']:.4f} | "
          f"MEDSYMAC:{mean_metrics['medsymac']:.4f}")
    return mean_metrics


def save_summary_csv(all_patient_metrics, output_dir):
    """保存所有患者的汇总指标CSV"""
    if not all_patient_metrics:
        return
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'all_patients_summary_metrics.csv')
    fieldnames = ['patient_id', 'ms_ssim', 'mi', 'nrmse', 'smape', 'logac', 'medsymac']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for patient_id, metrics in all_patient_metrics.items():
            row = {'patient_id': patient_id}
            row.update(metrics)
            writer.writerow(row)
        # 添加平均值行
        mean_row = {'patient_id': 'MEAN'}
        for k in fieldnames[1:]:
            mean_row[k] = float(np.mean([m[k] for m in all_patient_metrics.values()]))
        writer.writerow(mean_row)
    print(f"\n汇总指标CSV已保存: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='推理结果与原始T1CE对比可视化')
    parser.add_argument('--pred_dir', '-p', required=True,
                        help='预测nii目录 (如: results/.../pred_nii_concat_checkpoint)')
    parser.add_argument('--gt_dir', '-g', required=True,
                        help='原始nii目录 (如: pre-data/05_normalized)')
    parser.add_argument('--output', '-o', required=True,
                        help='可视化输出目录')
    parser.add_argument('--patient', default=None,
                        help='指定患者ID (如: P003248663)，不指定则处理所有')
    parser.add_argument('--all_patients', action='store_true',
                        help='处理预测目录下的所有患者')
    parser.add_argument('--layers', '-l', nargs='+', type=int, default=None,
                        help='指定层号，空格分隔 (如: 30 40 50)')
    parser.add_argument('--sample', '-s', type=int, default=None,
                        help='均匀采样N层（不指定则处理全部）')
    parser.add_argument('--figsize', nargs=2, type=int, default=[16, 5],
                        help='图像尺寸 (宽 高)，默认 16 5')

    args = parser.parse_args()

    figsize = tuple(args.figsize)
    all_patient_metrics = {}

    if args.patient:
        metrics = visualize_patient(args.pred_dir, args.gt_dir, args.patient, args.output,
                                    layers=args.layers, sample=args.sample, figsize=figsize)
        if metrics:
            all_patient_metrics[args.patient] = metrics
            save_summary_csv(all_patient_metrics, args.output)
    elif args.all_patients:
        pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith('_pred.nii.gz')])
        print(f"发现 {len(pred_files)} 个预测文件")
        for pred_file in pred_files:
            patient_id = pred_file.replace('_pred.nii.gz', '')
            metrics = visualize_patient(args.pred_dir, args.gt_dir, patient_id, args.output,
                                        layers=args.layers, sample=args.sample, figsize=figsize)
            if metrics:
                all_patient_metrics[patient_id] = metrics
        if all_patient_metrics:
            save_summary_csv(all_patient_metrics, args.output)
    else:
        print("错误: 请指定 --patient 或 --all_patients")
        sys.exit(1)

    print(f"\n全部完成! 输出目录: {args.output}")


if __name__ == '__main__':
    main()
